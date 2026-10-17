"""Turn documents into model batches and model outputs back into words.

Documents longer than the encoder's 512-token limit are split into
overlapping windows. Each word is predicted from its first sub-word token;
where windows overlap, a word's probabilities are averaged. Pooling back to
words lets the structural loss see the whole document at once.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch

from .constraints import Equation, equation_status
from .numbers import word_values
from .schema import Document, LabelSpace, tags_to_entities

MAX_LENGTH = 512  # encoder limit, including the two special tokens
OVERLAP = 128  # tokens shared by consecutive windows
IGNORE = -100


def tokenize_words(tokenizer, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and the word index of each token, without special tokens."""
    if "boxes" in inspect.signature(tokenizer.__call__).parameters:  # LayoutLMv3 tokenizer
        enc = tokenizer(words, boxes=[[0, 0, 0, 0]] * len(words), add_special_tokens=False)
    else:
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = np.array([-1 if w is None else w for w in enc.word_ids()], dtype=np.int64)
    return np.array(enc["input_ids"], dtype=np.int64), word_ids


def window_spans(n_tokens: int, size: int = MAX_LENGTH - 2, overlap: int = OVERLAP) -> list[tuple[int, int]]:
    if n_tokens <= size:
        return [(0, n_tokens)]
    stride = size - overlap
    starts = list(range(0, n_tokens - size, stride)) + [n_tokens - size]
    return [(s, s + size) for s in starts]


class Collator:
    """Builds a batch of windows from a list of documents.

    max_windows caps the windows taken from one document during training
    (a random contiguous run); such a document is marked partial and is left
    out of the document-level loss terms.
    """

    def __init__(
        self,
        tokenizer,
        label_space: LabelSpace,
        equations: list[Equation] = (),
        max_windows: int | None = None,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.label_space = label_space
        self.equations = list(equations)
        self.max_windows = max_windows
        self.rng = np.random.default_rng(seed)
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id

    def __call__(self, docs: list[Document]) -> dict:
        windows, word_doc, values, eq_scale, eq_active, doc_active = [], [], [], [], [], []
        offset = 0
        for d, doc in enumerate(docs):
            ids, token_word = tokenize_words(self.tokenizer, doc.words)
            first = np.zeros(len(ids), dtype=bool)
            if len(ids):
                first[1:] = token_word[1:] != token_word[:-1]
                first[0] = True
            labels = self.label_space.encode(doc.tags)
            spans = window_spans(len(ids))
            complete = True
            if self.max_windows and len(spans) > self.max_windows:
                start = int(self.rng.integers(len(spans) - self.max_windows + 1))
                spans, complete = spans[start : start + self.max_windows], False

            for s, e in spans:
                w = token_word[s:e]
                is_first = first[s:e]
                windows.append(
                    {
                        "input_ids": [self.cls, *ids[s:e], self.sep],
                        "bbox": np.vstack([np.zeros((1, 4)), doc.boxes[w], np.zeros((1, 4))]),
                        "labels": [IGNORE, *np.where(is_first, labels[w], IGNORE), IGNORE],
                        "word_index": [-1, *np.where(is_first, w + offset, -1), -1],
                    }
                )

            vals = word_values(doc)
            status = equation_status(tags_to_entities(doc.tags), vals, self.equations)
            word_doc.append(np.full(len(doc), d))
            values.append(vals)
            eq_scale.append([scale for _, _, scale in status])
            eq_active.append([ok for _, ok, _ in status])
            doc_active.append(complete)
            offset += len(doc)

        return self._pad(windows) | {
            "word_doc": torch.from_numpy(np.concatenate(word_doc)).long(),
            "values": torch.from_numpy(np.concatenate(values)).float(),
            "eq_scale": torch.tensor(eq_scale, dtype=torch.float32).reshape(len(docs), len(self.equations)),
            "eq_active": torch.tensor(eq_active, dtype=torch.bool).reshape(len(docs), len(self.equations)),
            "doc_active": torch.tensor(doc_active, dtype=torch.bool),
            "n_words": offset,
        }

    def _pad(self, windows: list[dict]) -> dict:
        n, length = len(windows), max(len(w["input_ids"]) for w in windows)
        input_ids = torch.full((n, length), self.pad, dtype=torch.long)
        attention = torch.zeros((n, length), dtype=torch.long)
        bbox = torch.zeros((n, length, 4), dtype=torch.float32)
        labels = torch.full((n, length), IGNORE, dtype=torch.long)
        word_index = torch.full((n, length), -1, dtype=torch.long)
        for i, w in enumerate(windows):
            k = len(w["input_ids"])
            input_ids[i, :k] = torch.tensor(w["input_ids"])
            attention[i, :k] = 1
            bbox[i, :k] = torch.from_numpy(w["bbox"]).float()
            labels[i, :k] = torch.tensor(w["labels"])
            word_index[i, :k] = torch.tensor(w["word_index"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "bbox": bbox,
            "labels": labels,
            "word_index": word_index,
        }


def pool_words(
    per_token: torch.Tensor, word_index: torch.Tensor, n_words: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average token-level vectors (windows, length, K) into word-level
    vectors (n_words, K). Returns the vectors and how many windows saw each word."""
    mask = word_index >= 0
    idx = word_index[mask]
    rows = per_token[mask]
    sums = rows.new_zeros(n_words, rows.shape[-1]).index_add_(0, idx, rows)
    counts = rows.new_zeros(n_words).index_add_(0, idx, torch.ones_like(idx, dtype=rows.dtype))
    return sums / counts.clamp_min(1)[:, None], counts
