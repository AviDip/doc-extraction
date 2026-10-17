"""Batched inference: word-level labels, confidences and document embeddings."""

from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from .encoding import Collator, pool_words
from .schema import Document

EVAL_WINDOWS = 32  # windows per forward pass at inference


def amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device):
    dtype = amp_dtype(device)
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


@torch.no_grad()
def predict(
    model, collator: Collator, docs: list[Document], device: torch.device, batch_size: int = 8
) -> dict:
    """Returns {doc_id: {"pred": label ids, "conf": max probability per word,
    "emb": mean final-layer state over the document's words}}."""
    model.eval()
    out = {}
    loader = DataLoader(docs, batch_size=batch_size, shuffle=False, collate_fn=collator)
    for start, batch in zip(range(0, len(docs), batch_size), loader):
        probs, hidden = _forward_in_chunks(model, batch, device)
        word_doc = batch["word_doc"].to(device)
        n_docs = batch["doc_active"].shape[0]
        emb = hidden.new_zeros(n_docs, hidden.shape[1]).index_add_(0, word_doc, hidden)
        emb /= torch.bincount(word_doc, minlength=n_docs).clamp_min(1)[:, None]
        conf, pred = probs.max(dim=1)
        for d, doc in enumerate(docs[start : start + batch_size]):
            m = word_doc == d
            out[doc.doc_id] = {
                "pred": pred[m].cpu().numpy().astype(np.int16),
                "conf": conf[m].float().cpu().numpy().astype(np.float16),
                "emb": emb[d].float().cpu().numpy().astype(np.float16),
            }
    return out


def _forward_in_chunks(model, batch: dict, device: torch.device):
    n_words = batch["n_words"]
    prob_rows, hidden_rows, index_rows = [], [], []
    for s in range(0, batch["input_ids"].shape[0], EVAL_WINDOWS):
        part = {
            k: batch[k][s : s + EVAL_WINDOWS].to(device)
            for k in ("input_ids", "attention_mask", "bbox", "word_index")
        }
        with autocast(device):
            logits, hidden = model(part["input_ids"], part["attention_mask"], part["bbox"], hidden=True)
        mask = part["word_index"] >= 0
        prob_rows.append(logits.float().softmax(-1)[mask])
        hidden_rows.append(hidden.float()[mask])
        index_rows.append(part["word_index"][mask])
    idx = torch.cat(index_rows)[:, None]  # pool_words expects (windows, length)
    probs, counts = pool_words(torch.cat(prob_rows)[:, None], idx, n_words)
    hidden, _ = pool_words(torch.cat(hidden_rows)[:, None], idx, n_words)
    # words that produced no tokens (rare unicode) default to "O"
    probs[counts == 0, 0] = 1.0
    return probs, hidden


FIELDS = ("pred", "conf", "emb")


def save_predictions(path, predictions: dict) -> None:
    """Store as .npz (plain arrays), which is safe to share: unlike pickle,
    loading it cannot execute code."""
    arrays = {f"{doc_id}/{k}": v[k] for doc_id, v in predictions.items() for k in FIELDS}
    np.savez_compressed(path, **arrays)


def load_predictions(path) -> dict:
    out: dict = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            doc_id, _, field = key.rpartition("/")
            out.setdefault(doc_id, {})[field] = data[key]
    return out
