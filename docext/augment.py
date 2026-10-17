"""Training-time augmentation: page jitter and key-phrase swapping.

Neither dataset annotates keys (the printed text naming a field, such as
"Total:"), so a key is taken to be the unlabelled words immediately left of
a field on the same line. Collecting these across the training set gives a
per-field vocabulary of real key phrases ("Total", "Amount due", "Celkem").
Swapping a key for another phrase of the same field stops the model from
relying on one exact wording.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .layout import jitter_boxes, union_box
from .schema import OUTSIDE, Document, tags_to_entities

MAX_KEY_WORDS = 4  # keys longer than this are rare and usually a mis-detection
MIN_KEY_COUNT = 2  # a phrase must recur to enter the vocabulary


def find_key_spans(doc: Document) -> list[tuple[int, int, str]]:
    """(start, end, field_type) of the key phrase preceding each field."""
    spans = []
    for entity in tags_to_entities(doc.tags):
        line = doc.lines[entity.start]
        start = entity.start
        while (
            start > 0
            and entity.start - start < MAX_KEY_WORDS
            and doc.lines[start - 1] == line
            and doc.tags[start - 1] == OUTSIDE
            and not any(c.isdigit() for c in doc.words[start - 1])
        ):
            start -= 1
        phrase = doc.words[start : entity.start]
        if phrase and any(c.isalpha() for w in phrase for c in w):
            spans.append((start, entity.start, entity.label))
    return spans


def key_vocabulary(docs: list[Document]) -> dict[str, list[tuple[str, ...]]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc in docs:
        for start, end, field_type in find_key_spans(doc):
            counts[field_type][tuple(doc.words[start:end])] += 1
    return {f: [phrase for phrase, n in c.most_common() if n >= MIN_KEY_COUNT] for f, c in counts.items()}


def swap_keys(doc: Document, vocab: dict, prob: float, rng: np.random.Generator) -> Document:
    """Replace each detected key, with probability `prob`, by another phrase
    of the same field. New words share the old key's box, split by length."""
    words, boxes = list(doc.words), doc.boxes
    pages, lines, tags = doc.pages, doc.lines, list(doc.tags)
    for start, end, field_type in reversed(find_key_spans(doc)):
        options = [p for p in vocab.get(field_type, []) if list(p) != doc.words[start:end]]
        if not options or rng.random() >= prob:
            continue
        phrase = list(options[rng.integers(len(options))])
        new_boxes = _split_box(union_box(boxes[start:end]), phrase)
        n = len(phrase)
        words[start:end] = phrase
        tags[start:end] = [OUTSIDE] * n
        boxes = np.concatenate([boxes[:start], new_boxes, boxes[end:]])
        pages = np.concatenate([pages[:start], np.full(n, pages[start]), pages[end:]])
        lines = np.concatenate([lines[:start], np.full(n, lines[start]), lines[end:]])
    return doc.copy(words=words, boxes=boxes, pages=pages, lines=lines, tags=tags)


def _split_box(box: tuple, words: list[str]) -> np.ndarray:
    x0, y0, x1, y1 = box
    widths = np.array([len(w) + 1 for w in words], dtype=np.float32)
    edges = x0 + (x1 - x0) * np.concatenate([[0], np.cumsum(widths) / widths.sum()])
    return np.array([(edges[i], y0, edges[i + 1], y1) for i in range(len(words))], dtype=np.float32)


class Augmenter:
    """Applies the configured augmentations to a training document."""

    def __init__(self, jitter: float, key_swap: float, vocab: dict | None = None):
        self.jitter = jitter
        self.key_swap = key_swap
        self.vocab = vocab or {}

    def __call__(self, doc: Document, rng: np.random.Generator) -> Document:
        if self.key_swap > 0 and self.vocab:
            doc = swap_keys(doc, self.vocab, self.key_swap, rng)
        if self.jitter > 0:
            doc = doc.copy(boxes=jitter_boxes(doc.boxes, doc.pages, self.jitter, rng))
        return doc
