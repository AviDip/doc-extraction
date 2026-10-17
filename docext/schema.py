"""Shared data structures: the word-level document and the BIO label space."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

OUTSIDE = "O"


@dataclass
class Document:
    """A document as a sequence of OCR words in reading order.

    boxes are (x0, y0, x1, y1) relative to their page, in [0, 1].
    tags are BIO strings ("B-total", "I-total", "O").
    lines give the text-line index of each word, used for key detection
    and number merging.
    """

    doc_id: str
    words: list[str]
    boxes: np.ndarray
    pages: np.ndarray
    lines: np.ndarray
    tags: list[str]
    cluster: int | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.words)

    def copy(self, **changes) -> "Document":
        return replace(self, **changes)


@dataclass(frozen=True)
class Entity:
    """A contiguous run of words carrying one field type."""

    label: str
    start: int
    end: int  # exclusive

    def key(self) -> tuple[str, int, int]:
        return self.label, self.start, self.end


class LabelSpace:
    """Maps field types to BIO label ids and back."""

    def __init__(self, field_types: list[str]):
        self.field_types = list(field_types)
        self.labels = [OUTSIDE] + [f"{p}-{f}" for f in self.field_types for p in ("B", "I")]
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.field_to_index = {f: i for i, f in enumerate(self.field_types)}

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, tags: list[str]) -> np.ndarray:
        return np.array([self.label_to_id[t] for t in tags], dtype=np.int64)

    def decode(self, ids) -> list[str]:
        return [self.labels[int(i)] for i in ids]

    def begin_ids(self) -> list[int]:
        """Label id of B-f for every field f, in field order."""
        return [self.label_to_id[f"B-{f}"] for f in self.field_types]

    def inside_ids(self) -> list[int]:
        return [self.label_to_id[f"I-{f}"] for f in self.field_types]


def tags_to_entities(tags: list[str]) -> list[Entity]:
    """Group BIO tags into entities. An I- tag that does not continue the
    current entity starts a new one, matching the usual lenient decoding."""
    entities: list[Entity] = []
    label, start = None, 0
    for i, tag in enumerate(tags + [OUTSIDE]):
        prefix, _, field_type = tag.partition("-")
        continues = prefix == "I" and field_type == label
        if label is not None and not continues:
            entities.append(Entity(label, start, i))
            label = None
        if prefix in ("B", "I") and not continues:
            label, start = field_type, i
    return entities


def entities_to_tags(entities: list[Entity], n_words: int) -> list[str]:
    tags = [OUTSIDE] * n_words
    for e in entities:
        tags[e.start] = f"B-{e.label}"
        for i in range(e.start + 1, e.end):
            tags[i] = f"I-{e.label}"
    return tags


# ---------------------------------------------------------------- storage


def save_split(path: str | Path, dataset: str, field_types: list[str], docs: list[Document]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"dataset": dataset, "field_types": field_types, "docs": docs}, f)


def load_split(path: str | Path) -> tuple[str, list[str], list[Document]]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["dataset"], blob["field_types"], blob["docs"]
