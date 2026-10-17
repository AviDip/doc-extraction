"""DocILE (github.com/rossumai/docile) as Documents, and predictions back
into the official evaluator's format.

A word is assigned to an annotated field when at least half of its area lies
inside the field box, the rule used by the DocILE NER baselines. A word that
qualifies for several fields takes the one with the largest overlap; the
number of such conflicts is reported so its effect can be stated.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..layout import build_document, union_box
from ..schema import Document, Entity

MIN_OVERLAP = 0.5


def field_types() -> list[str]:
    from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES

    return list(KILE_FIELDTYPES) + list(LIR_FIELDTYPES)


def _page_words(document, page: int):
    try:
        return document.ocr.get_all_words(page, snapped=True)
    except ValueError:  # snapped boxes not cached and no page image available
        return document.ocr.get_all_words(page, snapped=False)


def _overlap_ratio(words: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """Share of each word's area inside each field box, shape (words, fields)."""
    left = np.maximum(words[:, None, 0], fields[None, :, 0])
    top = np.maximum(words[:, None, 1], fields[None, :, 1])
    right = np.minimum(words[:, None, 2], fields[None, :, 2])
    bottom = np.minimum(words[:, None, 3], fields[None, :, 3])
    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area = np.clip((words[:, 2] - words[:, 0]) * (words[:, 3] - words[:, 1]), 1e-9, None)
    return inter / area[:, None]


def convert_document(document) -> tuple[Document, Counter]:
    stats: Counter = Counter()
    annotation = document.annotation
    fields = annotation.fields + annotation.li_fields
    labels = {i: f.fieldtype for i, f in enumerate(fields)}

    words, boxes, pages, entity_ids = [], [], [], []
    for page in range(document.page_count):
        page_words = [w for w in _page_words(document, page) if w.text and w.text.strip()]
        if not page_words:
            continue
        word_boxes = np.array([w.bbox.to_tuple() for w in page_words], dtype=np.float32)
        field_idx = [i for i, f in enumerate(fields) if f.page == page]
        assigned = np.full(len(page_words), -1)
        if field_idx:
            field_boxes = np.array([fields[i].bbox.to_tuple() for i in field_idx], dtype=np.float32)
            ratio = _overlap_ratio(word_boxes, field_boxes)
            hits = ratio >= MIN_OVERLAP
            for w in np.flatnonzero(hits.any(axis=1)):
                candidates = {labels[field_idx[j]] for j in np.flatnonzero(hits[w])}
                stats["conflicting_words"] += len(candidates) > 1
                assigned[w] = field_idx[int(ratio[w].argmax())]
        words += [w.text.strip() for w in page_words]
        boxes.append(word_boxes)
        pages += [page] * len(page_words)
        entity_ids += assigned.tolist()

    covered = set(entity_ids)
    stats["fields"] += len(fields)
    stats["fields_without_words"] += sum(i not in covered for i in labels)
    stats["labelled_words"] += sum(e >= 0 for e in entity_ids)
    stats["words"] += len(words)

    doc = build_document(
        document.docid,
        words,
        np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        np.array(pages),
        entity_ids,
        labels,
        cluster=annotation.cluster_id,
    )
    return doc, stats


def load_docile(path: str, split: str) -> tuple[list[Document], Counter]:
    from docile.dataset import CachingConfig, Dataset
    from tqdm import tqdm

    dataset = Dataset(split, path, load_annotations=False, load_ocr=False, cache_images=CachingConfig.OFF)
    docs, stats = [], Counter()
    for document in tqdm(dataset, desc=f"DocILE {split}"):
        with document:
            doc, doc_stats = convert_document(document)
        docs.append(doc)
        stats.update(doc_stats)
    return docs, stats


# ------------------------------------------------------- official evaluation


def to_kile_fields(doc: Document, entities: list[Entity], scores: list[float]):
    """Convert predicted entities to DocILE Fields (KILE field types only).

    The evaluator matches fields by location, so each entity becomes the
    union box of its words on the page where it starts.
    """
    from docile.dataset import KILE_FIELDTYPES, BBox, Field

    kile = set(KILE_FIELDTYPES)
    out = []
    for entity, score in zip(entities, scores):
        if entity.label not in kile:
            continue
        idx = np.arange(entity.start, entity.end)
        idx = idx[doc.pages[idx] == doc.pages[entity.start]]
        out.append(
            Field(
                bbox=BBox(*union_box(doc.boxes[idx])),
                page=int(doc.pages[entity.start]),
                score=float(score),
                fieldtype=entity.label,
            )
        )
    return out
