"""Box geometry: reading order, line grouping and page-level jitter."""

from __future__ import annotations

import numpy as np

from .schema import OUTSIDE, Document


def reading_order(boxes: np.ndarray, pages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order words page by page, top to bottom, then left to right.

    Words are grouped into a line when their vertical centres lie within half
    a typical word height of the line's running centre. OCR engines often
    emit words grouped by block, which mixes rows; this restores row order.

    Returns the permutation and the line id of each word in the new order.
    """
    order: list[int] = []
    line_ids: list[int] = []
    line = 0
    centres = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    for page in np.unique(pages):
        idx = np.flatnonzero(pages == page)
        if idx.size == 0:
            continue
        tolerance = 0.5 * max(float(np.median(heights[idx])), 1e-6)
        current: list[int] = []
        current_centre = 0.0
        for i in idx[np.argsort(centres[idx], kind="stable")]:
            if current and abs(centres[i] - current_centre) > tolerance:
                order.extend(sorted(current, key=lambda j: boxes[j, 0]))
                line_ids.extend([line] * len(current))
                line, current = line + 1, []
            current.append(i)
            current_centre = float(np.mean(centres[current]))
        order.extend(sorted(current, key=lambda j: boxes[j, 0]))
        line_ids.extend([line] * len(current))
        line += 1
    return np.array(order, dtype=np.int64), np.array(line_ids, dtype=np.int64)


def build_document(
    doc_id: str,
    words: list[str],
    boxes: np.ndarray,
    pages: np.ndarray,
    entity_ids: list[int],
    entity_labels: dict[int, str],
    cluster: int | None = None,
) -> Document:
    """Create a Document with words sorted into reading order.

    entity_ids[i] is the annotated field instance word i belongs to (-1 for
    none). BIO tags are assigned after sorting, so B always marks the first
    word of a field in reading order.
    """
    boxes = np.clip(np.asarray(boxes, dtype=np.float32), 0.0, 1.0)
    pages = np.asarray(pages, dtype=np.int64)
    order, lines = reading_order(boxes, pages)

    tags, seen, last_pos = [], set(), {}
    fragmented = set()
    for pos, i in enumerate(order):
        eid = entity_ids[i]
        if eid < 0:
            tags.append(OUTSIDE)
            continue
        if eid in seen and last_pos[eid] != pos - 1:
            fragmented.add(eid)  # field broken up by other words in reading order
        tags.append(("I-" if eid in seen else "B-") + entity_labels[eid])
        seen.add(eid)
        last_pos[eid] = pos

    return Document(
        doc_id=doc_id,
        words=[words[i] for i in order],
        boxes=boxes[order],
        pages=pages[order],
        lines=lines,
        tags=tags,
        cluster=cluster,
        meta={"fields": len(seen), "fragmented_fields": len(fragmented)},
    )


def jitter_boxes(
    boxes: np.ndarray, pages: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """Randomly rescale and shift each page by up to `magnitude` of its size.

    The transform is shared by all words on a page, so relative layout (rows,
    columns, key-value alignment) is preserved while absolute positions, the
    shortcut a model can memorise on repeated templates, never repeat.
    """
    if magnitude <= 0:
        return boxes
    out = boxes.astype(np.float32, copy=True)
    for page in np.unique(pages):
        mask = pages == page
        b = 0.5 + (out[mask] - 0.5) * rng.uniform(1 - magnitude, 1 + magnitude)
        low, high = b[:, :2].min(axis=0), b[:, 2:].max(axis=0)
        t_min = np.maximum(-magnitude, -low)
        t_max = np.maximum(t_min, np.minimum(magnitude, 1 - high))
        shift = rng.uniform(t_min, t_max)
        b[:, [0, 2]] += shift[0]
        b[:, [1, 3]] += shift[1]
        out[mask] = np.clip(b, 0.0, 1.0)
    return out


def union_box(boxes: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(boxes[:, 0].min()),
        float(boxes[:, 1].min()),
        float(boxes[:, 2].max()),
        float(boxes[:, 3].max()),
    )
