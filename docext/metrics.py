"""Entity-level extraction scores and layout-familiarity subsets."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .schema import Document, Entity, LabelSpace, tags_to_entities

# DocILE's definition: a test layout cluster is k-shot if k training documents share it
SHOT_RANGES = {"zero-shot": (0, 0), "few-shot": (1, 3), "many-shot": (4, None)}


def predicted_entities(pred_ids: np.ndarray, label_space: LabelSpace) -> list[Entity]:
    return tags_to_entities(label_space.decode(pred_ids))


def match_counts(gold: list[Entity], pred: list[Entity]) -> tuple[int, int, int]:
    """True positives, false positives, false negatives under exact span match."""
    g, p = Counter(e.key() for e in gold), Counter(e.key() for e in pred)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_document_counts(docs: list[Document], predictions: dict, label_space: LabelSpace) -> dict:
    """{doc_id: (tp, fp, fn)}."""
    return {
        d.doc_id: match_counts(
            tags_to_entities(d.tags), predicted_entities(predictions[d.doc_id]["pred"], label_space)
        )
        for d in docs
    }


def micro_f1(counts: dict, doc_ids=None) -> dict:
    ids = counts.keys() if doc_ids is None else doc_ids
    tp, fp, fn = (sum(counts[i][k] for i in ids) for k in range(3))
    return prf(tp, fp, fn)


def shot_groups(train_docs: list[Document], eval_docs: list[Document]) -> dict[str, list[str]]:
    """Split evaluation documents by how many training documents share their
    layout cluster. Empty if clusters are unknown."""
    if any(d.cluster is None for d in train_docs + eval_docs):
        return {}
    seen = Counter(d.cluster for d in train_docs)
    groups = {name: [] for name in SHOT_RANGES}
    for d in eval_docs:
        k = seen[d.cluster]
        for name, (low, high) in SHOT_RANGES.items():
            if low <= k and (high is None or k <= high):
                groups[name].append(d.doc_id)
    return {name: ids for name, ids in groups.items() if ids}


def extraction_report(train_docs, eval_docs, predictions, label_space) -> dict:
    counts = per_document_counts(eval_docs, predictions, label_space)
    report = {"all": micro_f1(counts) | {"documents": len(eval_docs)}}
    for name, ids in shot_groups(train_docs, eval_docs).items():
        report[name] = micro_f1(counts, ids) | {"documents": len(ids)}
    if "many-shot" in report and "zero-shot" in report:
        report["delta_many_minus_zero"] = report["many-shot"]["f1"] - report["zero-shot"]["f1"]
    return report
