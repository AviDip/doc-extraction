"""CORD v2 receipts (Hugging Face: naver-clova-ix/cord-v2) as Documents.

Each annotated line in CORD carries one category, e.g. "total.total_price".
Words flagged is_key (printed keys such as "TOTAL") are tagged O, so only
values are extraction targets, as in DocILE.
"""

from __future__ import annotations

import json

import numpy as np

from ..layout import build_document
from ..schema import Document

HF_NAME = "naver-clova-ix/cord-v2"
SPLITS = ("train", "validation", "test")


def parse_record(doc_id: str, ground_truth: str) -> Document:
    gt = json.loads(ground_truth)
    size = gt["meta"]["image_size"]
    width, height = float(size["width"]), float(size["height"])

    words, boxes, entity_ids, labels = [], [], [], {}
    for eid, line in enumerate(gt["valid_line"]):
        labels[eid] = line["category"]
        for word in line["words"]:
            text = word["text"].strip()
            if not text:
                continue
            q = word["quad"]
            xs, ys = (q["x1"], q["x2"], q["x3"], q["x4"]), (q["y1"], q["y2"], q["y3"], q["y4"])
            words.append(text)
            boxes.append((min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height))
            entity_ids.append(-1 if word.get("is_key") else eid)

    return build_document(
        doc_id, words, np.array(boxes).reshape(-1, 4), np.zeros(len(words)), entity_ids, labels
    )


def load_cord() -> dict[str, list[Document]]:
    from datasets import load_dataset

    out = {}
    for split in SPLITS:
        ds = load_dataset(HF_NAME, split=split).remove_columns("image")
        out[split] = [parse_record(f"{split}_{i:04d}", r["ground_truth"]) for i, r in enumerate(ds)]
    return out
