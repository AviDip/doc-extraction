"""WildReceipt (Sun et al., 2021) as Documents.

Public download, no registration: 1,268 training and 472 test receipts, with
test templates unseen in training by construction. Each OCR box carries one
of 25 classes: a value or key for 12 fields, plus Others. Value boxes become
entities; keys and Others are tagged O. Each box is one word, since the
released text has its spaces removed ("THANKYOU").

    wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..layout import build_document
from ..schema import Document

VALUE_SUFFIX = "_value"


def read_classes(path: Path) -> dict[int, str]:
    classes = {}
    for line in path.read_text().splitlines():
        if line.strip():
            idx, name = line.split(maxsplit=1)
            classes[int(idx)] = name.strip()
    return classes


def field_types(classes: dict[int, str]) -> list[str]:
    return sorted(n[: -len(VALUE_SUFFIX)] for n in classes.values() if n.endswith(VALUE_SUFFIX))


def parse_record(doc_id: str, record: dict, classes: dict[int, str]) -> Document:
    width, height = float(record["width"]), float(record["height"])
    words, boxes, entity_ids, labels = [], [], [], {}
    for i, box in enumerate(record["annotations"]):
        text = box["text"].strip()
        if not text:
            continue
        xs, ys = box["box"][0::2], box["box"][1::2]
        words.append(text)
        boxes.append((min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height))
        name = classes.get(int(box["label"]), "")
        if name.endswith(VALUE_SUFFIX):
            labels[i] = name[: -len(VALUE_SUFFIX)]
            entity_ids.append(i)
        else:
            entity_ids.append(-1)
    return build_document(
        doc_id, words, np.array(boxes).reshape(-1, 4), np.zeros(len(words)), entity_ids, labels
    )


def load_wildreceipt(root: str | Path) -> tuple[dict[str, list[Document]], list[str]]:
    root = Path(root)
    classes = read_classes(root / "class_list.txt")
    splits = {}
    for split in ("train", "test"):
        lines = (root / f"{split}.txt").read_text().splitlines()
        splits[split] = [
            parse_record(f"{split}_{i:04d}", json.loads(line), classes) for i, line in enumerate(lines) if line
        ]
    return splits, field_types(classes)
