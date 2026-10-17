"""Score a finished run, optionally with DocILE's official evaluator.

    python -m docext.evaluate runs/<run>                 # entity F1, x-shot subsets
    python -m docext.evaluate runs/<run> --docile-path data/docile_raw

Entity F1 uses exact word-span matches over all field types. The official
evaluator scores KILE fields by location and reports AP and F1 for the
zero-, few- and many-shot layout clusters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import extraction_report, predicted_entities, shot_groups
from .predict import load_predictions as read_predictions
from .schema import LabelSpace, load_split


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = meta["config"]
    label_space = LabelSpace(meta["field_types"])
    _, _, train_docs = load_split(Path(cfg["data"]) / f"{cfg['train_split']}.pkl")
    return meta, label_space, train_docs


def load_predictions(run_dir: Path, split: str) -> dict:
    return read_predictions(run_dir / f"pred_{split}.npz")


def docile_official(docile_path: str, split: str, docs, predictions, label_space, groups: dict) -> dict:
    from docile.dataset import Dataset
    from docile.evaluation import evaluate_dataset

    from .data.docile import to_kile_fields

    kile = {}
    for doc in docs:
        p = predictions[doc.doc_id]
        entities = predicted_entities(p["pred"], label_space)
        scores = [float(np.mean(p["conf"][e.start : e.end])) for e in entities]
        kile[doc.doc_id] = to_kile_fields(doc, entities, scores)

    if not any(kile.values()):  # the evaluator rejects an empty submission
        return {"all": None, "note": "no KILE fields predicted"}
    dataset = Dataset(split, docile_path, load_ocr=False)
    result = evaluate_dataset(dataset, kile, {})
    report = {"all": result.get_metrics("kile")}
    for name, doc_ids in groups.items():  # same x-shot definition as the DocILE toolkit
        report[name] = result.get_metrics("kile", docids=doc_ids)
    return report


def evaluate_run(run_dir: Path, docile_path: str | None = None) -> dict:
    meta, label_space, train_docs = load_run(run_dir)
    cfg = meta["config"]
    metrics = {}
    for split in cfg["eval_splits"]:
        _, _, docs = load_split(Path(cfg["data"]) / f"{split}.pkl")
        preds = load_predictions(run_dir, split)
        metrics[split] = extraction_report(train_docs, docs, preds, label_space)
        if docile_path and meta["dataset"] == "docile":
            groups = shot_groups(train_docs, docs)
            metrics[split]["docile_kile"] = docile_official(
                docile_path, split, docs, preds, label_space, groups
            )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--docile-path", help="unzipped DocILE folder, enables the official evaluator")
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run, args.docile_path), indent=2))


if __name__ == "__main__":
    main()
