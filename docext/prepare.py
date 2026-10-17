"""Convert raw datasets into cached Document splits.

python -m docext.prepare cord --out data
python -m docext.prepare cord-disjoint --out data
python -m docext.prepare wildreceipt --wildreceipt-path data/wildreceipt_raw --out data
python -m docext.prepare docile --docile-path data/docile_raw --out data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import load_split, save_split, tags_to_entities
from .split import DISTANCE_THRESHOLD


def field_types_of(docs) -> list[str]:
    return sorted({e.label for d in docs for e in tags_to_entities(d.tags)})


def prepare_cord(out: Path) -> None:
    from .data.cord import load_cord

    splits = load_cord()
    types = field_types_of([d for docs in splits.values() for d in docs])
    for name, docs in splits.items():
        save_split(out / "cord" / f"{name}.pkl", "cord", types, docs)
        print(f"cord/{name}: {len(docs)} documents")


def prepare_cord_disjoint(out: Path, seed: int, threshold: float) -> None:
    from .split import cluster_templates, disjoint_split, replication_rate

    splits = {name: load_split(out / "cord" / f"{name}.pkl") for name in ("train", "validation", "test")}
    _, types, _ = splits["train"]
    docs = [d for name in splits for d in splits[name][2]]
    clusters = cluster_templates(docs, threshold)
    for doc, c in zip(docs, clusters):
        doc.cluster = int(c)

    n_train = len(splits["train"][2])
    n_val = len(splits["validation"][2])
    official = {"train": clusters[:n_train], "test": clusters[n_train + n_val :]}
    sizes = {"test": len(splits["test"][2]), "validation": n_val, "train": len(docs)}
    assignment = disjoint_split(clusters, sizes, seed)

    report = {
        "threshold": threshold,
        "templates": int(len(np.unique(clusters))),
        "official_test_replication": replication_rate(official["train"], official["test"]),
        "disjoint_test_replication": replication_rate(
            clusters[assignment["train"]], clusters[assignment["test"]]
        ),
        "sizes": {k: int(len(v)) for k, v in assignment.items()},
    }
    # official splits keep the same cluster ids, so x-shot analysis works on both
    for name, (_, _, split_docs) in splits.items():
        save_split(out / "cord" / f"{name}.pkl", "cord", types, split_docs)
    for name, idx in assignment.items():
        save_split(out / "cord_disjoint" / f"{name}.pkl", "cord", types, [docs[i] for i in idx])
    (out / "cord_disjoint" / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def prepare_wildreceipt(raw: str, out: Path, seed: int, threshold: float, validation_size: int = 100) -> None:
    """WildReceipt's test templates are unseen by design. A validation set of
    seen templates is held out from training, so
    Δ = F1(validation, seen layouts) - F1(test, unseen layouts)."""
    from .data.wildreceipt import load_wildreceipt
    from .split import cluster_templates, replication_rate, seen_template_holdout

    splits, types = load_wildreceipt(raw)
    train, test = splits["train"], splits["test"]
    clusters = cluster_templates(train + test, threshold)
    for doc, c in zip(train + test, clusters):
        doc.cluster = int(c)
    train_clusters = clusters[: len(train)]
    held = set(seen_template_holdout(train_clusters, validation_size, seed).tolist())

    out_splits = {
        "train": [d for i, d in enumerate(train) if i not in held],
        "validation": [d for i, d in enumerate(train) if i in held],
        "test": test,
    }
    keep = np.array([i not in held for i in range(len(train))])
    report = {
        "threshold": threshold,
        "templates": int(len(np.unique(clusters))),
        # near 0 if our key-based clusters agree with the authors' unseen-template test
        "test_replication": replication_rate(train_clusters[keep], clusters[len(train):]),
        "validation_replication": replication_rate(train_clusters[keep], train_clusters[~keep]),
        "sizes": {k: len(v) for k, v in out_splits.items()},
    }
    for name, docs in out_splits.items():
        save_split(out / "wildreceipt" / f"{name}.pkl", "wildreceipt", types, docs)
    (out / "wildreceipt" / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def prepare_docile(docile_path: str, out: Path) -> None:
    from .data.docile import field_types, load_docile

    for split in ("train", "val"):
        docs, stats = load_docile(docile_path, split)
        save_split(out / "docile" / f"{split}.pkl", "docile", field_types(), docs)
        print(f"docile/{split}: {len(docs)} documents, {dict(stats)}")
        (out / "docile" / f"{split}_labelling.json").write_text(json.dumps(dict(stats), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=["cord", "cord-disjoint", "wildreceipt", "docile"])
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--wildreceipt-path", help="extracted wildreceipt folder (class_list.txt, train.txt, test.txt)")
    parser.add_argument("--docile-path", help="unzipped DocILE annotated-trainval folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=DISTANCE_THRESHOLD,
                        help="template distance threshold (calibrate with python -m docext.split)")
    args = parser.parse_args()

    if args.dataset == "cord":
        prepare_cord(args.out)
    elif args.dataset == "cord-disjoint":
        prepare_cord_disjoint(args.out, args.seed, args.threshold)
    elif args.dataset == "wildreceipt":
        if not args.wildreceipt_path:
            parser.error("--wildreceipt-path is required for wildreceipt")
        prepare_wildreceipt(args.wildreceipt_path, args.out, args.seed, args.threshold)
    else:
        if not args.docile_path:
            parser.error("--docile-path is required for docile")
        prepare_docile(args.docile_path, args.out)


if __name__ == "__main__":
    main()
