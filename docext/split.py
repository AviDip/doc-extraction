"""Template-disjoint splits for datasets without published layout clusters.

Following Laatiri et al. (2023), documents that come from the same source
share a template. A receipt's source is identified by its header (shop name
and address in the first lines) and by its key wording ("PB1" vs "Tax").
Generic content, such as which field types appear, is left out: almost every
receipt has a total, so it would make unrelated shops look alike.

The distance threshold is calibrated on WildReceipt, whose authors guarantee
that test templates never occur in training (`python -m docext.split`).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .augment import find_key_spans
from .schema import Document, load_split

DISTANCE_THRESHOLD = 0.5  # Jaccard distance below which two documents share a template
HEADER_LINES = 5  # text lines at the top that identify the source
CALIBRATION_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def template_signature(doc: Document) -> set[str]:
    """Header words (letters only, 3+ characters) plus key phrases."""
    signature = set()
    for word, line in zip(doc.words, doc.lines):
        if line - doc.lines[0] >= HEADER_LINES:
            break
        token = re.sub(r"[^a-z]", "", word.lower())
        if len(token) >= 3:
            signature.add(f"header:{token}")
    for start, end, field_type in find_key_spans(doc):
        signature.add(f"key:{field_type}:{' '.join(doc.words[start:end]).lower()}")
    return signature


def signature_distances(docs: list[Document]) -> np.ndarray:
    """Pairwise Jaccard distance between template signatures."""
    signatures = [template_signature(d) for d in docs]
    vocab = {token: i for i, token in enumerate(sorted(set().union(*signatures)))}
    x = np.zeros((len(docs), max(len(vocab), 1)), dtype=np.float32)
    for row, sig in enumerate(signatures):
        x[row, [vocab[t] for t in sig]] = 1.0
    inter = x @ x.T
    sizes = x.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1.0)


def cluster(distance: np.ndarray, threshold: float = DISTANCE_THRESHOLD) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold
    )
    return model.fit_predict(distance)


def cluster_templates(docs: list[Document], threshold: float = DISTANCE_THRESHOLD) -> np.ndarray:
    return cluster(signature_distances(docs), threshold)


def replication_rate(train_clusters: np.ndarray, test_clusters: np.ndarray) -> float:
    """Share of test documents whose template also occurs in training."""
    return float(np.isin(test_clusters, train_clusters).mean())


def disjoint_split(clusters: np.ndarray, sizes: dict[str, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Assign whole clusters, in random order, to the first split in `sizes`
    that still has room for all of them; the last split takes the rest."""
    rng = np.random.default_rng(seed)
    names = list(sizes)
    assignment = {name: [] for name in names}
    for c in rng.permutation(np.unique(clusters)):
        members = np.flatnonzero(clusters == c).tolist()
        target = next((n for n in names[:-1] if len(assignment[n]) + len(members) <= sizes[n]), names[-1])
        assignment[target].extend(members)
    empty = [n for n, idx in assignment.items() if not idx]
    if empty:
        raise ValueError(f"no template fits split(s) {empty}; lower the distance threshold")
    return {name: np.array(sorted(idx), dtype=np.int64) for name, idx in assignment.items()}


def seen_template_holdout(clusters: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """Pick up to `size` documents whose template stays represented in the
    rest: from every cluster with two or more members, all but one are
    candidates. Gives a validation set of seen layouts."""
    rng = np.random.default_rng(seed)
    candidates = []
    for c in np.unique(clusters):
        members = rng.permutation(np.flatnonzero(clusters == c))
        candidates.extend(members[1:].tolist())
    chosen = rng.permutation(candidates)[:size]
    return np.sort(chosen).astype(np.int64)


def calibrate(train: list[Document], test: list[Document]) -> list[dict]:
    """Templates found and test replication at each threshold. On a split
    whose test templates are known to be unseen, the right threshold is the
    loosest one that keeps replication near zero."""
    distance = signature_distances(train + test)
    rows = []
    for threshold in CALIBRATION_THRESHOLDS:
        clusters = cluster(distance, threshold)
        train_c, test_c = clusters[: len(train)], clusters[len(train):]
        sizes = np.bincount(train_c)
        rows.append({
            "threshold": threshold,
            "templates": int(len(np.unique(clusters))),
            "test_replication": replication_rate(train_c, test_c),
            "train_docs_in_shared_templates": float((sizes[train_c] >= 2).mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the template distance threshold.")
    parser.add_argument("data", type=Path, help="folder with train/test .pkl splits")
    parser.add_argument("--train", nargs="+", default=["train"], help="splits that count as training")
    parser.add_argument("--test", default="test")
    args = parser.parse_args()
    train = [d for name in args.train for d in load_split(args.data / f"{name}.pkl")[2]]
    _, _, test = load_split(args.data / f"{args.test}.pkl")
    rows = calibrate(train, test)
    (args.data / "calibration.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
