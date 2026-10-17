"""Template-disjoint splits for datasets without published layout clusters.

Following Laatiri et al. (2023), documents are clustered on the field keys
they share: two receipts from the same shop print the same keys ("SUBTOTAL",
"PB1", "CASH") for the same fields. Whole clusters are then assigned to one
side of the split, so no template appears in both training and test.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .augment import find_key_spans
from .schema import Document, tags_to_entities

DISTANCE_THRESHOLD = 0.5  # Jaccard distance below which two documents share a template


def template_signature(doc: Document) -> set[str]:
    """Field types present plus the lower-cased key words that name them."""
    signature = {f"field:{e.label}" for e in tags_to_entities(doc.tags)}
    for start, end, field_type in find_key_spans(doc):
        signature.add(f"key:{field_type}:{' '.join(doc.words[start:end]).lower()}")
    return signature


def cluster_templates(docs: list[Document]) -> np.ndarray:
    signatures = [template_signature(d) for d in docs]
    vocab = {token: i for i, token in enumerate(sorted(set().union(*signatures)))}
    x = np.zeros((len(docs), len(vocab)), dtype=bool)
    for row, sig in enumerate(signatures):
        x[row, [vocab[t] for t in sig]] = True
    inter = x.astype(np.float32) @ x.T.astype(np.float32)
    sizes = x.sum(axis=1).astype(np.float32)
    union = sizes[:, None] + sizes[None, :] - inter
    distance = 1.0 - inter / np.maximum(union, 1.0)
    model = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=DISTANCE_THRESHOLD
    )
    return model.fit_predict(distance)


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
        raise ValueError(f"no template fits split(s) {empty}; lower DISTANCE_THRESHOLD")
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
