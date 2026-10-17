"""Which documents can skip human review? Compare label-free risk signals.

    python -m docext.routing runs/<run> --split val --target 0.95

Each signal ranks documents from safest to riskiest. Auto-processing the
safest share (the coverage) and scoring extraction on that share gives a
risk-coverage curve. The headline is the largest coverage whose extraction
F1 still meets the target: the straight-through rate at that accuracy.

Signals (none uses labels of the evaluated documents):
  novelty      layout distance to the nearest training documents, from boxes only
  msp          1 - lowest word confidence (maximum softmax probability)
  mahalanobis  distance of the document embedding from the training distribution
  consistency  constraints broken by the prediction (arithmetic, cardinality)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from .constraints import count_violations, equations_for
from .evaluate import load_predictions, load_run
from .metrics import per_document_counts, predicted_entities, prf
from .numbers import word_values
from .schema import Document, load_split

GRID = 16  # layout descriptor resolution: GRID x GRID cells per page
NEIGHBOURS = 5


# ------------------------------------------------------------------ signals


def layout_descriptor(doc: Document) -> np.ndarray:
    """Ink map of the page: box area falling into each grid cell, L2-normalised.
    Pages of a multi-page document are summed."""
    centres = np.stack([(doc.boxes[:, 0] + doc.boxes[:, 2]) / 2, (doc.boxes[:, 1] + doc.boxes[:, 3]) / 2], 1)
    area = (doc.boxes[:, 2] - doc.boxes[:, 0]) * (doc.boxes[:, 3] - doc.boxes[:, 1])
    grid, _, _ = np.histogram2d(centres[:, 1], centres[:, 0], bins=GRID, range=[[0, 1], [0, 1]], weights=area)
    v = grid.ravel()
    return v / max(np.linalg.norm(v), 1e-12)


def novelty(train_docs, eval_docs) -> np.ndarray:
    knn = NearestNeighbors(n_neighbors=NEIGHBOURS, metric="cosine")
    knn.fit(np.stack([layout_descriptor(d) for d in train_docs]))
    dist, _ = knn.kneighbors(np.stack([layout_descriptor(d) for d in eval_docs]))
    return dist.mean(axis=1)


def msp_risk(eval_docs, preds) -> np.ndarray:
    return np.array([1.0 - float(preds[d.doc_id]["conf"].min()) if len(d) else 0.0 for d in eval_docs])


def mahalanobis(train_preds, eval_docs, preds) -> np.ndarray:
    train = np.stack([p["emb"] for p in train_preds.values()]).astype(np.float64)
    cov = LedoitWolf().fit(train)  # shrinkage keeps the 768-d covariance invertible
    x = np.stack([preds[d.doc_id]["emb"] for d in eval_docs]).astype(np.float64)
    return np.sqrt(cov.mahalanobis(x))


def consistency_risk(eval_docs, preds, label_space, equations, singletons) -> np.ndarray:
    return np.array(
        [
            count_violations(
                predicted_entities(preds[d.doc_id]["pred"], label_space),
                word_values(d),
                equations,
                singletons,
            )
            for d in eval_docs
        ],
        dtype=float,
    )


# ------------------------------------------------------- risk-coverage curve


def risk_coverage(risk: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Rows of (coverage, F1, exact-document share) at every distinct risk
    level. Tied documents enter together, so ties never favour a signal."""
    order = np.argsort(risk, kind="stable")
    r, c = risk[order], counts[order]
    exact = (c[:, 1] + c[:, 2] == 0).astype(float)
    cum, cum_exact = np.cumsum(c, axis=0), np.cumsum(exact)
    ends = np.flatnonzero(np.r_[r[1:] != r[:-1], True])
    rows = [((e + 1) / len(r), prf(*cum[e])["f1"], cum_exact[e] / (e + 1)) for e in ends]
    return np.array(rows)


def summarise(curve: np.ndarray, target: float) -> dict:
    coverage, f1, exact = curve.T
    meets = coverage[f1 >= target]
    cov = np.r_[0.0, coverage]
    return {
        "coverage_at_target": float(meets.max()) if meets.size else 0.0,
        # area under the error-coverage curve (lower is better), step-wise between tie groups
        "aurc": float(np.sum(np.diff(cov) * (1 - f1))),
        "f1_at_full_coverage": float(f1[-1]),
        "exact_at_full_coverage": float(exact[-1]),
    }


def plot(curves: dict, target: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for (name, curve), colour in zip(((k, v) for k, v in curves.items() if k != "oracle"), colours):
        ax.plot(curve[:, 0], curve[:, 1], lw=2, color=colour, label=name, drawstyle="steps-post")
    if "oracle" in curves:
        ax.plot(
            curves["oracle"][:, 0], curves["oracle"][:, 1], lw=1.5, ls="--", color="#8a8984", label="oracle"
        )
    ax.axhline(target, lw=1, color="#52514e", ls=":")
    ax.text(0.01, target, f" target F1 {target:.2f}", va="bottom", fontsize=8, color="#52514e")
    ax.set(
        xlabel="coverage (share of documents auto-processed)",
        ylabel="entity F1 on covered documents",
        xlim=(0, 1),
    )
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------- main


def route(run_dir: Path, split: str, target: float) -> dict:
    meta, label_space, train_docs = load_run(run_dir)
    cfg = meta["config"]
    _, _, docs = load_split(Path(cfg["data"]) / f"{split}.pkl")
    preds = load_predictions(run_dir, split)
    equations = equations_for(meta["dataset"], label_space)

    doc_counts = per_document_counts(docs, preds, label_space)
    counts = np.array([doc_counts[d.doc_id] for d in docs], dtype=float)
    signals = {
        "novelty": novelty(train_docs, docs),
        "msp": msp_risk(docs, preds),
        "mahalanobis": mahalanobis(load_predictions(run_dir, cfg["train_split"]), docs, preds),
        "consistency": consistency_risk(docs, preds, label_space, equations, meta["singletons"]),
        "oracle": counts[:, 1] + counts[:, 2],  # true errors per document: the best any signal can do
    }
    curves = {name: risk_coverage(risk, counts) for name, risk in signals.items()}
    report = {name: summarise(curve, target) for name, curve in curves.items()}

    out = run_dir / f"routing_{split}"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"target_f1": target, "signals": report}, indent=2))
    np.savez(out / "signals.npz", doc_ids=np.array([d.doc_id for d in docs]), counts=counts, **signals)
    plot(curves, target, out / "risk_coverage.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--target", type=float, default=0.95, help="entity F1 the auto-processed share must reach"
    )
    args = parser.parse_args()
    for name, row in route(args.run, args.split, args.target).items():
        print(f"{name:12s} " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
