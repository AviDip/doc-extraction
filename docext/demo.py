"""Show one document's extraction and its routing decision, from a finished run.

Needs only a run folder (from the results branch) and the converted split, so
it runs on a laptop CPU. Used by demo.ipynb:

    from docext.demo import show, pick
    show("runs/cord_layoutlmv3_j0.0_k0.0_a0.0_c0.0_s0", "test", pick(run, "test", "riskiest")[0])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constraints import equation_status, equations_for
from .evaluate import load_predictions, load_run
from .metrics import predicted_entities
from .numbers import word_values
from .schema import load_split, tags_to_entities

COLOURS = {"correct": "#1baf7a", "wrong": "#e34948", "missed": "#eda100"}
SIGNAL = "mahalanobis"


def _load(run: Path, split: str):
    meta, label_space, _ = load_run(run)
    _, _, docs = load_split(Path(meta["config"]["data"]) / f"{split}.pkl")
    signals = np.load(run / f"routing_{split}" / "signals.npz")
    return meta, label_space, {d.doc_id: d for d in docs}, load_predictions(run, split), signals


def word_status(gold, pred, n_words: int) -> list[str | None]:
    """Per word: correct (in a correctly extracted field), wrong (in a
    predicted field that does not match), missed (in a field not found)."""
    gold_keys = {e.key() for e in gold}
    status: list[str | None] = [None] * n_words
    for e in gold:
        for i in range(e.start, e.end):
            status[i] = "missed"
    for e in pred:
        for i in range(e.start, e.end):
            status[i] = "correct" if e.key() in gold_keys else "wrong"
    return status


def auto_threshold(signals, signal: str, coverage: float) -> float:
    """Risk below which documents are auto-processed at the given coverage."""
    risk = np.sort(signals[signal])
    return float(risk[max(int(round(coverage * len(risk))) - 1, 0)])


def pick(run: str | Path, split: str, kind: str = "riskiest", n: int = 5, signal: str = SIGNAL) -> list[str]:
    """Document ids worth showing: "riskiest" or "safest" by the signal, or
    "errors" (documents with at least one extraction error)."""
    signals = np.load(Path(run) / f"routing_{split}" / "signals.npz")
    ids, counts = signals["doc_ids"], signals["counts"]
    if kind == "errors":
        return [str(i) for i in ids[(counts[:, 1] + counts[:, 2]) > 0][:n]]
    order = np.argsort(signals[signal])
    if kind == "riskiest":
        order = order[::-1]
    return [str(i) for i in ids[order[:n]]]


def show(run: str | Path, split: str, doc_id: str, signal: str = SIGNAL) -> None:
    import json

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    run = Path(run)
    meta, label_space, docs, preds, signals = _load(run, split)
    doc, p = docs[doc_id], preds[doc_id]
    gold, pred = tags_to_entities(doc.tags), predicted_entities(p["pred"], label_space)
    status = word_status(gold, pred, len(doc))

    # page with coloured boxes (first page)
    page = doc.pages == doc.pages[0]
    x0, y0 = doc.boxes[page, 0].min(), doc.boxes[page, 1].min()
    x1, y1 = doc.boxes[page, 2].max(), doc.boxes[page, 3].max()
    height = 7.0
    fig, ax = plt.subplots(figsize=(max(3.0, height * (x1 - x0) / max(y1 - y0, 1e-6)), height), dpi=110)
    for i in np.flatnonzero(page):
        b = doc.boxes[i]
        colour = COLOURS.get(status[i], "#c3c2b7")
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=status[i] is not None,
                               facecolor=colour, edgecolor=colour, alpha=0.45 if status[i] else 1, lw=0.6))
        ax.text(b[0], (b[1] + b[3]) / 2, doc.words[i], fontsize=5, va="center")
    ax.set(xlim=(x0, x1), ylim=(y1, y0), xticks=[], yticks=[], title=doc_id)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLOURS.items()], fontsize=7, loc="lower right")
    plt.show()

    # extracted fields
    values = word_values(doc)
    gold_keys = {e.key() for e in gold}
    print(f"{'field':28s} {'extracted':28s} {'conf':>5s}  ok")
    for e in pred:
        conf = float(np.mean(p["conf"][e.start:e.end]))
        text = " ".join(doc.words[e.start:e.end])[:28]
        print(f"{e.label:28s} {text:28s} {conf:5.2f}  {'✓' if e.key() in gold_keys else '✗'}")
    missed = [e for e in gold if e.key() not in {q.key() for q in pred}]
    for e in missed:
        print(f"{e.label:28s} {'(missed) ' + ' '.join(doc.words[e.start:e.end])[:19]:28s}")

    # structural checks on the prediction
    equations = equations_for(meta["dataset"], label_space)
    for eq, (applies, ok, _) in zip(equations, equation_status(pred, values, equations)):
        if applies:
            print(f"check {eq.name:30s} {'holds' if ok else 'BROKEN'}")

    # routing decision
    summary = json.loads((run / f"routing_{split}" / "summary.json").read_text())
    coverage = summary["signals"][signal]["coverage_at_target"]
    cut = auto_threshold(signals, signal, coverage)
    idx = int(np.flatnonzero(signals["doc_ids"] == doc_id)[0])
    risk = float(signals[signal][idx])
    route = "AUTO-PROCESS" if risk <= cut else "SEND TO HUMAN REVIEW"
    print(f"\n{signal} risk {risk:.3f} vs cut-off {cut:.3f} "
          f"(auto-processes {coverage:.0%} of documents at F1 ≥ {summary['target_f1']}) → {route}")
