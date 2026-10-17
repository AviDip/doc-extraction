"""The generated project.ipynb must run as one notebook: its library cells are
executed in a single namespace (as Colab does) and then drive a small
end-to-end experiment offline."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .helpers import tiny_models, write_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def notebook_namespace():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook.py")], check=True, capture_output=True)
    nb = json.loads((ROOT / "project.ipynb").read_text())
    namespace: dict = {}
    for c in nb["cells"]:
        if "library" in c["metadata"].get("tags", []):
            exec(compile("".join(c["source"]), "<notebook>", "exec"), namespace)  # noqa: S102
    return namespace


def test_notebook_runs_end_to_end(notebook_namespace, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    ns = notebook_namespace
    data = write_dataset(tmp_path, n_train=30, n_eval=10)
    models = tiny_models(tmp_path / "models")
    monkeypatch.chdir(tmp_path)

    report = ns["analyse"](data / "train.pkl", models["layoutlmv3"])
    assert report["documents"] == 30 and "tokens" in report

    rows = ns["calibrate"](ns["load_split"](data / "train.pkl")[2], ns["load_split"](data / "validation.pkl")[2])
    assert {r["threshold"] for r in rows} == set(ns["CALIBRATION_THRESHOLDS"])

    cfg = ns["TrainConfig"](data=str(data), model=models["bros"], out="runs", epochs=2, lr=1e-3,
                             jitter=0.05, key_swap=0.5, lambda_arith=0.3, lambda_card=0.3)
    ns["train_many"]([cfg], parallel=1)
    run = Path("runs") / cfg.run_name()
    assert (run / "metrics.json").exists()
    ns["route"](run, "validation", target=0.5)
    signals = dict(np.load(run / "routing_validation" / "signals.npz"))
    assert set(ns["fixed_cutoff"](signals, signals, 0.5)) == set(ns["SIGNALS"])
    ns["show"](run, "validation", ns["pick"](run, "validation", "riskiest", 1)[0])
    table = ns["results_table"](Path("runs"), "validation")
    assert len(table) == 1 and "all" in table.columns
