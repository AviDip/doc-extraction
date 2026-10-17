"""Aggregate runs that differ only in seed: mean ± standard deviation.

python -m docext.summarise runs --split val [--csv results.csv]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = ("all", "many-shot", "few-shot", "zero-shot", "delta_many_minus_zero")


def collect(runs_dir: Path, split: str) -> dict[str, dict[str, list[float]]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_file.read_text()).get(split)
        if not metrics:
            continue
        setting = re.sub(r"_s\d+$", "", metrics_file.parent.name)
        for col in COLUMNS:
            value = metrics.get(col)
            if value is not None:
                groups[setting][col].append(value if isinstance(value, float) else value["f1"])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    groups = collect(args.runs, args.split)
    cols = [c for c in COLUMNS if any(c in g for g in groups.values())]
    lines = ["setting,seeds," + ",".join(cols)]
    for setting, values in sorted(groups.items()):
        cells = [
            f"{100 * np.mean(values[c]):.2f} ± {100 * np.std(values[c], ddof=1 if len(values[c]) > 1 else 0):.2f}"
            if values.get(c)
            else ""
            for c in cols
        ]
        n = max(len(v) for v in values.values())
        lines.append(f"{setting},{n}," + ",".join(cells))
    print("\n".join(lines))
    if args.csv:
        args.csv.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
