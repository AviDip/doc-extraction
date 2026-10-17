"""Noise floor: how far does the ground truth itself satisfy the constraints?

    python -m docext.analyse data/cord/train.pkl [--model layoutlmv3]

This is the go/no-go gate for the structural loss. It reports, on labelled
documents only:
  - per equation: documents where it applies, and the share where the
    ground-truth values satisfy it (the loss is applied only to those);
  - the same check with each field parsed as one string, which separates
    word-level parsing failures from documents that genuinely do not add up;
  - fields that occur at most once (the cardinality constraint's targets);
  - fields broken apart by reading order, and key phrases found;
  - with --model: the share of documents longer than one 512-token window.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .augment import key_vocabulary
from .constraints import equation_status, equations_for, singleton_fields
from .numbers import parse_amount, word_values
from .schema import LabelSpace, load_split, tags_to_entities


def text_values(doc, entities) -> np.ndarray:
    """Alternative values: each entity's words joined and parsed once,
    placed on its last word."""
    values = np.zeros(len(doc), dtype=np.float32)
    for e in entities:
        v = parse_amount("".join(doc.words[e.start : e.end]))
        values[e.end - 1] = 0.0 if v is None else v
    return values


def analyse(path: Path, model: str | None = None) -> dict:
    dataset, types, docs = load_split(path)
    equations = equations_for(dataset, LabelSpace(types))
    lengths = np.array([len(d) for d in docs])
    n_fields = sum(d.meta.get("fields", 0) for d in docs)

    applicable, satisfied, satisfied_text = Counter(), Counter(), Counter()
    for doc in docs:
        entities = tags_to_entities(doc.tags)
        word_status = equation_status(entities, word_values(doc), equations)
        text_status = equation_status(entities, text_values(doc, entities), equations)
        for eq, (app, ok, _), (_, ok_text, _) in zip(equations, word_status, text_status):
            applicable[eq.name] += app
            satisfied[eq.name] += ok
            satisfied_text[eq.name] += ok_text

    report = {
        "split": str(path),
        "documents": len(docs),
        "words_per_document": {"mean": float(lengths.mean()), "p95": float(np.percentile(lengths, 95))},
        "fields": n_fields,
        "fragmented_fields_share": sum(d.meta.get("fragmented_fields", 0) for d in docs) / max(n_fields, 1),
        "equations": {
            e.name: {
                "applicable_docs": applicable[e.name],
                "satisfied_share": satisfied[e.name] / max(applicable[e.name], 1),
                "satisfied_share_text_parse": satisfied_text[e.name] / max(applicable[e.name], 1),
            }
            for e in equations
        },
        "docs_with_any_satisfied_equation": sum(
            any(ok for _, ok, _ in equation_status(tags_to_entities(d.tags), word_values(d), equations))
            for d in docs
        ),
        "singleton_fields": singleton_fields(docs),
        "key_phrases_per_field": {f: len(p) for f, p in sorted(key_vocabulary(docs).items())},
    }

    if model:
        from .encoding import MAX_LENGTH, tokenize_words, window_spans
        from .models import load_tokenizer

        tokenizer = load_tokenizer(model)
        tokens = np.array([len(tokenize_words(tokenizer, d.words)[0]) for d in docs])
        windows = np.array([len(window_spans(t)) for t in tokens])
        report["tokens"] = {
            "model": model,
            "mean": float(tokens.mean()),
            "p95": float(np.percentile(tokens, 95)),
            "longer_than_one_window": float((tokens > MAX_LENGTH - 2).mean()),
            "mean_windows": float(windows.mean()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("split", type=Path, help="a .pkl file from docext.prepare")
    parser.add_argument("--model", help="also measure token lengths with this model's tokenizer")
    args = parser.parse_args()
    report = analyse(args.split, args.model)
    out = args.split.with_name(args.split.stem + "_analysis.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
