"""Build project.ipynb: the whole project in one explained notebook.

The package in docext/ stays the single source of truth. This script inlines
each module in dependency order, removes package plumbing (relative imports,
command-line entry points), splits the code into small cells and places the
explanations from notebook_text.py around them.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docext"
sys.path.insert(0, str(Path(__file__).parent))

from notebook_text import INTRO, PART_B, SECTIONS, SETTINGS, SETUP, SETUP_CODE, TABLE_HELPER  # noqa: E402

# functions that would clash once every module shares one namespace
RENAMES = {
    "data/cord.py": {"parse_record": "parse_cord_record"},
    "data/wildreceipt.py": {"parse_record": "parse_wildreceipt_record", "field_types": "wildreceipt_field_types"},
    "data/docile.py": {"field_types": "docile_field_types"},
    "prepare.py": {"field_types()": "docile_field_types()"},
}
# an alias needed where a relative import used "as"
PRELUDES = {"evaluate.py": "read_predictions = load_predictions  # the file reader from Section 11\n\n"}
# evaluate's run-level load_predictions intentionally replaces the file-level reader
ALLOWED_REDEFINITIONS = {"load_predictions"}
DROP_FUNCTIONS = {"main", "parse_args"}


def inline(module: str) -> str:
    """Module source without its docstring, relative imports, future import
    and command-line entry point."""
    source = (PKG / module).read_text()
    for old, new in RENAMES.get(module, {}).items():
        pattern = re.escape(old) if old.endswith(")") else rf"\b{re.escape(old)}\b"
        source = re.sub(pattern, new, source)
    tree = ast.parse(source)
    drop = []
    for node in tree.body:
        docstring = node is tree.body[0] and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        entry = isinstance(node, ast.FunctionDef) and node.name in DROP_FUNCTIONS
        main_guard = isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if docstring or entry or main_guard or future:
            drop.append((node.lineno, node.end_lineno))
    lines = source.splitlines()
    kept = [line for i, line in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop)]
    kept = [line for line in kept if not re.match(r"\s*from \.+\w*(\.\w+)* import ", line)]
    text = "\n".join(kept)
    if "argparse." not in text:
        text = re.sub(r"^import argparse\n", "", text, flags=re.M)
    return PRELUDES.get(module, "") + re.sub(r"\n{3,}", "\n\n\n", text).strip() + "\n"


def definition_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        return next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
    return None


def split_code(code: str, starters: list[str]) -> list[str]:
    """Cut code just before each listed top-level definition."""
    lines = code.splitlines()
    cuts = {}
    for node in ast.parse(code).body:
        name = definition_name(node)
        if name in starters:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            while first > 1 and lines[first - 2].lstrip().startswith("#"):
                first -= 1  # keep a comment block with the definition it describes
            cuts[name] = first - 1
    if missing := set(starters) - set(cuts):
        raise SystemExit(f"split points not found: {missing}")
    if list(cuts) != starters:  # the texts are matched to the pieces in order
        raise SystemExit(f"list split points in source order: {list(cuts)}")
    bounds = [0] + sorted(cuts.values()) + [len(lines)]
    return ["\n".join(lines[a:b]).strip("\n") + "\n" for a, b in zip(bounds, bounds[1:])]


def top_level_names(code: str) -> set[str]:
    return {name for node in ast.parse(code).body if (name := definition_name(node))}


def cell(kind: str, source: str, tags: list[str] | None = None) -> dict:
    lines = source.strip("\n").split("\n")
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
         "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]}
    if tags:
        c["metadata"]["tags"] = tags
    if kind != "md":
        c |= {"execution_count": None, "outputs": []}
    return c


def build() -> dict:
    cells = [cell("md", INTRO), cell("md", SETUP), cell("code", SETUP_CODE), cell("code", SETTINGS)]
    seen: dict[str, str] = {}
    for section in SECTIONS:
        module, chunks = section["module"], section["chunks"]
        code = inline(module)
        if module == "summarise.py":
            code += "\n\n" + TABLE_HELPER.strip() + "\n"

        clashes = {n: seen[n] for n in top_level_names(code) if n in seen and n not in ALLOWED_REDEFINITIONS}
        if clashes:
            raise SystemExit(f"{module} redefines {clashes}")
        seen |= {n: module for n in top_level_names(code)}

        pieces = split_code(code, list(chunks))
        cells += [cell("md", section["text"]), cell("code", pieces[0], ["library"])]
        for text, piece in zip(chunks.values(), pieces[1:]):
            cells += [cell("md", text), cell("code", piece, ["library"])]
    cells.append(cell("code", "import json\nfrom IPython.display import Image, display"))
    cells += [cell(kind, source) for kind, source in PART_B]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    notebook = build()
    (ROOT / "project.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"project.ipynb: {len(notebook['cells'])} cells")
