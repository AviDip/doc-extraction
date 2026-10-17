"""Offline fixtures: synthetic receipts and tiny randomly initialised models,
so the whole pipeline can be tested without downloads or a GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docext.layout import build_document
from docext.schema import save_split

FIELDS = [
    "menu.nm",
    "menu.price",
    "sub_total.service_price",
    "sub_total.subtotal_price",
    "sub_total.tax_price",
    "total.total_price",
]
ITEMS = ["Nasi Goreng", "Teh Manis", "Ayam Bakar", "Es Jeruk", "Mie Ayam", "Kopi Susu"]
KEYS = {"total.total_price": ["TOTAL", "GRAND TOTAL", "Total Due"], "sub_total.tax_price": ["TAX", "PB1"]}


def make_receipt(doc_id: str, rng: np.random.Generator, consistent: bool = True):
    words, boxes, entity_ids, labels = [], [], [], {}
    y, eid = 0.05 + rng.uniform(0, 0.1), 0
    left = rng.uniform(0.05, 0.2)

    def add(text, x, field=None, key=False):
        nonlocal eid
        for j, w in enumerate(text.split()):
            words.append(w)
            boxes.append((x + 0.08 * j, y, x + 0.08 * j + 0.07, y + 0.02))
            entity_ids.append(-1 if field is None or key else eid)
        if field and not key:
            labels[eid] = field
            eid += 1

    prices = []
    for _ in range(rng.integers(1, 4)):
        price = int(rng.integers(1, 50)) * 1000
        prices.append(price)
        add(rng.choice(ITEMS), left, "menu.nm")
        add(f"{price:,}", 0.7, "menu.price")
        y += 0.04
    tax = sum(prices) // 10
    total = sum(prices) + tax + (0 if consistent else sum(prices) // 4)
    for field, value in (("sub_total.tax_price", tax), ("total.total_price", total)):
        add(str(rng.choice(KEYS[field])), left, field, key=True)
        add(f"{value:,}", 0.7, field)
        y += 0.04
    return build_document(
        doc_id,
        words,
        np.array(boxes),
        np.zeros(len(words)),
        entity_ids,
        labels,
        cluster=int(rng.integers(0, 4)),
    )


def write_dataset(root: Path, n_train: int = 24, n_eval: int = 8, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data = root / "synthetic"
    for split, n in (("train", n_train), ("validation", n_eval)):
        docs = [make_receipt(f"{split}_{i}", rng, consistent=i % 5 != 0) for i in range(n)]
        save_split(data / f"{split}.pkl", "cord", FIELDS, docs)
    return data


def tiny_models(root: Path) -> dict[str, str]:
    """Save tiny RoBERTa, LayoutLMv3 and BROS checkpoints with their tokenizers."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (
        BertTokenizerFast,
        BrosConfig,
        BrosForTokenClassification,
        LayoutLMv3Config,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3TokenizerFast,
        RobertaConfig,
        RobertaForTokenClassification,
        RobertaTokenizerFast,
    )

    corpus = [" ".join(ITEMS + sum(KEYS.values(), [])) + " 0123456789 , ."] * 20
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        corpus, vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    bpe_dir = root / "bpe"
    bpe_dir.mkdir(parents=True, exist_ok=True)
    bpe.save_model(str(bpe_dir))
    vocab, merges = str(bpe_dir / "vocab.json"), str(bpe_dir / "merges.txt")

    wordpiece = root / "wordpiece.txt"
    pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(
        set("abcdefghijklmnopqrstuvwxyz0123456789,.") | {w.lower() for t in ITEMS for w in t.split()}
    )
    pieces += ["##" + c for c in "abcdefghijklmnopqrstuvwxyz0123456789,."]
    wordpiece.write_text("\n".join(pieces))

    small = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    specs = {
        "roberta": (
            RobertaTokenizerFast(vocab=vocab, merges=merges, add_prefix_space=True),
            RobertaForTokenClassification(
                RobertaConfig(vocab_size=300, max_position_embeddings=514, **small)
            ),
        ),
        "layoutlmv3": (
            LayoutLMv3TokenizerFast(vocab=vocab, merges=merges),
            LayoutLMv3ForTokenClassification(
                LayoutLMv3Config(vocab_size=300, coordinate_size=5, shape_size=6, **small)
            ),
        ),
        "bros": (
            BertTokenizerFast(str(wordpiece)),
            BrosForTokenClassification(
                BrosConfig(vocab_size=len(pieces), **(small | {"hidden_size": 64}), dim_bbox_projection=32)
            ),
        ),
    }
    paths = {}
    for name, (tokenizer, model) in specs.items():
        path = root / name
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
        paths[name] = str(path)
    return paths
