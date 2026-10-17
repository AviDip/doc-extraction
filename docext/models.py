"""The three encoders behind one interface.

RoBERTa sees text only (the control), LayoutLMv3 adds absolute 2-D position
embeddings and a relative spatial attention bias (the working model), BROS
encodes relative positions between every pair of boxes (the reference).
LayoutLMv3 is used without page images: the image would reveal the exact
layout that jitter is meant to hide, and RoBERTa and BROS receive none.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from .schema import LabelSpace

ALIASES = {
    "roberta": "roberta-base",
    "layoutlmv3": "microsoft/layoutlmv3-base",
    "bros": "jinho8345/bros-base-uncased",  # BROS weights ported to the transformers implementation
}

# how each architecture expects boxes: none, integers on a 0-1000 grid, or floats in [0, 1]
BOX_FORMAT = {"roberta": None, "layoutlmv3": "grid1000", "bros": "unit"}
# byte-level BPE tokenizers need a leading space per word when input is pre-split
PREFIX_SPACE = {"roberta", "layoutlmv3"}


def resolve(name: str) -> str:
    return ALIASES.get(name, name)


def load_tokenizer(name: str):
    path = resolve(name)
    model_type = AutoConfig.from_pretrained(path).model_type
    kwargs = {"add_prefix_space": True} if model_type in PREFIX_SPACE else {}
    return AutoTokenizer.from_pretrained(path, **kwargs)


class Extractor(torch.nn.Module):
    """Token classifier that takes boxes as floats in [0, 1] and converts
    them to whatever the underlying architecture expects."""

    def __init__(self, name: str, label_space: LabelSpace):
        super().__init__()
        path = resolve(name)
        config = AutoConfig.from_pretrained(
            path,
            num_labels=len(label_space),
            id2label=dict(enumerate(label_space.labels)),
            label2id=label_space.label_to_id,
        )
        if config.model_type not in BOX_FORMAT:
            raise ValueError(
                f"unsupported architecture {config.model_type!r}; expected one of {list(BOX_FORMAT)}"
            )
        self.box_format = BOX_FORMAT[config.model_type]
        # a checkpoint fine-tuned on another label set gets a fresh classifier head
        self.model = AutoModelForTokenClassification.from_pretrained(
            path, config=config, ignore_mismatched_sizes=True
        )

    def forward(self, input_ids, attention_mask, bbox, hidden: bool = False):
        kwargs = {}
        if self.box_format == "grid1000":
            kwargs["bbox"] = (bbox * 1000).round().long().clamp(0, 1000)
        elif self.box_format == "unit":
            kwargs["bbox"] = bbox
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=hidden, **kwargs
        )
        return out.logits, (out.hidden_states[-1] if hidden else None)

    def save(self, directory: Path) -> None:
        self.model.save_pretrained(directory)
