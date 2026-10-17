"""Structural constraints on extracted fields, as a training loss and as a
label-free check on predictions.

Arithmetic: the values of some fields must add up, e.g. line items plus
tax equal the total. Because the numbers on the page are fixed by OCR and
only the labels are learned, the expected value of a field is a
differentiable function of the model's probabilities:

    S(f) = sum_i P(word_i in f) * value_i

and an equation sum_k S(part_k) - S(total) = 0 can be penalised directly.

Cardinality: most key fields occur at most once per document. The expected
number of instances of f is sum_i P(word_i = B-f), and any excess over one
is penalised. Absence is legitimate (not every invoice has a due date), so
only the excess is penalised.

Both terms are zero when nothing is labelled, so they cannot stand alone:
they redistribute probability mass that the token loss has committed,
which is why they are added to it with a weight and a warm-up.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .schema import Document, Entity, LabelSpace, tags_to_entities

REL_TOLERANCE = 0.01  # rounding on printed amounts; residuals below 1% count as satisfied
MAX_VALUE_RATIO = 10.0  # cap on |value / total|; IDs and dates parsed as numbers stay bounded
SINGLETON_SHARE = 0.95  # a field is "at most once" if this share of documents agree
HUBER_BETA = 0.05  # quadratic below a 5% relative residual, linear above: bounded gradients


@dataclass(frozen=True)
class Equation:
    """sum of S(part) == S(total). parts[0] is the principal term (items,
    subtotal or net); the others (tax, service) may be absent."""

    name: str
    parts: tuple[str, ...]
    total: str


EQUATIONS: dict[str, list[Equation]] = {
    "cord": [
        Equation("items=subtotal", ("menu.price",), "sub_total.subtotal_price"),
        Equation(
            "items+tax+service=total",
            ("menu.price", "sub_total.tax_price", "sub_total.service_price"),
            "total.total_price",
        ),
        Equation(
            "subtotal+tax+service=total",
            ("sub_total.subtotal_price", "sub_total.tax_price", "sub_total.service_price"),
            "total.total_price",
        ),
    ],
    "wildreceipt": [
        Equation("items=subtotal", ("Prod_price",), "Subtotal"),
        Equation("items+tax+tips=total", ("Prod_price", "Tax", "Tips"), "Total"),
        Equation("subtotal+tax+tips=total", ("Subtotal", "Tax", "Tips"), "Total"),
    ],
    "docile": [
        Equation("net_items+tax=gross", ("line_item_amount_net", "amount_total_tax"), "amount_total_gross"),
        Equation("gross_items=gross", ("line_item_amount_gross",), "amount_total_gross"),
        Equation("net+tax=gross", ("amount_total_net", "amount_total_tax"), "amount_total_gross"),
    ],
}


def equations_for(dataset: str, label_space: LabelSpace) -> list[Equation]:
    """The dataset's equations whose fields all exist in the label space."""
    known = set(label_space.field_types)
    return [e for e in EQUATIONS.get(dataset, []) if {e.total, *e.parts} <= known]


def coefficient_matrix(equations: list[Equation], label_space: LabelSpace) -> torch.Tensor:
    """(equations, fields) matrix: +1 for parts, -1 for the total."""
    c = torch.zeros(len(equations), len(label_space.field_types))
    for k, e in enumerate(equations):
        for f in e.parts:
            c[k, label_space.field_to_index[f]] = 1.0
        c[k, label_space.field_to_index[e.total]] = -1.0
    return c


# ------------------------------------------------------------ ground truth


def field_sums(entities: list[Entity], values: np.ndarray) -> dict[str, float]:
    sums: dict[str, float] = Counter()
    for e in entities:
        sums[e.label] += float(values[e.start : e.end].sum())
    return sums


def equation_status(entities: list[Entity], values: np.ndarray, equations: list[Equation]):
    """For each equation: (applicable, satisfied, scale).

    An equation applies when its total and principal part are present;
    the residual is measured relative to the total.
    """
    present = {e.label for e in entities}
    sums = field_sums(entities, values)
    out = []
    for eq in equations:
        scale = abs(sums.get(eq.total, 0.0))
        applicable = eq.total in present and eq.parts[0] in present and scale > 0
        residual = sum(sums.get(p, 0.0) for p in eq.parts) - sums.get(eq.total, 0.0)
        satisfied = applicable and abs(residual) <= REL_TOLERANCE * scale
        out.append((applicable, satisfied, scale))
    return out


def singleton_fields(docs: list[Document]) -> list[str]:
    """Fields that appear at most once in at least SINGLETON_SHARE of the
    training documents that contain them."""
    present, at_most_once = Counter(), Counter()
    for doc in docs:
        counts = Counter(e.label for e in tags_to_entities(doc.tags))
        for f, n in counts.items():
            present[f] += 1
            at_most_once[f] += n == 1
    return sorted(f for f in present if at_most_once[f] >= SINGLETON_SHARE * present[f])


# -------------------------------------------------------------- the loss


class ConsistencyLoss(torch.nn.Module):
    """Arithmetic and cardinality penalties on word-level probabilities.

    Inputs are pooled over all words of a batch:
      probs       (W, C)  word label probabilities
      word_doc    (W,)    document index of each word
      values      (W,)    numeric value of each word (0 if none)
      eq_scale    (D, E)  |total| per document and equation, from ground truth
      eq_active   (D, E)  equation holds on this document's ground truth
      doc_active  (D,)    document fully covered by the forward pass
    """

    def __init__(self, label_space: LabelSpace, equations: list[Equation], singletons: list[str]):
        super().__init__()
        self.register_buffer("begin", torch.tensor(label_space.begin_ids()))
        self.register_buffer("inside", torch.tensor(label_space.inside_ids()))
        self.register_buffer("coef", coefficient_matrix(equations, label_space))
        self.register_buffer(
            "singleton", torch.tensor([label_space.field_to_index[f] for f in singletons], dtype=torch.long)
        )

    def forward(self, probs, word_doc, values, eq_scale, eq_active, doc_active):
        n_docs = eq_scale.shape[0]
        begin_mass = probs[:, self.begin]  # (W, F)
        field_mass = begin_mass + probs[:, self.inside]

        arithmetic = probs.new_zeros(())
        active = eq_active & doc_active[:, None]
        if active.any():
            # value of each word relative to its document's total, per equation
            scale = eq_scale[word_doc].clamp_min(1e-6)  # (W, E)
            rel_value = (values[:, None] / scale).clamp(-MAX_VALUE_RATIO, MAX_VALUE_RATIO)
            signed_mass = field_mass @ self.coef.T  # (W, E)
            residual = torch.zeros_like(eq_scale).index_add_(0, word_doc, signed_mass * rel_value)
            arithmetic = F.smooth_l1_loss(
                residual[active], torch.zeros_like(residual[active]), beta=HUBER_BETA
            )

        cardinality = probs.new_zeros(())
        if self.singleton.numel() and doc_active.any():
            counts = probs.new_zeros(n_docs, begin_mass.shape[1]).index_add_(0, word_doc, begin_mass)
            excess = F.relu(counts[:, self.singleton] - 1).sum(dim=1)
            cardinality = excess[doc_active].mean()

        return arithmetic, cardinality


# ------------------------------------------------- label-free check (routing)


def count_violations(
    entities: list[Entity], values: np.ndarray, equations: list[Equation], singletons: list[str]
) -> int:
    """Constraints broken by a prediction: applicable equations that do not
    hold, plus singleton fields predicted more than once."""
    broken = sum(applicable and not ok for applicable, ok, _ in equation_status(entities, values, equations))
    counts = Counter(e.label for e in entities)
    return broken + sum(counts[f] > 1 for f in singletons)
