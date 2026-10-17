import json

import numpy as np
import pytest
import torch

from docext.augment import Augmenter, find_key_spans, key_vocabulary
from docext.constraints import ConsistencyLoss, equation_status, equations_for, singleton_fields
from docext.encoding import Collator, pool_words, window_spans
from docext.layout import jitter_boxes, reading_order
from docext.numbers import parse_amount, word_values
from docext.routing import risk_coverage
from docext.schema import LabelSpace, load_split, tags_to_entities

from .helpers import FIELDS, make_receipt, tiny_models, write_dataset


@pytest.mark.parametrize(
    "text,value",
    [
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("45,000", 45000),
        ("12,50", 12.5),
        ("$12.00", 12.0),
        ("(5.00)", -5.0),
        ("5.00-", -5.0),
        ("12.03.2021", None),
        ("21%", None),
        ("Total:", None),
    ],
)
def test_parse_amount(text, value):
    assert parse_amount(text) == value


def test_split_thousands_merged_onto_last_word():
    doc = make_receipt("x", np.random.default_rng(0))
    doc = doc.copy(
        words=["1", "234,56"],
        boxes=np.array([[0.1, 0.1, 0.12, 0.12], [0.125, 0.1, 0.2, 0.12]]),
        pages=np.zeros(2, int),
        lines=np.zeros(2, int),
        tags=["B-total", "I-total"],
    )
    assert word_values(doc).tolist() == pytest.approx([0.0, 1234.56])


def test_reading_order_restores_rows():
    boxes = np.array([[0.6, 0.1, 0.7, 0.12], [0.1, 0.3, 0.2, 0.32], [0.1, 0.105, 0.2, 0.125]])
    order, lines = reading_order(boxes, np.zeros(3, int))
    assert order.tolist() == [2, 0, 1] and lines.tolist() == [0, 0, 1]


def test_jitter_keeps_relative_layout_and_bounds():
    boxes = np.array([[0.1, 0.1, 0.2, 0.12], [0.5, 0.1, 0.6, 0.12]], dtype=np.float32)
    out = jitter_boxes(boxes, np.zeros(2, int), 0.1, np.random.default_rng(1))
    assert out.min() >= 0 and out.max() <= 1
    assert out[0, 1] == pytest.approx(out[1, 1])  # same row stays the same row
    assert not np.allclose(out, boxes)


def test_ground_truth_constraints_and_keys():
    rng = np.random.default_rng(0)
    good, bad = make_receipt("a", rng), make_receipt("b", rng, consistent=False)
    space = LabelSpace(FIELDS)
    eqs = equations_for("cord", space)
    assert len(eqs) == 3
    # items+tax+service=total applies (no service line); the subtotal equations do not
    assert [s[:2] for s in equation_status(tags_to_entities(good.tags), word_values(good), eqs)] == [
        (False, False),
        (True, True),
        (False, False),
    ]
    assert equation_status(tags_to_entities(bad.tags), word_values(bad), eqs)[1][:2] == (True, False)
    assert {f for _, _, f in find_key_spans(good)} == {"sub_total.tax_price", "total.total_price"}
    assert "total.total_price" in singleton_fields([good, bad])


def test_consistency_loss_is_zero_for_correct_labels_and_penalises_errors():
    rng = np.random.default_rng(0)
    doc = make_receipt("a", rng)
    space = LabelSpace(FIELDS)
    eqs = [e for e in equations_for("cord", space) if e.name == "items+tax+service=total"]
    loss = ConsistencyLoss(space, eqs, ["total.total_price"])
    values = torch.tensor(word_values(doc))
    scale = equation_status(tags_to_entities(doc.tags), word_values(doc), eqs)[0][2]
    args = (
        torch.zeros(len(doc), dtype=torch.long),
        values,
        torch.tensor([[scale]]),
        torch.tensor([[True]]),
        torch.tensor([True]),
    )

    exact = torch.nn.functional.one_hot(torch.tensor(space.encode(doc.tags)), len(space)).float()
    arith, card = loss(exact, *args)
    assert arith.item() == pytest.approx(0, abs=1e-6) and card.item() == 0

    wrong = exact.clone()
    total = space.label_to_id["B-total.total_price"]
    first_price = doc.tags.index("B-menu.price")
    wrong[first_price] = 0
    wrong[first_price, total] = 1  # an item price mistaken for a second total

    arith, card = loss(wrong, *args)
    assert arith.item() > 0 and card.item() > 0


def test_risk_coverage_handles_ties():
    counts = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    curve = risk_coverage(np.array([0.0, 0.0, 1.0]), counts)
    assert curve[:, 0].tolist() == pytest.approx([2 / 3, 1.0])  # tied documents enter together


def test_window_spans_cover_everything():
    spans = window_spans(1200)
    assert spans[0][0] == 0 and spans[-1][1] == 1200
    assert all(e - s == 510 for s, e in spans)


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    return write_dataset(root), tiny_models(root / "models"), root


@pytest.mark.parametrize("arch", ["roberta", "layoutlmv3", "bros"])
def test_collator_and_forward(offline, arch):
    from docext.models import Extractor, load_tokenizer

    data, models, _ = offline
    dataset, types, docs = load_split(data / "train.pkl")
    space = LabelSpace(types)
    vocab = key_vocabulary(docs)
    aug = Augmenter(0.05, 1.0, vocab)
    batch_docs = [aug(d, np.random.default_rng(0)) for d in docs[:3]]
    collator = Collator(load_tokenizer(models[arch]), space, equations_for(dataset, space))
    batch = collator(batch_docs)
    assert batch["n_words"] == sum(len(d) for d in batch_docs)
    model = Extractor(models[arch], space)
    logits, hidden = model(batch["input_ids"], batch["attention_mask"], batch["bbox"], hidden=True)
    probs, counts = pool_words(logits.softmax(-1), batch["word_index"], batch["n_words"])
    assert probs.shape == (batch["n_words"], len(space)) and (counts > 0).all()


def test_train_evaluate_route_end_to_end(offline):
    from docext.routing import route
    from docext.train import TrainConfig, train

    data, models, root = offline
    cfg = TrainConfig(
        data=str(data),
        model=models["layoutlmv3"],
        out=str(root / "runs"),
        epochs=2,
        batch_size=4,
        lr=1e-3,
        jitter=0.05,
        key_swap=0.5,
        lambda_arith=0.5,
        lambda_card=0.5,
    )
    run = train(cfg)
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0.0 <= metrics["validation"]["all"]["f1"] <= 1.0
    report = route(run, "validation", target=0.5)
    assert set(report) == {"novelty", "msp", "mahalanobis", "consistency", "oracle"}
    assert (run / "routing_validation" / "risk_coverage.png").exists()


def test_train_many_runs_in_parallel_and_resumes(offline, capsys):
    from docext.train import TrainConfig, is_finished, train_many

    data, models, root = offline
    out = str(root / "runs_many")
    good = [TrainConfig(data=str(data), model=models[m], out=out, epochs=1, lr=1e-3) for m in ("roberta", "bros")]
    broken = TrainConfig(data=str(root / "missing"), model=models["bros"], out=out, epochs=1)
    train_many([*good, broken], parallel=2)
    printed = capsys.readouterr().out
    assert all(is_finished(c) for c in good) and not is_finished(broken)
    assert printed.count("done ") == 2 and "FAILED" in printed

    train_many(good, parallel=2)  # nothing left to train
    assert "2 of 2 runs already finished; 0 to train" in capsys.readouterr().out


def test_wildreceipt_record():
    from docext.data.wildreceipt import field_types, parse_record

    classes = {0: "Ignore", 15: "Prod_price_value", 23: "Total_value", 24: "Total_key", 25: "Others"}
    record = {"width": 100, "height": 100, "annotations": [
        {"box": [60, 10, 80, 10, 80, 14, 60, 14], "text": "$4.00", "label": 15},
        {"box": [10, 20, 30, 20, 30, 24, 10, 24], "text": "TOTAL", "label": 24},
        {"box": [60, 20, 80, 20, 80, 24, 60, 24], "text": "$4.00", "label": 23},
    ]}
    doc = parse_record("r", record, classes)
    assert field_types(classes) == ["Prod_price", "Total"]
    assert doc.tags == ["B-Prod_price", "O", "B-Total"] and doc.words[1] == "TOTAL"
