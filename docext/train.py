"""Fine-tune an extractor, then predict and score every evaluation split.

    python -m docext.train --data data/cord --eval-splits validation test --jitter 0.05

A run directory holds the config, training history, word-level predictions
for every split and metrics.json, so evaluation and routing can be redone
without a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .augment import Augmenter, key_vocabulary
from .constraints import ConsistencyLoss, equations_for, singleton_fields
from .encoding import IGNORE, Collator, pool_words
from .metrics import extraction_report
from .models import Extractor, load_tokenizer
from .predict import amp_dtype, autocast, predict, save_predictions
from .schema import LabelSpace, load_split

WARMUP_SHARE = 0.1  # share of steps with linear learning-rate warm-up
LAMBDA_RAMP_SHARE = 0.2  # share of steps over which the structural weights grow from 0
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def option(default, help: str):
    """A dataclass field that also carries its command-line help."""
    if callable(default):
        return field(default_factory=default, metadata={"help": help})
    return field(default=default, metadata={"help": help})


@dataclass
class TrainConfig:
    data: str = option("", "folder with <split>.pkl files from docext.prepare")
    model: str = option("layoutlmv3", "roberta | layoutlmv3 | bros, or a Hugging Face path")
    train_split: str = option("train", "split to train on")
    eval_splits: list[str] = option(lambda: ["validation"], "splits to predict and score")
    out: str = option("runs", "parent folder for run directories")
    name: str = option("", "run folder name; derived from the settings if empty")
    epochs: int = option(20, "passes over the training split")
    batch_size: int = option(4, "documents per optimisation step")
    lr: float = option(5e-5, "peak learning rate")
    jitter: float = option(0.0, "page shift/scale magnitude as a share of page size (0 = off)")
    key_swap: float = option(0.0, "probability of replacing each detected key phrase (0 = off)")
    lambda_arith: float = option(0.0, "weight of the arithmetic consistency term")
    lambda_card: float = option(0.0, "weight of the cardinality term")
    max_windows: int = option(4, "512-token windows taken per document per step")
    seed: int = option(0, "random seed")
    save_model: bool = option(False, "keep the fine-tuned weights (~500 MB)")

    def run_name(self) -> str:
        return self.name or (
            f"{Path(self.data).name}_{Path(self.model).name}_j{self.jitter}_k{self.key_swap}"
            f"_a{self.lambda_arith}_c{self.lambda_card}_s{self.seed}"
        )


class AugmentedDocs(Dataset):
    def __init__(self, docs, augmenter: Augmenter, seed: int):
        self.docs, self.augmenter = docs, augmenter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return self.augmenter(self.docs[i], self.rng)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.out) / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset, types, train_docs = load_split(Path(cfg.data) / f"{cfg.train_split}.pkl")
    label_space = LabelSpace(types)
    equations = equations_for(dataset, label_space)
    singletons = singleton_fields(train_docs)
    vocab = key_vocabulary(train_docs) if cfg.key_swap > 0 else {}

    tokenizer = load_tokenizer(cfg.model)
    model = Extractor(cfg.model, label_space).to(device)
    structure = ConsistencyLoss(label_space, equations, singletons).to(device)
    collator = Collator(tokenizer, label_space, equations, cfg.max_windows, cfg.seed)
    loader = DataLoader(
        AugmentedDocs(train_docs, Augmenter(cfg.jitter, cfg.key_swap, vocab), cfg.seed),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    total_steps = cfg.epochs * len(loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=WEIGHT_DECAY, fused=device.type == "cuda"
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=amp_dtype(device) == torch.float16)

    meta = {
        "config": asdict(cfg),
        "dataset": dataset,
        "field_types": types,
        "singletons": singletons,
        "equations": [e.name for e in equations],
        "device": str(device),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{run_dir.name}: {len(train_docs)} documents, {total_steps} steps on {device}")

    history, step = [], 0
    for epoch in range(cfg.epochs):
        model.train()
        sums, start = np.zeros(3), time.time()
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast(device):
                logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["bbox"])
            logits = logits.float()
            ce = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=IGNORE)
            probs, _ = pool_words(logits.softmax(-1), batch["word_index"], batch["n_words"])
            arith, card = structure(
                probs,
                batch["word_doc"],
                batch["values"],
                batch["eq_scale"],
                batch["eq_active"],
                batch["doc_active"],
            )
            ramp = min(1.0, step / max(1, LAMBDA_RAMP_SHARE * total_steps))
            loss = ce + ramp * (cfg.lambda_arith * arith + cfg.lambda_card * card)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            sums += [ce.item(), arith.item(), card.item()]
            step += 1

        record = dict(zip(("ce", "arith", "card"), (sums / len(loader)).round(5).tolist()))
        record |= {"epoch": epoch + 1, "seconds": round(time.time() - start)}
        history.append(record)
        print(record)

    if cfg.save_model:
        model.save(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "model")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    eval_collator = Collator(tokenizer, label_space)
    metrics = {}
    for split in [cfg.train_split, *cfg.eval_splits]:
        _, _, docs = load_split(Path(cfg.data) / f"{split}.pkl")
        preds = predict(model, eval_collator, docs, device)
        save_predictions(run_dir / f"pred_{split}.npz", preds)
        if split != cfg.train_split:
            metrics[split] = extraction_report(train_docs, docs, preds, label_space)
            print(split, json.dumps(metrics[split]["all"]))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return run_dir


# ------------------------------------------------------ several runs at once


def is_finished(cfg: TrainConfig) -> bool:
    return (Path(cfg.out) / cfg.run_name() / "metrics.json").exists()


def _train_logged(cfg: TrainConfig) -> None:
    """Train in a child process, writing everything it prints to train.log."""
    # the parent's processor thread pool is not copied into the child; with one
    # thread, the child never waits for threads that do not exist
    torch.set_num_threads(1)
    run_dir = Path(cfg.out) / cfg.run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train.log", "w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            train(cfg)
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1)


def _report(cfg: TrainConfig, ok: bool, minutes: float) -> str:
    run_dir = Path(cfg.out) / cfg.run_name()
    if not ok:
        tail = (run_dir / "train.log").read_text(errors="replace").splitlines()[-5:]
        return f"FAILED  {run_dir.name} after {minutes:.1f} min\n    " + "\n    ".join(tail)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    scores = "  ".join(f"{split} F1 {100 * m['all']['f1']:.1f}" for split, m in metrics.items())
    return f"done    {run_dir.name}  {minutes:.1f} min  {scores}"


def train_many(configs: list[TrainConfig], parallel: int = 1) -> None:
    """Train every unfinished run, `parallel` at a time, each in its own process.

    Runs share the GPU: while one waits for the processor to prepare its next
    batch, another computes. A run counts as finished once its metrics.json
    exists, so calling this again continues where an interrupted call stopped.
    """
    if torch.cuda.is_initialized():  # a forked process cannot use a GPU its parent has opened
        raise RuntimeError("the GPU is already in use by this notebook; restart the runtime and run all cells")
    todo = [c for c in configs if not is_finished(c)]
    print(f"{len(configs) - len(todo)} of {len(configs)} runs already finished; "
          f"{len(todo)} to train, {parallel} at a time")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer threads do not survive a fork
    fork = multiprocessing.get_context("fork")
    running: dict = {}  # process -> (config, start time)
    try:
        while todo or running:
            while todo and len(running) < parallel:
                cfg = todo.pop(0)
                process = fork.Process(target=_train_logged, args=(cfg,))
                process.start()
                running[process] = (cfg, time.time())
            finished = wait([p.sentinel for p in running])
            for process in [p for p in running if p.sentinel in finished]:
                process.join()
                cfg, began = running.pop(process)
                print(_report(cfg, process.exitcode == 0, (time.time() - began) / 60), flush=True)
    finally:  # stopping the cell also stops the runs in progress
        for process in running:
            process.terminate()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag, default, help = "--" + f.name.replace("_", "-"), getattr(defaults, f.name), f.metadata["help"]
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help)
        elif isinstance(default, list):
            parser.add_argument(flag, nargs="+", default=default, help=help)
        else:
            parser.add_argument(
                flag, type=type(default), default=default, help=help, required=f.name == "data"
            )
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
