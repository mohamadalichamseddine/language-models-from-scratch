import csv
import json
import math

import numpy as np
import pytest
import torch

from lm.transformer.batches import BatchSampler
from lm.transformer.model import TransformerConfig, TransformerLM, load_checkpoint
from lm.transformer.training import (
    TrainingConfig,
    evaluate_loss,
    learning_rate,
    make_optimizer,
    train,
    train_step,
)

VOCAB_SIZE = 20

# A stream that repeats 0, 1, ..., 19, so the next token is easy to learn.
PATTERN = np.tile(np.arange(VOCAB_SIZE), 100)


def tiny_model(dropout=0.0):
    torch.manual_seed(0)
    config = TransformerConfig(vocab_size=VOCAB_SIZE, context=8, n_layers=1, n_heads=2, d_model=16, dropout=dropout)
    return TransformerLM(config)


def test_learning_rate_warms_up_then_decays():
    config = TrainingConfig(lr=1e-3, min_lr_ratio=0.1, warmup_steps=10, max_steps=110)
    rates = [learning_rate(step, config) for step in range(config.max_steps + 1)]

    assert rates[0] == pytest.approx(1e-4)
    assert rates[9] == pytest.approx(1e-3)  # end of warmup
    assert rates[10] == pytest.approx(1e-3)  # start of the cosine
    assert rates[60] == pytest.approx(0.55e-3)  # halfway: midway between lr and min lr
    assert rates[110] == pytest.approx(1e-4)  # end: lr * min_lr_ratio
    assert all(a <= b for a, b in zip(rates[:10], rates[1:11]))
    assert all(a >= b for a, b in zip(rates[10:], rates[11:]))


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)
    model = tiny_model()
    config = TrainingConfig(lr=1e-2, weight_decay=0.0)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    random_ids = np.random.default_rng(0).integers(0, VOCAB_SIZE, 1000)
    inputs, targets = BatchSampler(random_ids, context=8, tokens_per_batch=32).next_batch()
    for _ in range(300):
        loss = train_step(model, optimizer, scaler, inputs, targets, config.grad_clip)

    assert loss.item() < 0.05


def test_evaluate_loss_starts_near_uniform():
    loss = evaluate_loss(tiny_model().eval(), PATTERN, tokens_per_batch=64)
    assert loss == pytest.approx(math.log(VOCAB_SIZE), abs=0.1)


def test_short_run_lowers_val_loss_and_logs_it(tmp_path):
    model = tiny_model(dropout=0.1)
    config = TrainingConfig(tokens_per_batch=64, max_steps=60, lr=3e-3, warmup_steps=5, eval_every=20)
    summary = train(model, PATTERN, PATTERN[:400], config, tmp_path, tmp_path / "best.pt")

    with open(tmp_path / "curves.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows] == [0, 20, 40, 60]
    assert rows[0]["train_loss"] == ""  # no training steps yet at step 0
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"]) - 1

    saved = json.loads((tmp_path / "training.json").read_text())
    assert saved == summary
    assert summary["steps"] == 60 and not summary["stopped_early"]
    assert summary["best_val_loss"] == min(float(row["val_loss"]) for row in rows)

    best = load_checkpoint(tmp_path / "best.pt")
    assert evaluate_loss(best, PATTERN[:400], 64) == pytest.approx(summary["best_val_loss"])


def test_training_stops_when_val_loss_stops_improving(tmp_path):
    # With a learning rate of 0 the model never changes, so the val loss never improves.
    config = TrainingConfig(tokens_per_batch=64, max_steps=100, lr=0.0, eval_every=10, patience=2)
    summary = train(tiny_model(), PATTERN, PATTERN[:400], config, tmp_path, tmp_path / "best.pt")

    assert summary["stopped_early"]
    assert summary["steps"] == 20
    assert summary["best_step"] == 0
