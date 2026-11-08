import csv
import json

import numpy as np
import pytest

from lm.evaluation import (
    RESULTS_DIR,
    evaluate_run,
    predictions_path,
    run_name,
    save_config,
    write_table,
)
from lm.metrics import save_predictions


def make_run(run_dir, split="valid"):
    """A run with 4 predictions: CE = 1.5 bits, perplexity = 2 ** 1.5."""
    logprob = np.log([0.5, 0.25, 0.25, 0.5])
    rank = np.array([0, 1, 3, 0])
    save_predictions(predictions_path(run_dir, split), logprob, rank)


def test_save_config(tmp_path):
    save_config(tmp_path / "run", {"model": "ngram", "n": 3, "k": 1})
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config == {"model": "ngram", "n": 3, "k": 1}


def test_evaluate_run_saves_metrics(tmp_path):
    make_run(tmp_path)
    metrics = evaluate_run(tmp_path, "valid", num_tokens=4)

    assert metrics["cross_entropy_bits"] == pytest.approx(1.5)
    assert metrics["perplexity"] == pytest.approx(2 ** 1.5)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == {"valid": metrics}


def test_evaluate_run_keeps_other_splits(tmp_path):
    make_run(tmp_path, "valid")
    make_run(tmp_path, "test")

    evaluate_run(tmp_path, "valid", num_tokens=4)
    evaluate_run(tmp_path, "test", num_tokens=4)

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert set(saved) == {"valid", "test"}


def test_evaluate_run_needs_every_token(tmp_path):
    make_run(tmp_path)
    with pytest.raises(ValueError):
        evaluate_run(tmp_path, "valid", num_tokens=5)


def test_evaluate_run_without_predictions(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_run(tmp_path, "valid", num_tokens=4)


def test_run_name():
    assert run_name(RESULTS_DIR / "ngram" / "n3_k1") == "ngram/n3_k1"


def test_write_table(tmp_path):
    make_run(tmp_path)
    metrics = evaluate_run(tmp_path, "valid", num_tokens=4)

    write_table([("toy", metrics)], tmp_path / "table", "Toy table.")

    markdown = (tmp_path / "table.md").read_text()
    assert "| toy | 1.0397 | 1.5000 | 2.83 | 50.00% | 100.00% | 100.00% |" in markdown

    with open(tmp_path / "table.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run"] == "toy"
    assert float(rows[0]["perplexity"]) == metrics["perplexity"]
