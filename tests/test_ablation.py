import copy
import math

import numpy as np
import pytest

from lm.ablation import (
    data_identity,
    experiment_configs,
    read_json,
    score_checkpoint,
    train_ablation_run,
    write_json,
)
from lm.ablation_plots import build_ablation_figures, collect_runs, summarize_runs
from lm.data import ROOT, Vocab


def toy_experiment(tmp_path, seed=0):
    vocab = Vocab(["<eos>", "<unk>", "a", "b"])
    ids = np.tile(np.arange(len(vocab)), 40)
    settings = {"n_layers": 1, "n_heads": 1, "d_model": 8, "dropout": 0.0,
                "max_steps": 2, "tokens_per_batch": 12, "eval_every": 1, "warmup_steps": 0}
    spec = experiment_configs(settings, contexts=[6], seeds=[seed])[0]
    identity = data_identity(ids, ids[:24], vocab)
    run = tmp_path / spec["run"]
    train_ablation_run(spec, run, ids, ids[:24], vocab, identity, batch_size=4)
    return spec, run, ids, vocab, identity


def test_frozen_grid_preserves_hyperparameters_and_equal_budget():
    settings = read_json(ROOT / "configs/transformer.json")
    before = copy.deepcopy(settings)
    runs = experiment_configs(settings)
    assert len(runs) == 8
    assert {(r["model_config"]["context"], r["training"]["seed"]) for r in runs} == {
        (k, seed) for k in (6, 32, 64, 128) for seed in (0, 1)
    }
    for run in runs:
        assert run["token_budget"] == 49_152_000
        assert not run["training"]["early_stopping"]
        effective = {**run["model_config"], **run["training"]}
        for key in settings.keys() - {"context", "seed", "early_stopping"}:
            assert effective[key] == settings[key]
    assert settings == before


@pytest.mark.parametrize("settings,contexts,seeds", [
    ({}, [6, 6], [0]), ({}, [7], [0]), ({}, [6], [0, 0]),
    ({}, [6], [-1]), ({"tokens_per_batch": 128}, [6], [0]),
    ({"max_steps": 0}, [6], [0]), ({"typo": 1}, [6], [0]),
])
def test_bad_experiment_plan_is_rejected(settings, contexts, seeds):
    with pytest.raises(ValueError):
        experiment_configs(settings, contexts, seeds)


def test_run_reuse_checks_settings_and_data(tmp_path):
    spec, run, ids, vocab, identity = toy_experiment(tmp_path)
    before = (run / "best.pt").read_bytes()
    prediction_time = (run / "valid.npz").stat().st_mtime_ns
    train_ablation_run(spec, run, ids, ids[:24], vocab, identity, batch_size=3)
    assert (run / "best.pt").read_bytes() == before
    assert (run / "valid.npz").stat().st_mtime_ns == prediction_time
    assert read_json(run / "training.json")["tokens_seen"] == 24
    changed = copy.deepcopy(spec)
    changed["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="different settings or data"):
        train_ablation_run(changed, run, ids, ids[:24], vocab, identity)
    with pytest.raises(ValueError, match="different settings or data"):
        train_ablation_run(spec, run, ids, ids[:24], vocab, {**identity, "valid_sha256": "different"})


def test_scoring_rebuilds_corrupted_predictions_and_keeps_truncation_separate(tmp_path):
    spec, run, ids, vocab, identity = toy_experiment(tmp_path)
    full = (run / "valid.npz").read_bytes()
    (run / "valid.npz").write_bytes(b"interrupted write")
    train_ablation_run(spec, run, ids, ids[:24], vocab, identity, batch_size=4)
    from lm.metrics import load_predictions
    logprob, rank = load_predictions(run / "valid.npz")
    assert len(logprob) == len(rank) == 24
    restored = (run / "valid.npz").read_bytes()
    out = tmp_path / "truncated"
    score_checkpoint(run / "best.pt", out, ids[:24], vocab.ids["<eos>"], identity, context=2)
    assert (run / "valid.npz").read_bytes() == restored
    assert read_json(out / "scoring.json")["signature"]["scoring_context"] == 2
    assert full  # the original full-context output was nonempty


def test_summary_uses_sample_sd_and_marks_single_seed_unknown():
    rows = [
        {"context": 6, "seed": 0, "perplexity": 10, "cross_entropy_nats": 2, "top1_accuracy": 0.2},
        {"context": 6, "seed": 1, "perplexity": 14, "cross_entropy_nats": 4, "top1_accuracy": 0.4},
        {"context": 32, "seed": 0, "perplexity": 9, "cross_entropy_nats": 1, "top1_accuracy": 0.5},
    ]
    first, second = summarize_runs(rows)
    assert first["perplexity_mean"] == 12
    assert first["perplexity_std"] == pytest.approx(math.sqrt(8))
    assert first["cross_entropy_nats_std"] == pytest.approx(math.sqrt(2))
    assert first["top1_accuracy_mean"] == pytest.approx(0.3)
    assert second["perplexity_std"] is None


def test_plots_reject_incomplete_grid_and_label_preview(tmp_path):
    toy_experiment(tmp_path)
    with pytest.raises(ValueError, match="Missing runs"):
        collect_runs(tmp_path, contexts=[6], seeds=[0, 1])
    out = tmp_path / "figures"
    status = build_ablation_figures(tmp_path, ROOT / "results", out, contexts=[6], seeds=[0, 1], allow_partial=True)
    assert status["missing_runs"] == ["k6_seed1"]
    assert not status["complete_requested_grid"]
    assert "SD n/a" in (out / "ablation_summary.md").read_text()
    assert len(list(out.glob("*.png"))) == 3


def test_plots_reject_wrong_budget_and_mixed_settings(tmp_path):
    _, first, _, _, _ = toy_experiment(tmp_path, seed=0)
    _, second, _, _, _ = toy_experiment(tmp_path, seed=1)
    rows, missing = collect_runs(tmp_path, contexts=[6], seeds=[0, 1])
    assert len(rows) == 2 and not missing
    config = read_json(second / "config.json")
    config["training"]["lr"] *= 2
    write_json(second / "config.json", config)
    with pytest.raises(ValueError, match="differ beyond"):
        collect_runs(tmp_path, contexts=[6], seeds=[0, 1])
    summary = read_json(first / "training.json")
    summary["tokens_seen"] -= 1
    write_json(first / "training.json", summary)
    with pytest.raises(ValueError, match="budget"):
        collect_runs(tmp_path, contexts=[6], seeds=[0])
