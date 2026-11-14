import copy

import numpy as np
import pytest

from lm.ablation import (
    data_identity,
    experiment_configs,
    read_json,
    train_ablation_run,
)
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


def test_scoring_rebuilds_corrupted_predictions(tmp_path):
    spec, run, ids, vocab, identity = toy_experiment(tmp_path)
    (run / "valid.npz").write_bytes(b"interrupted write")
    train_ablation_run(spec, run, ids, ids[:24], vocab, identity, batch_size=4)
    from lm.metrics import load_predictions
    logprob, rank = load_predictions(run / "valid.npz")
    assert len(logprob) == len(rank) == 24
