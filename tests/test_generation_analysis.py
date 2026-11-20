from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from lm.ablation import data_identity, write_json
from lm.data import Vocab
from lm.generation import aggregate_generation, generation_metrics, generate_transformer
from lm.generation_analysis import GenerationConfig, generation_rows, run_generation, write_generation_report
from lm.transformer.model import TransformerConfig, TransformerLM, save_checkpoint
from lm.utils import set_seed


@pytest.mark.parametrize("ids,expected", [
    ([], (0, 0, 0)),
    ([1], (1, 0, 0)),
    ([1, 2, 3], (1, 1, 0)),
    ([1] * 6, (1 / 6, 1 / 5, 2 / 3)),
    ([1, 2, 1, 2, 1, 2], (1 / 3, 2 / 5, 1 / 3)),
])
def test_hand_computed_generation_metrics(ids, expected):
    assert list(generation_metrics(ids).values()) == pytest.approx(expected)


def test_macro_average_does_not_weight_long_samples_more():
    assert aggregate_generation([[1], [2] * 4])["distinct_1"] == pytest.approx((1 + 0.25) / 2)
    with pytest.raises(ValueError, match="No generated"):
        aggregate_generation([])


@pytest.mark.parametrize("settings", [
    {"tokens": 0}, {"prompts": 1.5}, {"temperature": float("nan")}, {"top_k": -1},
    {"contexts": [6, 6]}, {"training_seeds": []}, {"seed": -1},
])
def test_invalid_generation_config(settings):
    with pytest.raises(ValueError):
        GenerationConfig(**settings)


@pytest.fixture
def generated(tmp_path):
    vocab = Vocab(["<unk>", "<eos>", "a", "b", "c"])
    train = np.array([2, 3, 4, 1, 2, 3, 2, 1] * 3)
    valid = np.array([2, 3, 1, 2, 4, 3, 1, 2])
    config = GenerationConfig(prompts=2, prompt_length=2, tokens=6, contexts=[6], training_seeds=[0, 1], top_k=3)
    for seed in config.training_seeds:
        set_seed(seed)
        model = TransformerLM(TransformerConfig(vocab_size=5, context=6, n_layers=1, n_heads=1, d_model=8))
        run = tmp_path / f"ablation/k6_seed{seed}"
        save_checkpoint(model, run / "best.pt")
        write_json(run / "config.json", {"data": data_identity(train, valid, vocab),
                   "model_config": asdict(model.config), "training": {"seed": seed}})
    write_json(tmp_path / "ngram/tuning.json", {"best_k": {name: 0.1 for name in ("unigram", "bigram", "trigram")}})
    write_json(tmp_path / "ngram/interpolated/config.json", {"lambdas": [0.2, 0.3, 0.5], "vocab_size": 5})
    return run_generation(config, tmp_path, tmp_path / "generated", train, valid, vocab)


def test_complete_grid_and_reports_use_generated_tokens_only(generated, tmp_path):
    rows = write_generation_report(generated, tmp_path)
    assert len(rows) == 6
    assert len(generated["samples"]) == 12
    for row in rows:
        samples = [s for s in generated["samples"] if s["run"] == row["run"]]
        expected = np.mean([len(set(s["generated_ids"])) / 6 for s in samples])
        assert row["distinct_1"] == pytest.approx(expected)
    assert [r["training_seed"] for r in rows if r["model"] == "Transformer"] == [0, 1]
    assert (tmp_path / "metrics.csv").is_file()
    assert "Prompt 1" in (tmp_path / "representative_samples.md").read_text(encoding="utf-8")


def test_missing_duplicate_and_misaligned_samples_are_rejected(generated):
    data = deepcopy(generated)
    data["samples"].pop()
    with pytest.raises(ValueError, match="exactly one"):
        generation_rows(data)
    data = deepcopy(generated)
    data["samples"].append(data["samples"][0])
    with pytest.raises(ValueError, match="exactly one"):
        generation_rows(data)
    data = deepcopy(generated)
    data["samples"][-1]["prompt_ids"] = [0, 0]
    with pytest.raises(ValueError, match="identical prompts"):
        generation_rows(data)
    data = deepcopy(generated)
    data["samples"] = [s for s in data["samples"] if s["training_seed"] != 1]
    with pytest.raises(ValueError, match="grid"):
        generation_rows(data)


def test_transformer_sampling_replays_each_seed_in_a_batch():
    set_seed(9)
    model = TransformerLM(TransformerConfig(vocab_size=5, context=6, n_layers=1, n_heads=1, d_model=8))
    prompts = [[2, 3], [3, 2]]
    batch = generate_transformer(model, prompts, 8, [42, 43], top_k=3)
    assert batch == generate_transformer(model, prompts, 8, [42, 43], top_k=3)
    assert batch == [generate_transformer(model, [prompt], 8, [seed], top_k=3)[0]
                     for prompt, seed in zip(prompts, [42, 43])]
