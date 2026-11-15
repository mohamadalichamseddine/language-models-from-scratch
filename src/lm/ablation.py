"""Equal-budget context experiments and checkpoint scoring with provenance."""

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from lm.evaluation import evaluate_run, predictions_path, save_config
from lm.transformer.model import TransformerConfig, TransformerLM, load_checkpoint
from lm.transformer.scoring import score_split
from lm.transformer.training import TrainingConfig, train
from lm.utils import set_seed

CONTEXTS = (6, 32, 64, 128)
SEEDS = (0, 1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_identity(train_ids, valid_ids, vocab):
    """Identify the shared token order and vocabulary, independent of paths."""
    return {
        "train_sha256": hashlib.sha256(np.asarray(train_ids, dtype="<i4").tobytes()).hexdigest(),
        "valid_sha256": hashlib.sha256(np.asarray(valid_ids, dtype="<i4").tobytes()).hexdigest(),
        "vocab_sha256": hashlib.sha256(json.dumps(vocab.tokens, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "train_tokens": len(train_ids),
        "valid_tokens": len(valid_ids),
        "vocab_size": len(vocab),
    }


def experiment_configs(settings, contexts=CONTEXTS, seeds=SEEDS):
    """Expand frozen settings; only context, seed and early stopping differ."""
    model_keys = {f.name for f in fields(TransformerConfig)} - {"vocab_size"}
    training_keys = {f.name for f in fields(TrainingConfig)}
    unknown = set(settings) - model_keys - training_keys
    if unknown:
        raise ValueError(f"Unknown frozen settings: {sorted(unknown)}")
    if not contexts or len(set(contexts)) != len(contexts) or any(k not in CONTEXTS for k in contexts):
        raise ValueError(f"Choose distinct contexts from {CONTEXTS}")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Choose distinct seeds in [0, 2**32)")
    training = asdict(TrainingConfig(**{k: v for k, v in settings.items() if k in training_keys}))
    training["early_stopping"] = False
    if training["max_steps"] < 1 or training["tokens_per_batch"] < 1 or training["eval_every"] < 1:
        raise ValueError("Steps, tokens per batch and evaluation interval must be positive")
    runs = []
    for context in contexts:
        if training["tokens_per_batch"] % context:
            raise ValueError("tokens_per_batch must be divisible by every selected context")
        model = asdict(TransformerConfig(vocab_size=0, **{k: v for k, v in settings.items() if k in model_keys}))
        model.pop("vocab_size")
        model["context"] = context
        for seed in seeds:
            runs.append({
                "run": f"k{context}_seed{seed}",
                "model_config": model.copy(),
                "training": {**training, "seed": seed},
                "token_budget": training["max_steps"] * training["tokens_per_batch"],
            })
    return runs


def score_checkpoint(checkpoint, out, ids, eos_id, identity, context=None, device="cpu", batch_size=256):
    """Score validation into its own folder; reuse only matching, intact outputs."""
    checkpoint, out = Path(checkpoint), Path(out)
    model = load_checkpoint(checkpoint, device)
    if model.config.vocab_size != identity["vocab_size"]:
        raise ValueError("Checkpoint vocabulary size does not match the prepared data")
    context = model.config.context if context is None else context
    if not 1 <= context <= model.config.context:
        raise ValueError("Scoring context exceeds the trained context or is nonpositive")
    signature = {
        "checkpoint_sha256": file_hash(checkpoint),
        "trained_context": model.config.context,
        "scoring_context": context,
        "split": "valid",
        "num_tokens": len(ids),
        "data": identity,
        "position_policy": "window-relative, starting at zero; prepended eos",
        "scorer_sha256": file_hash(Path(__file__).parent / "transformer/scoring.py"),
    }
    record_path = out / "scoring.json"
    prediction = predictions_path(out, "valid")
    previous = read_json(record_path) if record_path.exists() else {}
    reusable = (
        previous.get("signature") == signature
        and prediction.exists()
        and previous.get("predictions_sha256") == file_hash(prediction)
    )
    if not reusable:
        score_split(model, ids, eos_id, batch_size=batch_size, path=prediction, context=context)
        write_json(record_path, {
            "signature": signature,
            "checkpoint": str(checkpoint.resolve()),
            "predictions_sha256": file_hash(prediction),
            "torch_version": str(torch.__version__),
        })
    return evaluate_run(out, "valid", len(ids))


def train_ablation_run(spec, out, train_ids, valid_ids, vocab, identity, device="cpu", batch_size=256):
    """Restart interrupted runs from their seed; skip only completed matching runs.

    Checkpoints contain inference weights, not optimizer state. An interrupted
    training run therefore restarts from step zero rather than claiming to resume.
    """
    out = Path(out)
    config = {
        "model": "transformer",
        "experiment": "T18",
        "model_config": {**spec["model_config"], "vocab_size": len(vocab)},
        "training": spec["training"],
        "token_budget": spec["token_budget"],
        "data": identity,
        "train_split": "train",
        "val_tokens_for_val_loss": len(valid_ids),
    }
    config_path = out / "config.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise ValueError(f"{out} has different settings or data; choose a new output directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} contains files without a config; choose a new output directory")
    save_config(out, config)
    summary_path = out / "training.json"
    checkpoint = out / "best.pt"
    if summary_path.exists():
        summary = read_json(summary_path)
        if (summary["tokens_seen"] != spec["token_budget"] or summary["stopped_early"]
                or summary["steps"] != spec["training"]["max_steps"]):
            raise ValueError(f"{out} did not complete the required token budget")
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} is missing; restore it or choose a new output directory")
        print(f"Keeping completed training: {out.name}", flush=True)
    else:
        set_seed(spec["training"]["seed"])
        model = TransformerLM(TransformerConfig(**config["model_config"]))
        train(model, train_ids, valid_ids, TrainingConfig(**spec["training"]), out, device)
        del model
    return score_checkpoint(checkpoint, out, valid_ids, vocab.ids["<eos>"], identity,
                            device=device, batch_size=batch_size)
