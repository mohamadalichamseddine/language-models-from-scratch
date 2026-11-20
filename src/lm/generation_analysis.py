"""Matched-prompt generation for the frozen n-grams and context ablations."""

import csv
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from lm.ablation import CONTEXTS, SEEDS, data_identity, file_hash, read_json, write_json
from lm.generation import aggregate_generation, generate_ngram, generate_transformer, select_prompts
from lm.ngram import InterpolatedNGram, NGramModel
from lm.transformer.model import load_checkpoint

NGRAM_NAMES = ("unigram", "bigram", "trigram", "interpolated trigram")


@dataclass
class GenerationConfig:
    prompts: int = 30
    prompt_length: int = 10
    tokens: int = 100
    seed: int = 42
    temperature: float = 1.0
    top_k: int = 50
    contexts: list = field(default_factory=lambda: list(CONTEXTS))
    training_seeds: list = field(default_factory=lambda: list(SEEDS))

    def __post_init__(self):
        for name in ("prompts", "prompt_length", "tokens", "top_k"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - self.prompts:
            raise ValueError("seed and per-prompt seeds must be in [0, 2**32)")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not self.contexts or len(set(self.contexts)) != len(self.contexts) or any(
            type(k) is not int or k not in CONTEXTS for k in self.contexts
        ):
            raise ValueError(f"Choose distinct contexts from {CONTEXTS}")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds) or any(
            type(s) is not int or not 0 <= s < 2**32 for s in self.training_seeds
        ):
            raise ValueError("Choose distinct training seeds in [0, 2**32)")


def run_generation(config, results_dir, out, train_ids, valid_ids, vocab, device="cpu"):
    """Generate from frozen settings; validate all checkpoint provenance before starting."""
    results_dir, out = Path(results_dir), Path(out)
    identity = data_identity(train_ids, valid_ids, vocab)
    checkpoints = []
    for context in config.contexts:
        for seed in config.training_seeds:
            run = f"ablation/k{context}_seed{seed}"
            checkpoint = results_dir / run / "best.pt"
            saved = read_json(checkpoint.with_name("config.json"))
            if saved["data"] != identity:
                raise ValueError(f"Training data or vocabulary differs for {run}")
            if saved["model_config"]["context"] != context or saved["training"]["seed"] != seed:
                raise ValueError(f"Context or training seed differs for {run}")
            checkpoints.append((run, context, seed, checkpoint, file_hash(checkpoint)))

    tuning = read_json(results_dir / "ngram/tuning.json")
    interpolation = read_json(results_dir / "ngram/interpolated/config.json")
    if interpolation["vocab_size"] != len(vocab):
        raise ValueError("Interpolated vocabulary size differs from prepared data")
    prompts = select_prompts(valid_ids, config.prompts, config.prompt_length, config.seed)
    seeds = [config.seed + i for i in range(config.prompts)]
    samples = []

    def append_outputs(model, run, context, training_seed, outputs, settings):
        for i, generated in enumerate(outputs):
            samples.append({
                "model": model, "run": run, "context": context, "training_seed": training_seed,
                "prompt_index": i, "seed": seeds[i], "prompt_ids": prompts[i],
                "prompt": vocab.decode(prompts[i]), "generated_ids": generated,
                "generated": vocab.decode(generated), "settings": settings,
            })

    for n, name in enumerate(NGRAM_NAMES[:3], 1):
        k = tuning["best_k"][name]
        print(f"Generating {name} (k={k:g})...", flush=True)
        model = NGramModel(n, len(vocab), k=k).fit(train_ids)
        outputs = [generate_ngram(model, prompt, config.tokens, seed) for prompt, seed in zip(prompts, seeds)]
        append_outputs(name, f"ngram/n{n}_k{k:g}", n - 1, None, outputs,
                       {"smoothing": "add-k", "k": k, "num_tokens": config.tokens})
        del model

    print("Generating interpolated trigram...", flush=True)
    parts = [NGramModel(n, len(vocab), k=0).fit(train_ids) for n in (1, 2, 3)]
    model = InterpolatedNGram(parts, interpolation["lambdas"])
    outputs = [generate_ngram(model, prompt, config.tokens, seed) for prompt, seed in zip(prompts, seeds)]
    append_outputs(NGRAM_NAMES[3], "ngram/interpolated", 2, None, outputs,
                   {"parts_k": 0, "lambdas": interpolation["lambdas"], "num_tokens": config.tokens})
    del model, parts

    for run, context, seed, checkpoint, digest in checkpoints:
        print(f"Generating {run} on {device}...", flush=True)
        model = load_checkpoint(checkpoint, device)
        if model.config.vocab_size != len(vocab) or model.config.context != context:
            raise ValueError(f"Checkpoint configuration differs for {run}")
        outputs = generate_transformer(model, prompts, config.tokens, seeds, config.temperature, config.top_k)
        append_outputs("Transformer", run, context, seed, outputs,
                       {"temperature": config.temperature, "top_k": config.top_k,
                        "num_tokens": config.tokens, "checkpoint": f"results/{run}/best.pt",
                        "checkpoint_sha256": digest})
        del model

    manifest = {
        "split": "valid", "config": asdict(config), "data": identity,
        "runtime": {"python": platform.python_version(), "numpy": version("numpy"),
                    "torch": version("torch"), "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "CPU"},
        "sampling": "N-grams: full distribution. Transformer: temperature and top-k. Fixed-length continuations include <eos>.",
        "samples": samples,
    }
    write_generation_report(manifest, out)
    write_json(out / "config.json", {key: value for key, value in manifest.items() if key != "samples"})
    write_json(out / "samples.json", manifest)
    return manifest


def generation_rows(data):
    """Macro-average within each run after checking the complete, matched prompt grid."""
    config = GenerationConfig(**data["config"])
    if data["split"] != "valid":
        raise ValueError("Generation analysis requires validation prompts")
    expected = {(name, n if n < 3 else 2, None) for n, name in enumerate(NGRAM_NAMES)}
    expected |= {("Transformer", k, seed) for k in config.contexts for seed in config.training_seeds}
    grouped = defaultdict(list)
    prompts = {}
    vocab_size = data["data"]["vocab_size"]
    for sample in data["samples"]:
        key = (sample["model"], sample["context"], sample["training_seed"])
        grouped[key].append(sample)
        i = sample["prompt_index"]
        if type(i) is not int or not 0 <= i < config.prompts or sample["seed"] != config.seed + i:
            raise ValueError("Unexpected prompt index or generation seed")
        ids = sample["prompt_ids"]
        generated = sample["generated_ids"]
        if len(ids) != config.prompt_length or len(generated) != config.tokens:
            raise ValueError("Unexpected prompt or continuation length")
        settings = sample["settings"]
        if settings["num_tokens"] != config.tokens:
            raise ValueError("Saved sampling length differs from the experiment config")
        if sample["model"] == "Transformer" and (
            settings["temperature"] != config.temperature or settings["top_k"] != config.top_k
        ):
            raise ValueError("Saved Transformer sampling settings differ from the experiment config")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids + generated):
            raise ValueError("Token IDs must be integers within the vocabulary")
        if i in prompts and prompts[i] != ids:
            raise ValueError("All models must use identical prompts in the same order")
        prompts[i] = ids
    if set(grouped) != expected:
        raise ValueError("Generation file does not contain the requested model/context/seed grid")
    rows = []
    for key, samples in grouped.items():
        if sorted(s["prompt_index"] for s in samples) != list(range(config.prompts)):
            raise ValueError("Every run must have exactly one sample per prompt")
        runs = {s["run"] for s in samples}
        if len(runs) != 1:
            raise ValueError("One model/context/training seed must identify one run")
        if any(s["settings"] != samples[0]["settings"] for s in samples):
            raise ValueError("All samples in a run must use the same sampling settings")
        rows.append({"model": key[0], "run": samples[0]["run"], "context": key[1],
                     "training_seed": key[2], "samples": len(samples),
                     **aggregate_generation([s["generated_ids"] for s in samples])})
    return sorted(rows, key=lambda row: (row["model"] == "Transformer", row["context"], row["run"]))


def write_generation_report(data, out):
    """Save full-precision CSV, a readable table, and fixed prompt examples (0, 1, 2)."""
    rows = generation_rows(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = data["config"]
    lines = ["# Full generation analysis", "",
             f"Validation prompts: {config['prompts']} per run, {config['prompt_length']} tokens each; "
             f"{config['tokens']} generated tokens per continuation. Prompt-selection seed: {config['seed']}.", "",
             "Scores are macro-averages over generated tokens only. Context is the maximum number of preceding "
             "tokens; training seeds are separate rows. N-grams use their full frozen distribution; "
             f"Transformers use temperature {config['temperature']:g} and top-k {config['top_k']}.", "",
             "Distinct-n is unique n-grams / all n-grams. Repetition is duplicate 4-grams / all 4-grams. "
             "Samples shorter than n receive zero. These are diversity measures, not measures of fluency or correctness.", "",
             "| Model | Context | Training seed | Samples | Distinct-1 | Distinct-2 | Repetition-4 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        seed = "—" if row["training_seed"] is None else row["training_seed"]
        lines.append(f"| {row['model']} | {row['context']} | {seed} | {row['samples']} | "
                     f"{row['distinct_1']:.4f} | {row['distinct_2']:.4f} | {row['repetition_4']:.4f} |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# Representative continuations", "",
             "The first three selected prompts, fixed before examining outputs. Complete continuations "
             "are shown for every run; all samples and token IDs are in samples.json. These examples "
             "illustrate outputs and are not a quantitative estimate of typical quality.", ""]
    for i in range(min(3, config["prompts"])):
        matched = [s for s in data["samples"] if s["prompt_index"] == i]
        lines.extend([f"## Prompt {i} (sampling seed {config['seed'] + i})", "", "```text",
                      " ".join(matched[0]["prompt"]), "```", ""])
        for sample in matched:
            lines.extend([f"### {sample['run']}", "", "```text", " ".join(sample["generated"]), "```", ""])
    (out / "representative_samples.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return rows
