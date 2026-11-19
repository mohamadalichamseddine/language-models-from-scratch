"""T14: continue the same validation prompts with every model.

Usage:
    python scripts/generate.py

Every model continues the same prompts (30 ten-token snippets from valid by
default) by 100 tokens, and prompt i always samples with seed + i. The n-gram
models sample from their add-k distribution at the best k from T07; the
Transformer uses temperature + top-k sampling. The samples and their settings
are saved in results/generation/samples.json. If the Transformer checkpoint is
missing, only the n-gram samples are written.
"""

import argparse
import json
from pathlib import Path

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR
from lm.generation import generate_ngram, generate_transformer, select_prompts
from lm.ngram import NGramModel
from lm.transformer.model import load_checkpoint
from lm.utils import repo_path

NGRAM_NAMES = ("unigram", "bigram", "trigram")


def parse_args():
    parser = argparse.ArgumentParser(description="Continue the same validation prompts with every model.")
    parser.add_argument("--transformer-run", default="transformer/best", help="run folder under results/")
    parser.add_argument("--checkpoint", type=Path, help="use this model instead of results/<transformer-run>/best.pt")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "generation/samples.json")
    parser.add_argument("--prompts", type=int, default=30, help="number of prompts")
    parser.add_argument("--prompt-length", type=int, default=10, help="tokens per prompt")
    parser.add_argument("--tokens", type=int, default=100, help="new tokens per sample")
    parser.add_argument("--seed", type=int, default=42, help="picks the prompts; prompt i samples with seed + i")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    return args


def sample(vocab, prompt_index, seed, prompt, generated, settings, **model):
    """One entry of samples.json: the model, its prompt and output, and the settings."""
    return {
        **model,
        "prompt_index": prompt_index,
        "seed": seed,
        "prompt_ids": prompt,
        "prompt": vocab.decode(prompt),
        "generated_ids": generated,
        "generated": vocab.decode(generated),
        "settings": settings,
    }


def main():
    args = parse_args()
    vocab = load_vocab()
    prompts = select_prompts(load_split("valid"), args.prompts, args.prompt_length, args.seed)
    seeds = [args.seed + i for i in range(len(prompts))]
    samples = []

    train_ids = load_split("train")
    best_k = json.loads((RESULTS_DIR / "ngram/tuning.json").read_text(encoding="utf-8"))["best_k"]
    for n, name in enumerate(NGRAM_NAMES, 1):
        k = float(best_k[name])
        print(f"Fitting and generating {name} (k={k:g})...", flush=True)
        model = NGramModel(n, len(vocab), k=k).fit(train_ids)
        settings = {"smoothing": "add-k", "k": k, "num_tokens": args.tokens}
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            generated = generate_ngram(model, prompt, args.tokens, seed)
            samples.append(sample(vocab, i, seed, prompt, generated, settings, model=name))

    checkpoint = args.checkpoint or RESULTS_DIR / args.transformer_run / "best.pt"
    if checkpoint.is_file():
        print(f"Generating Transformer from {checkpoint}...", flush=True)
        model = load_checkpoint(checkpoint, args.device)
        if model.config.vocab_size != len(vocab):
            raise ValueError("Checkpoint vocabulary size does not match the prepared data")
        outputs = generate_transformer(model, prompts, args.tokens, seeds, args.temperature, args.top_k)
        settings = {"temperature": args.temperature, "top_k": args.top_k, "num_tokens": args.tokens,
                    "checkpoint": repo_path(checkpoint)}
        for i, (prompt, seed, generated) in enumerate(zip(prompts, seeds, outputs)):
            samples.append(sample(vocab, i, seed, prompt, generated, settings,
                                  model="transformer", run=args.transformer_run))
    else:
        print(f"WARNING: no checkpoint at {checkpoint}; Transformer samples omitted.", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {"split": "valid", "prompt_length": args.prompt_length, "seed": args.seed, "samples": samples}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
