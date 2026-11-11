
"""T14: generate fixed-seed continuations from all baselines.

Usage: python scripts/generate.py --transformer-run transformer/best
Creates results/generation/samples.json. If a checkpoint has not been
provided, generates the three n-gram baselines and explicitly reports the
missing Transformer (run again after training).
"""

import argparse
import json
from pathlib import Path

from lm.data import load_split, load_vocab
from lm.evaluation import RESULTS_DIR
from lm.generation import generate_ngram, generate_transformer, select_prompts
from lm.ngram import NGramModel
from lm.transformer.model import load_checkpoint

CHECKPOINTS_DIR = RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible continuations from n-gram and Transformer models"
    )
    parser.add_argument("--transformer-run", default="transformer/best")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoints/<transformer-run>/best.pt"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "generation/samples.json"
    )
    parser.add_argument("--prompts", type=int, default=30)
    parser.add_argument("--prompt-length", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be positive")

    vocab = load_vocab()
    prompts = select_prompts(
        load_split("valid"),
        args.prompts,
        args.prompt_length,
        args.seed
    )

    train = load_split("train")
    tuning = json.loads(
        (RESULTS_DIR / "ngram/tuning.json").read_text(encoding="utf-8")
    )

    samples = []

    for n, name in enumerate(("unigram", "bigram", "trigram"), 1):
        k = float(tuning["best_k"][name])

        print(f"Fitting and generating {name} (k={k:g})...", flush=True)

        model = NGramModel(n, len(vocab), k=k).fit(train)

        for i, prompt in enumerate(prompts):
            generated = generate_ngram(
                model,
                prompt,
                args.tokens,
                args.seed + i
            )

            samples.append({
                "model": name,
                "prompt_index": i,
                "seed": args.seed + i,
                "prompt_ids": prompt,
                "prompt": vocab.decode(prompt),
                "generated_ids": generated,
                "generated": vocab.decode(generated),
                "settings": {
                    "smoothing": "add-k",
                    "k": k,
                    "num_tokens": args.tokens
                }
            })

        del model

    checkpoint = (
        args.checkpoint
        or CHECKPOINTS_DIR / args.transformer_run / "best.pt"
    )

    if checkpoint.is_file():
        print(
            f"Generating Transformer from {checkpoint}...",
            flush=True
        )

        model = load_checkpoint(checkpoint, args.device)

        if len(vocab) != model.config.vocab_size:
            raise ValueError(
                "Checkpoint vocabulary size does not match prepared data"
            )

        # Batch prompts for efficient CPU/GPU generation.
        import torch

        model.eval()

        contexts = [list(p) for p in prompts]
        outputs = [[] for _ in prompts]

        generators = [
            torch.Generator(device="cpu").manual_seed(args.seed + i)
            for i in range(len(prompts))
        ]

        with torch.inference_mode():
            for step in range(args.tokens):
                windows = torch.tensor(
                    [c[-model.config.context:] for c in contexts],
                    dtype=torch.long,
                    device=args.device
                )

                logits = model.next_token_logits(windows)

                values, indices = torch.topk(
                    logits / args.temperature,
                    min(args.top_k, logits.size(-1)),
                    dim=-1
                )

                probs = torch.softmax(
                    values.float().cpu(),
                    dim=-1
                )

                for i in range(len(prompts)):
                    sampled = torch.multinomial(
                        probs[i],
                        1,
                        generator=generators[i]
                    ).item()

                    token = int(indices[i, sampled].item())

                    contexts[i].append(token)
                    outputs[i].append(token)

        for i, prompt in enumerate(prompts):
            generated = outputs[i]

            samples.append({
                "model": "transformer",
                "run": args.transformer_run,
                "prompt_index": i,
                "seed": args.seed + i,
                "prompt_ids": prompt,
                "prompt": vocab.decode(prompt),
                "generated_ids": generated,
                "generated": vocab.decode(generated),
                "settings": {
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "num_tokens": args.tokens,
                    "checkpoint": str(checkpoint)
                }
            })

    else:
        print(
            f"WARNING: checkpoint not found at {checkpoint}; "
            "Transformer samples omitted.",
            flush=True
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    args.out.write_text(
        json.dumps(
            {
                "split": "valid",
                "prompt_length": args.prompt_length,
                "seed": args.seed,
                "samples": samples
            },
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
