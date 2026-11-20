"""T21: score the test split once with every final model, with confidence intervals.

The final models are the add-k n-grams at their best k (T07), the interpolated
trigram (T08), the eight context-ablation Transformers (T18), and the two
k=128 Transformers with their history cut to 6, 32 and 64 tokens (T19). Each
model saves its test predictions as test.npz in its run folder and adds its
test metrics to that folder's metrics.json, next to the validation results.

Every metric in the final table comes with a 95% bootstrap confidence
interval. The test split is cut into 1,000-token blocks; each of the 10,000
resamples draws that many blocks with replacement, and the interval holds the
middle 95% of the resampled values. All models share the same resamples.
Transformer rows average their two seeds: the interval is for that average,
and the seed SD (sample SD, ddof=1) shows how much the two training runs differ.
"""

import csv
import hashlib
import math
import time

import numpy as np
import torch

from lm.ablation import CONTEXTS, SEEDS, file_hash, read_json, write_json
from lm.evaluation import evaluate_run, predictions_path
from lm.metrics import compute_metrics, load_predictions
from lm.ngram import InterpolatedNGram, NGramModel
from lm.transformer.model import load_checkpoint
from lm.transformer.scoring import score_split
from lm.utils import repo_path

NGRAMS = {1: "unigram", 2: "bigram", 3: "trigram"}
TRUNCATED_CONTEXTS = (6, 32, 64)
METRICS = ("cross_entropy_nats", "cross_entropy_bits", "perplexity",
           "top1_accuracy", "top5_accuracy", "top10_accuracy")
TOP_K = (1, 5, 10)
BLOCK_TOKENS = 1000
RESAMPLES = 10_000


def ablation_dir(results_dir, context, seed):
    """Run folder of the T18 Transformer trained with this context and seed."""
    return results_dir / "ablation" / f"k{context}_seed{seed}"


def truncation_dir(results_dir, seed, context):
    """Run folder of the T19 k=128 Transformer scored with a shorter context."""
    return results_dir / "context_truncation/ablation" / f"k128_seed{seed}" / f"k{context}"


def ngram_runs(results_dir):
    """The best add-k n-gram runs, as (name, n, k, run folder)."""
    best_k = read_json(results_dir / "ngram/tuning.json")["best_k"]
    return [(name, n, best_k[name], results_dir / "ngram" / f"n{n}_k{best_k[name]:g}")
            for n, name in NGRAMS.items()]


def model_rows(results_dir):
    """One row per trained model, as (row name, the run folders it averages).

    These are the n-grams and the Transformer at each training context.
    """
    rows = [(f"{name} (k={k:g})", [run_dir]) for name, _, k, run_dir in ngram_runs(results_dir)]
    rows.append(("interpolated trigram", [results_dir / "ngram/interpolated"]))
    for context in CONTEXTS:
        rows.append((f"Transformer k={context}", [ablation_dir(results_dir, context, seed) for seed in SEEDS]))
    return rows


def final_rows(results_dir):
    """The rows of the final table: every trained model, then the T19 shorter-context scorings."""
    rows = model_rows(results_dir)
    for context in TRUNCATED_CONTEXTS:
        rows.append((f"Transformer k=128, scored at k={context}",
                     [truncation_dir(results_dir, seed, context) for seed in SEEDS]))
    return rows


def score_transformers(results_dir, test_ids, eos_id, device, batch_size=256):
    """Score test with every Transformer run that has no test predictions yet.

    Returns a record of each scoring done now: checkpoint, context, device and time.
    """
    runs = []  # (run folder, checkpoint, scoring context)
    for context in CONTEXTS:
        for seed in SEEDS:
            run_dir = ablation_dir(results_dir, context, seed)
            runs.append((run_dir, run_dir / "best.pt", context))
    for seed in SEEDS:
        for context in TRUNCATED_CONTEXTS:
            checkpoint = ablation_dir(results_dir, 128, seed) / "best.pt"
            runs.append((truncation_dir(results_dir, seed, context), checkpoint, context))

    records = {}
    for run_dir, checkpoint, context in runs:
        path = predictions_path(run_dir, "test")
        if not path.exists():
            records[repo_path(run_dir)] = _score_checkpoint(checkpoint, path, test_ids, eos_id, context, device, batch_size)
        evaluate_run(run_dir, "test", len(test_ids))
    return records


def _score_checkpoint(checkpoint, path, test_ids, eos_id, context, device, batch_size):
    print(f"Scoring {repo_path(path)} with context {context} on {device}...", flush=True)
    start = time.time()
    model = load_checkpoint(checkpoint, device)
    score_split(model, test_ids, eos_id, batch_size, path=path, context=context)
    return {
        "checkpoint": repo_path(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "scoring_context": context,
        "device": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "seconds": round(time.time() - start, 1),
    }


def score_ngrams(results_dir, train_ids, test_ids, vocab_size):
    """Score test with the best add-k n-grams and the interpolated trigram, where not done yet."""
    records = {}
    for name, n, k, run_dir in ngram_runs(results_dir):
        path = predictions_path(run_dir, "test")
        if not path.exists():
            print(f"Scoring {repo_path(path)}...", flush=True)
            start = time.time()
            NGramModel(n, vocab_size, k=k).fit(train_ids).score_split(test_ids, path)
            records[repo_path(run_dir)] = {"model": f"{name}, add-k with k={k:g}", "seconds": round(time.time() - start, 1)}
        evaluate_run(run_dir, "test", len(test_ids))

    run_dir = results_dir / "ngram/interpolated"
    path = predictions_path(run_dir, "test")
    if not path.exists():
        print(f"Scoring {repo_path(path)}...", flush=True)
        start = time.time()
        lambdas = read_json(run_dir / "config.json")["lambdas"]
        parts = [NGramModel(n, vocab_size, k=0).fit(train_ids) for n in NGRAMS]
        InterpolatedNGram(parts, lambdas).score_split(test_ids, path)
        records[repo_path(run_dir)] = {"model": f"interpolated trigram, weights {lambdas}",
                                       "seconds": round(time.time() - start, 1)}
    evaluate_run(run_dir, "test", len(test_ids))
    return records


def load_test_predictions(run_dir):
    """Load the test predictions a final run saved in T21, as (logprob, rank)."""
    path = predictions_path(run_dir, "test")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: unzip results/t21_results.zip from the repo root, "
                                "or run scripts/final_evaluation.py")
    return load_predictions(path)


def bootstrap_resamples(num_tokens, seed=0):
    """Block indices of every resample: each draws as many blocks as the split has, with replacement."""
    blocks = math.ceil(num_tokens / BLOCK_TOKENS)
    return np.random.default_rng(seed).integers(0, blocks, size=(RESAMPLES, blocks))


def block_totals(logprob, rank):
    """Per 1,000-token block: number of tokens, summed log-prob and top-k hits."""
    starts = np.arange(0, len(logprob), BLOCK_TOKENS)
    totals = {
        "tokens": np.add.reduceat(np.ones(len(logprob)), starts),
        "logprob": np.add.reduceat(np.asarray(logprob, dtype=np.float64), starts),
    }
    for k in TOP_K:
        totals[f"top{k}"] = np.add.reduceat((np.asarray(rank) < k).astype(np.float64), starts)
    return totals


def resampled_metrics(totals, resamples):
    """The metrics of every resample; resamples[b] lists the blocks drawn for resample b."""
    tokens = totals["tokens"][resamples].sum(axis=1)
    cross_entropy = -totals["logprob"][resamples].sum(axis=1) / tokens
    metrics = {
        "cross_entropy_nats": cross_entropy,
        "cross_entropy_bits": cross_entropy / np.log(2),
        "perplexity": np.exp(cross_entropy),
    }
    for k in TOP_K:
        metrics[f"top{k}_accuracy"] = totals[f"top{k}"][resamples].sum(axis=1) / tokens
    return metrics


def summarize_row(run_dirs, resamples, num_tokens):
    """Mean over the runs, the 95% bootstrap interval of that mean, and the seed SD, for every metric."""
    point, resampled = [], []
    for run_dir in run_dirs:
        logprob, rank = load_predictions(predictions_path(run_dir, "test"))
        if len(logprob) != num_tokens:
            raise ValueError(f"{run_dir} has {len(logprob):,} test predictions, expected {num_tokens:,}")
        point.append(compute_metrics(logprob, rank))
        resampled.append(resampled_metrics(block_totals(logprob, rank), resamples))

    summary = {"runs": len(run_dirs)}
    for metric in METRICS:
        values = [metrics[metric] for metrics in point]
        low, high = np.percentile(np.mean([metrics[metric] for metrics in resampled], axis=0), [2.5, 97.5])
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_ci_low"] = float(low)
        summary[f"{metric}_ci_high"] = float(high)
        summary[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return summary


def build_final_table(results_dir, out, test_ids, seed=0):
    """Write the final table (final.md, .csv, .json) and its settings (config.json) to out.

    Returns the Markdown table.
    """
    resamples = bootstrap_resamples(len(test_ids), seed)
    rows = [{"model": name, **summarize_row(run_dirs, resamples, len(test_ids))}
            for name, run_dirs in final_rows(results_dir)]

    write_json(out / "config.json", {
        "split": "test",
        "test_tokens": len(test_ids),
        "test_sha256": hashlib.sha256(np.asarray(test_ids, dtype="<i4").tobytes()).hexdigest(),
        "bootstrap": {"block_tokens": BLOCK_TOKENS, "blocks": resamples.shape[1], "resamples": RESAMPLES,
                      "seed": seed, "interval": "95% percentile"},
        "rows": {name: [repo_path(run_dir) for run_dir in run_dirs] for name, run_dirs in final_rows(results_dir)},
    })
    write_json(out / "final.json", rows)
    return write_final_table(rows, out, len(test_ids))


def write_final_table(rows, out, num_tokens):
    """Render saved bootstrap summaries without loading or rescoring test predictions."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "final.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = _markdown(rows, num_tokens)
    (out / "final.md").write_text(markdown, encoding="utf-8")
    return markdown


def _markdown(rows, num_tokens):
    lines = [
        "# Final test-set results",
        "",
        f"Test split ({num_tokens:,} tokens), scored once with every final model. The 95% CIs come from "
        f"{RESAMPLES:,} bootstrap resamples of {BLOCK_TOKENS:,}-token blocks. Transformer rows are the mean "
        "of two seeds, and the seed SD is the sample SD between them.",
        "",
        "| Model | Perplexity | 95% CI | Seed SD | CE (nats) | 95% CI | Top-1 | 95% CI | Top-5 | Top-10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        sd = row["perplexity_seed_sd"]
        cells = [
            row["model"],
            f"{row['perplexity']:.2f}", _interval(row, "perplexity", "{:.2f}"), "-" if sd is None else f"{sd:.2f}",
            f"{row['cross_entropy_nats']:.4f}", _interval(row, "cross_entropy_nats", "{:.4f}"),
            f"{row['top1_accuracy']:.2%}", _interval(row, "top1_accuracy", "{:.2%}"),
            f"{row['top5_accuracy']:.2%}", f"{row['top10_accuracy']:.2%}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _interval(row, metric, fmt):
    return f"[{fmt.format(row[f'{metric}_ci_low'])}, {fmt.format(row[f'{metric}_ci_high'])}]"
