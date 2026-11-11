
"""Reproducible free-running generation and diversity metrics."""

from collections import Counter

import numpy as np


def select_prompts(ids, count=30, prompt_length=10, seed=42):
    """Draw nonduplicate contiguous snippets from validation with a local RNG."""
    ids = np.asarray(ids)
    if prompt_length < 1 or count < 1 or len(ids) < prompt_length:
        raise ValueError("Invalid prompt count/length or too few validation tokens")
    possible = len(ids) - prompt_length + 1
    if count > possible:
        raise ValueError("More prompts requested than possible start offsets")
    rng = np.random.default_rng(seed)
    starts = rng.choice(possible, size=count, replace=False)
    return [ids[int(i):int(i) + prompt_length].tolist() for i in starts]


def sample_probs(probs, rng, temperature=1.0, top_k=None):
    """Sample a categorical distribution with optional temperature and top-k truncation."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or len(probs) == 0 or np.any(probs < 0) or not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        raise ValueError("Expected a nonempty finite probability vector")
    if top_k is not None and not 1 <= top_k <= len(probs):
        raise ValueError("top_k must be between 1 and vocabulary size")
    # Work in log-space to support highly peaked probabilities.
    logits = np.full(len(probs), -np.inf)
    nonzero = probs > 0
    logits[nonzero] = np.log(probs[nonzero]) / temperature
    if top_k is not None:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        sub = logits[indices]
    else:
        indices = np.arange(len(probs))
        sub = logits
    weights = np.exp(sub - np.max(sub))
    weights /= weights.sum()
    return int(rng.choice(indices, p=weights))


def generate_ngram(model, prompt_ids, length, seed):
    rng = np.random.default_rng(seed)
    context = list(prompt_ids)
    output = []
    for _ in range(length):
        token = sample_probs(model.distribution(context), rng)
        output.append(token)
        context.append(token)
    return output


def generate_transformer(model, prompt_ids, length, seed, temperature=1.0, top_k=50):
    import torch

    if temperature <= 0 or not np.isfinite(temperature) or top_k < 1:
        raise ValueError("Invalid sampling temperature or top_k")
    if not prompt_ids:
        raise ValueError("Prompt cannot be empty")
    device = next(model.parameters()).device
    # CPU RNG ensures identical sampling streams on CPU/GPU (given identical logits).
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    context = list(prompt_ids)
    output = []
    model.eval()
    with torch.inference_mode():
        for _ in range(length):
            window = context[-model.config.context:]
            logits = model.next_token_logits(torch.tensor([window], dtype=torch.long, device=device))[0]
            values, indices = torch.topk(logits / temperature, min(top_k, logits.numel()))
            probabilities = torch.softmax(values.float().cpu(), dim=-1)
            selection = torch.multinomial(probabilities, 1, generator=rng).item()
            token = int(indices[selection].item())
            output.append(token)
            context.append(token)
    return output


def generation_metrics(ids):
    """Distinct-n = unique n-grams / all n-grams; repetition = duplicate 4-grams / all 4-grams.

    Samples shorter than n receive 0 for that metric. Metrics use generated
    tokens only, not the prompt; an n-gram repeated three times counts twice.
    """
    ids = list(ids)
    def grams(n):
        return [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
    def distinct(n):
        g = grams(n)
        return len(set(g)) / len(g) if g else 0.0
    g4 = grams(4)
    return {
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "repetition_4": (len(g4) - len(set(g4))) / len(g4) if g4 else 0.0,
    }


def aggregate_generation(samples):
    """Macro-average sample-level scores, keeping all three denominators comparable."""
    if not samples:
        raise ValueError("No generated samples")
    rows = [generation_metrics(sample) for sample in samples]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
