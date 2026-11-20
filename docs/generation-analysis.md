# Full generation analysis (T24)

The [table](../results/generation/ablation/metrics.md) covers the four frozen
n-gram baselines (including interpolation) and all eight T18 checkpoints.
[Representative continuations](../results/generation/ablation/representative_samples.md)
show prompt indices 0, 1 and 2 for every run, selected before examining outputs.
The complete 360 continuations and their token IDs are in
`results/generation/ablation/samples.json`.

## Reproduce

After setup and `python scripts/prepare_data.py`, generate on CPU with:

```bash
python scripts/generation_analysis.py --config configs/generation.json --device cpu
```

The saved run used `--device cuda:1` on the AMD Radeon RX 9070 XT. Use your own
device index; see the [device check](context-ablation.md#setup-and-device-check).
Generation uses the committed checkpoints and frozen n-gram parameters; it
does not train, select hyperparameters, or score the test split. Missing
checkpoints or a different training vocabulary/token stream fail before generation.

To rebuild the table and representative examples from the recorded tokens:

```bash
python scripts/generation_analysis.py --samples results/generation/ablation/samples.json
```

`--out` selects an output directory for either command. T14/T15's original
`results/generation/samples.json` and tables remain available separately.

## Protocol and interpretation

`configs/generation.json` fixes 30 validation prompts of 10 tokens, 100 new
tokens per sample, prompt-selection seed 42 and sampling seed `42 + prompt_index`.
Every model gets the same prompts. An n-gram sees at most n-1 preceding tokens;
a Transformer sees at most its training context. Generation continues through
`<eos>` until all 100 new tokens have been sampled.

N-grams sample their entire probability distribution. Transformers use temperature
1 and top-k 50, matching T14. Consequently this compares the specified generation
procedures as well as the models; diversity differences cannot be attributed only
to architecture. Model settings, checkpoint hashes, data fingerprints, training
seeds and runtime versions are recorded alongside the outputs. Local generators
make sampling repeatable for identical probabilities. Different devices or library
builds can change logits and thus sampled sequences.

Each table row is the macro-average of the 30 per-sample scores, computed only on
newly generated tokens. Distinct-1/2 divides the number of unique n-grams by all
n-grams; repetition counts repeated 4-grams after their first occurrence. A sample
shorter than the corresponding n receives zero. Training seeds stay separate to
show variation; there are no confidence intervals for these descriptive scores.

The add-k trigram has almost maximal diversity (Distinct-1 0.9983, Distinct-2 1.0000),
but its examples mostly consist of unrelated words. Sparse histories and smoothing
can produce diverse yet incoherent text. Transformer Distinct-1 is roughly
0.59–0.65, with repetition-4 below 0.015 for every run. Longer context does not
monotonically improve these measures: for example, the k=64 seed-0 run repeats
more than seed 1. Inspect the continuations alongside the held-out prediction
metrics; high diversity or low repetition alone does not establish better language.
