# Implementation Plan

We compare classical n-gram language models (MLE unigram, bigram, trigram with add-k smoothing) against a small decoder-only Transformer, both written from scratch and trained on WikiText-2. We then vary the Transformer's context window (6, 32, 64, 128 tokens).

**Research questions**
1. How do the n-gram models compare with the Transformer on perplexity, cross-entropy, next-token accuracy and generation quality?
2. How does the Transformer's context window (6 / 32 / 64 / 128) change language-modeling performance and generated text?
3. When does longer context help most, especially for rare tokens and predictions that depend on more than the local context?

## Milestones

| Milestone | Goal |
|---|---|
| Checkpoint 1 | Data pipeline working, shared output format and metrics in place, first n-gram baseline scored on validation |
| Checkpoint 2 | N-gram baselines tuned and the Transformer trained, all scored by the same evaluator; first results, figures and generated samples |
| Checkpoint 3 | Context-window ablations, final test-set results, rare-token / long-context / generation analyses, and a reproducible codebase |

## Rules for every ticket

- **One vocabulary, one token stream.** Every model uses the vocabulary and encoded splits from T03. Scores can only be compared if all models predict the same targets.
- **Tune on validation, touch test once.** The test split is used only in T21 (final evaluation).
- **One output format.** Every model writes per-token predictions in the format from T04, so evaluation and analysis code never depends on model internals.
- **Reproducible runs.** Fixed seeds, and every run saves its config next to its results.

---

## Checkpoint 1

| ID | Title | Description | Depends on | Owner | Status |
|---|---|---|---|---|---|
| T01 | Repo skeleton and environment | Create the layout: `src/` (package code), `scripts/` (entry points), `configs/`, `tests/`, `results/`. Add `requirements.txt` (Python 3.10+, torch, numpy, datasets, matplotlib, pytest), a seed helper, and a README stub with setup steps. Decide where training runs (local GPU or Colab) and note it in the README. `.gitignore` already exists (it ignores `docs/internal/`, `data/processed/` and checkpoints).<br>**Done when:** fresh clone → `pip install -r requirements.txt` → `pytest` passes. | — | Mohamad Ali | Done |
| T02 | Add WikiText-2 to the repo and compute dataset statistics | Download `Salesforce/wikitext`, config `wikitext-2-v1` (the tokenized version that already contains `<unk>`, with the official train/val/test split), and commit the three splits as plain text in `data/raw/wikitext-2/` (about 13 MB). Add `data/README.md` with the source, the citation (Merity et al., 2016) and the license (CC BY-SA 3.0, which allows sharing with credit). Keep the download script so the files can be re-created. Stats per split: lines, articles, tokens and `<unk>` rate; also vocab size, the top-20 tokens and a rank-frequency (Zipf) plot.<br>**Done when:** the raw files and `data/README.md` are committed, and the stats table and plot are saved in `results/data_stats/`. | T01 | Mohamad Ali | Done |
| T03 | Tokenization, vocabulary and encoded splits | Split on whitespace, keeping the given tokens as they are, and add `<eos>` at the end of each line. Build the vocabulary from train only. Save `vocab.json` plus each split as one id array (`train.npy`, `val.npy`, `test.npy`) in `data/processed/`; these are generated, not committed. They are the only data source for every model.<br>**Done when:** decode(encode(x)) round-trips; token counts and vocab size match the published WikiText-2 numbers (2,088,628 / 217,646 / 245,569 tokens, 33,278 vocab) or the difference is explained; the `<eos>` convention is documented in the code. | T02 | | Not started |
| T04 | Prediction format and metrics module | Small module used by every model and by the evaluator. **(a) Output format:** save/load helpers for per-token predictions on a split, storing the target's log-prob and its rank in the model's predicted distribution (e.g. `.npz` with `logprob`, `rank`). **(b) Metrics** computed from that file over every target token of the split (including `<eos>`): cross-entropy in nats and bits per token, perplexity = exp(CE), and top-k accuracy = `mean(rank < k)` for k = 1, 5, 10. Put the definitions in docstrings. Unit tests against hand-computed toy examples.<br>**Done when:** tests pass and a teammate has reviewed the format. | T01 | | Not started |
| T05 | N-gram models: MLE and add-k (uni/bi/tri) | From scratch: count n-grams on the train ids and use P(w\|h) = (c(h,w) + k) / (c(h) + k·V). k = 0 gives MLE; a history never seen in train gives a uniform distribution. Add a function that scores a split and writes the T04 output file, breaking rank ties by unigram frequency. Unit test: for sample histories, probabilities sum to 1 over the vocab.<br>**Done when:** tests pass and val perplexity for n = 1, 2, 3 with k = 1 is saved in `results/`. | T03, T04 | | Not started |

## Checkpoint 2

| ID | Title | Description | Depends on | Owner | Status |
|---|---|---|---|---|---|
| T06 | Evaluator script and comparison table | Command-line script that reads one or more runs' T04 output files, computes their metrics with the T04 module, writes `metrics.json` per run, and builds one comparison table (Markdown + CSV) across runs.<br>**Done when:** it reproduces the n-gram numbers from T05. | T04, T05 | | Not started |
| T07 | Tune add-k on validation | Grid over k (e.g. 1, 0.1, 0.01, 0.001, 0.0001) for n = 1, 2, 3. Pick the best k for each n by val perplexity and plot val perplexity against k. Also report unsmoothed MLE: the share of val tokens that get zero probability (this shows why smoothing is needed).<br>**Done when:** the chosen k values and the plot are in `results/ngram/`. | T05, T06 | | Not started |
| T08 | (Optional) Interpolated trigram baseline | λ1·P_uni + λ2·P_bi + λ3·P_tri, with the λ weights tuned on val. Add-k is a weak baseline, so this gives the Transformer a stronger classical comparison.<br>**Done when:** its val metrics are in the comparison table. | T07 | | Not started |
| T09 | Transformer model from scratch | PyTorch, without `nn.Transformer` or Hugging Face models. Components: token and learned positional embeddings, causal multi-head self-attention, GELU MLP, pre-LayerNorm residual blocks, dropout, and tied input/output embeddings. A reasonable starting size is 4 layers, 4 heads, d_model 256 (about 12M parameters, most of them in the embeddings). Tests: output shapes; changing a future token never changes earlier logits (causal mask check); parameter count printed.<br>**Done when:** tests pass. | T01 | | Not started |
| T10 | Training data loader | Sample contiguous windows of k + 1 ids from `train.npy`: the first k are inputs and the targets are the same window shifted by one. Set batch size in tokens, so every context size sees the same number of tokens per step. Sampling is deterministic given a seed.<br>**Done when:** a unit test confirms targets are inputs shifted by one. | T03 | | Not started |
| T11 | Training loop and run logging | AdamW, warmup + cosine learning-rate schedule, gradient clipping, val loss every N steps, early stopping, and saving the best checkpoint. Each run saves its config, train/val curves (CSV) and metrics to `results/<run>/`. Sanity checks: initial loss ≈ ln(33,278) ≈ 10.4, and the model can overfit one batch to near-zero loss.<br>**Done when:** both sanity checks pass and a short run lowers val loss. | T09, T10 | | Not started |
| T12 | Transformer scoring (sliding window) | Score a split with context k using a sliding window, so every token sees min(position, k) previous tokens (batched for speed), and write the T04 output file.<br>**Done when:** on a short sequence it matches scoring token by token, and T06 reads its output. | T04, T06, T09 | | Not started |
| T13 | First full Transformer run + small tuning sweep | Train at k = 128 and run a small sweep of learning rate and dropout on val (WikiText-2 overfits quickly). Record training time and hardware. Freeze the resulting hyperparameters for all ablation runs.<br>**Done when:** the best config and its val metrics are in the comparison table. | T11, T12 | | Not started |
| T14 | Text generation for all models | Generation script. Use the same prompts for every model (e.g. 30 ten-token snippets from val), generate 100 new tokens each, with fixed seeds. N-grams sample from their smoothed distribution; the Transformer uses temperature + top-k sampling (record the settings). Save outputs as JSON with model, prompt and settings.<br>**Done when:** samples exist for the three n-gram models and the T13 model. | T05, T13 | | Not started |
| T15 | Generation metrics | Implement Distinct-1/2 and repetition rate (share of 4-grams in a sample that repeat an earlier one), with unit tests on small hand-made examples. Produce one table across models. The metric code and tests can start before T14 is done.<br>**Done when:** the table for the T14 samples is in `results/generation/`. | T14 | | Not started |
| T16 | Results table and figures script | One command that builds the val-set table (CE, perplexity and top-1/5/10 accuracy for unigram, bigram and trigram at their best k, interpolated if T08 is done, and the Transformer) plus figures: Transformer train/val loss curves and a bar chart of val perplexity per model.<br>**Done when:** the table and figures are generated into `results/figures/`. | T07, T13 | | Not started |
| T17 | Model disagreement tool | Script that takes two models' T04 output files and lists the tokens where their log-probs differ most, each printed with its surrounding context.<br>**Done when:** it prints the top-N tokens where the Transformer beats the trigram, and the top-N where the trigram wins. | T05, T13 | | Not started |

## Checkpoint 3

| ID | Title | Description | Depends on | Owner | Status |
|---|---|---|---|---|---|
| T18 | Context-window ablation runs | Train with k = 6, 32, 64 and 128, using the frozen T13 hyperparameters and the same token budget (tokens per step × number of steps), so differences come from context alone. Use at least 2 seeds per k if compute allows. Save best checkpoints and per-token val outputs. Runs can be split between people by k.<br>**Done when:** all runs and their configs are in `results/ablation/`. | T13 | | Not started |
| T19 | (Optional) Eval-time context truncation | Score the k = 128 model with its context cut to 6, 32 and 64 tokens. This separates being trained with long context from being allowed to use it at test time, and it costs no extra training. | T12, T18 | | Not started |
| T20 | Ablation plots | Perplexity, CE and top-1 accuracy against context size (mean ± std across seeds). Show the n-gram results as reference lines; a trigram uses 2 tokens of context.<br>**Done when:** the figures are in `results/figures/`. | T18 | | Not started |
| T21 | Final test-set evaluation | Once every choice is frozen, score the test split once with each final model. Build a table with 95% bootstrap confidence intervals (resampling test articles or 1k-token blocks) and the std across seeds. Nothing gets tuned after this.<br>**Done when:** the final table is in `results/final/`. | T07, T18 (+ T08, T19 if done) | | Not started |
| T22 | Rare-token analysis (RQ3) | Using the T21 test outputs, group target tokens by train frequency (e.g. 1–10, 11–100, 101–1k, >1k; `<unk>` reported on its own). Report CE and top-1 accuracy per group for each model.<br>**Done when:** the script produces the per-group table or figure and prints 3–5 examples. | T21 | | Not started |
| T23 | Long-context analysis (RQ2, RQ3) | Compute the per-token loss difference between the k = 128 and k = 6 models on test (building on T17). Group tokens by whether the target appeared within the previous 128 tokens but not the previous 6 (copy-like cases, e.g. names repeated in an article), and by function vs. content words. Show where more context helps and where it doesn't, and compare with Khandelwal et al. (2018).<br>**Done when:** the script produces the grouped table or figure and prints examples. | T17, T21 | | Not started |
| T24 | Full generation analysis | Run T14 and T15 for every ablation model and every n-gram model. Produce a table of Distinct-1/2 and repetition rate by model and context size, and save representative samples.<br>**Done when:** the table and samples are in `results/generation/`. | T15, T18 | | Not started |
| T25 | (Optional) Efficiency and KV cache | Measure training time, scoring time and generation speed. Add KV caching to Transformer generation, check that it produces the same tokens as uncached generation, and report tokens/sec with and without it. | T14, T18 | | Not started |
| T26 | Reproducibility cleanup | README with setup steps and the exact command behind every table and figure. Commit all configs and seeds, pin requirements, and add one `run_all` script. Add a script that bundles all source code into a single `.txt` file.<br>**Done when:** a teammate reproduces one table from a fresh clone. | T21 | | Not started |

---

*Last updated: 2026-10-23 17:14 EDT*
