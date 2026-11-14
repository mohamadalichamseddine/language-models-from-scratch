# T18-T20: context-window experiments

The eight T18 runs, both-seed T19 scoring, and T20 figures were completed on
2026-11-14 on an RX 9070 XT.
The [validation summary](../results/figures/ablation_summary.md) reports two seeds
at each context. Context 64 achieved the lowest mean perplexity (104.13), compared
with 138.20, 109.19, and 107.58 at contexts 6, 32, and 128 respectively. These are
validation results from this configuration and two seeds, not a general claim
that context 64 is optimal. The commands below never score the test split.

T19 comparisons are saved for [seed 0](../results/context_truncation/ablation/k128_seed0/comparison.md)
and [seed 1](../results/context_truncation/ablation/k128_seed1/comparison.md).
Both show a large perplexity penalty when the context-128 model is restricted
to six tokens (about 148 versus 107-108 with the full history).
The [artifact verification](../results/ablation/validation.json) checks all 16
prediction files and records numerical differences between repeated GPU scoring
passes. All 107 tests passed. The [environment record](../results/ablation/environment.json)
contains package versions, GPU details, and source fingerprints.

## Experiment protocol

| Setting | Value |
|---|---|
| Training contexts | 6, 32, 64, 128 tokens |
| Seeds | 0 and 1 for every context |
| Hyperparameters | Frozen T13 settings in `configs/transformer.json` |
| Tokens per step | 12,288 |
| Training steps | 4,000 |
| Tokens processed per run | 49,152,000 |
| Checkpoint selection | Lowest quick validation loss during the fixed budget |
| Reported metrics | Full validation split, sliding-window scoring of the selected checkpoint |

T18 disables early stopping while retaining best-checkpoint selection. This is
the explicit exception to the T13 training settings needed to meet the roadmap's
equal-budget requirement. `patience` is retained in saved configuration but is
inactive. The existing T13 checkpoint stopped at 3,800 steps, so it is not reused
as a completed T18 run. Best checkpoints can come from different steps even though
every run receives the same total training budget; `training.json` records both.

## Setup and device check

Use the repository environment from the README. On PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/prepare_data.py
python -m pytest -q
python -c "import torch; print(torch.__version__); print('HIP:', torch.version.hip); print('GPU:', torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

AMD ROCm builds of PyTorch also expose GPU devices through `torch.cuda`; use
`--device cuda` or `--device cuda:0` when the device check succeeds. A CPU-only
PyTorch build cannot use an AMD GPU. For native Windows, select the supported
GPU, OS, Python, runtime, and driver combination from the
[AMD compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
and [AMD PyTorch installation instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
Installing the Python packages does not update the graphics driver.

On the development machine, the RX 9070 XT was verified on 2026-11-14 using
Python 3.12 and `torch==2.14.0+rocm10.1.0`. After installing the repository's
requirements, the AMD wheel was installed into `.venv` with:

```powershell
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0"
```

The device check listed the integrated graphics at index 0 and the RX 9070 XT
at index 1. **Use `--device cuda:1` on this machine** in the commands below.
A five-step full-size-model training smoke test and scoring at all four contexts
passed on that device. Other machines may expose a different device order.

Use the CPU for unit tests and small smoke runs. Full training should use a
working GPU runtime. If several GPUs are exposed, select the intended device
explicitly. A scoring batch-size change affects memory use, not the training
token budget.

## T18: train and score

Preview the eight effective configurations without creating outputs:

```powershell
python scripts/run_ablation.py --dry-run
```

Run one context/seed first, then the full grid:

```powershell
python scripts/run_ablation.py --contexts 6 --seeds 0 --device cuda
python scripts/run_ablation.py --device cuda
```

The runner loads the shared prepared data, trains each selected run, and scores
every validation token. Outputs go to `results/ablation/k<context>_seed<seed>/`:

- `config.json`: effective model/training settings, budget, token counts, and
  data/vocabulary hashes.
- `curves.csv`, `training.json`, `best.pt`: training curves, actual budget and
  best step, and selected checkpoint.
- `valid.npz`, `metrics.json`, `scoring.json`: aligned predictions, validation
  metrics, and scoring context/checkpoint/data identity.

Re-running the command keeps completed matching training runs and intact matching
predictions. An interrupted training run restarts from its seed at step zero:
`best.pt` is not an optimizer-resume checkpoint. Conflicting settings/data or a
missing checkpoint in a completed run produce an error; use a new `--out`
directory for a changed experiment. If scoring runs out of memory, retry with
`--batch-size 64`; completed training remains reusable.

For code smoke tests, pass a separate tiny JSON with `--config` and put outputs
outside `results/ablation/` using `--out`. Do not modify the frozen T13 file to
shorten the real experiment, and do not mix smoke outputs into the final plots.

## T19: restrict the available history

After the k=128 runs finish, score the same checkpoint at k=6, 32, 64, and 128:

```powershell
python scripts/score_context_truncation.py --run ablation/k128_seed0 --device cuda
python scripts/score_context_truncation.py --run ablation/k128_seed1 --device cuda
```

For an earlier exploratory result, `--run transformer/best` uses the existing
T13 checkpoint. Keep that comparison labeled as T13; it has a different training
budget from T18.

The scoring override changes only the visible history. It preserves the trained
context setting, weights, and positional embedding table; positions start at zero
within each window, as in existing scoring. All settings predict the same targets,
including the first token through the prepended `<eos>` marker.

Repeated GPU scoring can differ at floating-point rounding precision, including
the order of nearly tied predictions. Compare numerical tolerances and aggregate
metrics rather than requiring bitwise equality. The full-context seed-0 repeat
changed perplexity by approximately 0.00000016 with identical top-1/5/10 accuracies.

Results go into `results/context_truncation/<source-run>/k<context>/`, with a
comparison Markdown/CSV table in the source-run folder. Full-context predictions
in the training folder are not overwritten.

## T20: aggregate and plot

```powershell
python scripts/plot_ablation.py
```

The default requires all eight T18 runs and checks matching settings/data,
completed budgets, and full-context validation scoring metadata. It writes:

- `results/figures/ablation_runs.csv`: each context/seed measurement.
- `ablation_summary.json`, `.csv`, `.md`: mean and sample standard deviation
  across seeds, plus counts.
- `ablation_perplexity.png`, `ablation_cross_entropy_nats.png`,
  `ablation_top1_accuracy.png`: mean curves with SD error bars and tuned n-gram
  reference lines. Perplexity uses a logarithmic axis.
- `ablation_status.json`: requested grid, missing runs, and seed coverage.

Perplexities are averaged across seeds directly; they are not computed by
exponentiating mean CE. Error bars use sample SD (`ddof=1`), not standard error
or confidence intervals. With one seed SD is undefined and is reported as `n/a`.

For an explicitly partial preview in a separate directory:

```powershell
python scripts/plot_ablation.py --allow-partial --out results/figures/ablation_preview
```

Do not mark T18 or T20 Done based on a partial preview. Preserve each selected
checkpoint for later tickets and commit the intended configs, summaries, and
figures. Prediction arrays remain ignored and can be regenerated.
