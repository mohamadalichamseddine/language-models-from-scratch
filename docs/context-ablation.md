# T18: context-window training experiments

Eight equal-budget runs (two seeds at each context) were completed on
2026-11-14 on an RX 9070 XT. Their configurations, selected checkpoints,
training logs, and validation metrics are in `results/ablation/`.
The test split was not scored. T19 context truncation and T20 figures
are delivered in separate follow-up PRs.

The [environment record](../results/ablation/environment.json) preserves
the original experiment's package versions and source fingerprints.
The experiments preceded the split into task-specific branches.

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
