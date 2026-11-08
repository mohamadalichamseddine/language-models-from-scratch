"""Training loop for the Transformer, with run logging.

train() runs AdamW with a warmup + cosine learning-rate schedule and gradient
clipping. Every eval_every steps it measures the validation loss, saves the
model when that loss is the best so far, and stops early when it has not
improved for `patience` evaluations in a row. The run folder gets:

    curves.csv      step, tokens, lr, train loss and val loss at every evaluation
    training.json   best step and val loss, steps trained, time and hardware
    best.pt         the model at its best val loss (see lm.transformer.model)

On a GPU, training uses mixed precision (float16), which is several times
faster on Colab's T4.
"""

import csv
import json
import math
import platform
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lm.transformer.batches import BatchSampler
from lm.transformer.model import save_checkpoint


@dataclass
class TrainingConfig:
    # 12,288 is a multiple of every context size we train (6, 32, 64, 128).
    tokens_per_batch: int = 12_288
    max_steps: int = 4000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1  # the schedule ends at lr * min_lr_ratio
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    patience: int = 5  # evaluations without a new best val loss before stopping
    seed: int = 0


def learning_rate(step, config):
    """Linear warmup to config.lr, then cosine decay to lr * min_lr_ratio at max_steps."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
    min_lr = config.lr * config.min_lr_ratio
    return min_lr + (config.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, config):
    """AdamW with weight decay on weight matrices and embeddings, not on biases or LayerNorm."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95))


def mixed_precision(device):
    """float16 autocast on a GPU; does nothing on the CPU."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def train_step(model, optimizer, scaler, inputs, targets, grad_clip):
    """One optimizer step on a batch. Returns the batch loss (before the step)."""
    with mixed_precision(inputs.device):
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()


@torch.no_grad()
def evaluate_loss(model, ids, tokens_per_batch):
    """Mean next-token loss (nats) over ids, cut into non-overlapping windows.

    This is a quick estimate for tracking training: a token near the start of
    a window sees only a few previous tokens. The reported scores come from
    lm.transformer.scoring, where every token sees its full context.
    """
    context = model.config.context
    device = next(model.parameters()).device
    ids = torch.as_tensor(np.asarray(ids, dtype=np.int64))

    num_windows = (len(ids) - 1) // context
    inputs = ids[: num_windows * context].view(num_windows, context)
    targets = ids[1 : num_windows * context + 1].view(num_windows, context)
    batch_size = max(1, tokens_per_batch // context)

    was_training = model.training
    model.eval()
    total = 0.0
    for start in range(0, num_windows, batch_size):
        x = inputs[start : start + batch_size].to(device)
        y = targets[start : start + batch_size].to(device)
        with mixed_precision(device):
            logits = model(x)
        total += F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum").item()
    model.train(was_training)

    return total / (num_windows * context)


def train(model, train_ids, val_ids, config, run_dir, device="cpu"):
    """Train the model, log the run in run_dir and keep the best model as run_dir/best.pt.

    Returns the summary that is also saved as run_dir/training.json.
    """
    device = torch.device(device)
    model.to(device).train()
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler = BatchSampler(train_ids, model.config.context, config.tokens_per_batch, seed=config.seed, device=device)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    curves = []
    best_step, best_val_loss = None, math.inf
    evals_without_improvement = 0
    train_loss_sum, train_batches = 0.0, 0
    stopped_early = False
    start_time = time.time()

    for step in range(config.max_steps + 1):
        if step % config.eval_every == 0 or step == config.max_steps:
            val_loss = evaluate_loss(model, val_ids, config.tokens_per_batch)
            # Mean training loss over the steps since the previous evaluation.
            train_loss = float(train_loss_sum) / train_batches if train_batches else None
            train_loss_sum, train_batches = 0.0, 0

            curves.append({
                "step": step,
                "tokens": step * config.tokens_per_batch,
                "lr": learning_rate(step, config),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.time() - start_time, 1),
            })
            _write_curves(run_dir / "curves.csv", curves)
            train_text = f"{train_loss:.4f}" if train_loss is not None else "  -   "
            print(f"step {step:5d} | train {train_text} | val {val_loss:.4f} | {curves[-1]['seconds']:.0f}s", flush=True)

            if val_loss < best_val_loss:
                best_step, best_val_loss = step, val_loss
                evals_without_improvement = 0
                save_checkpoint(model, run_dir / "best.pt", step=step, val_loss=val_loss)
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= config.patience:
                    stopped_early = True
                    break

        if step == config.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, config)
        inputs, targets = sampler.next_batch()
        train_loss_sum += train_step(model, optimizer, scaler, inputs, targets, config.grad_clip)
        train_batches += 1

    summary = {
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "steps": step,
        "stopped_early": stopped_early,
        "tokens_seen": step * config.tokens_per_batch,
        "train_seconds": round(time.time() - start_time, 1),
        "device": _device_name(device),
        "torch_version": torch.__version__,
    }
    (run_dir / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_curves(path, curves):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def _device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU ({platform.processor() or platform.machine()})"
