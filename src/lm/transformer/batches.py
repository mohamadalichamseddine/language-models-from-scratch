"""Training batches: random windows of the training token stream."""

import numpy as np
import torch


class BatchSampler:
    """Draws random contiguous windows of context + 1 token ids.

    The first context ids of a window are the inputs, and the targets are the
    same window shifted by one, so targets[:, i] is the token after inputs[:, i].

    The batch size is set in tokens, so every context size trains on the same
    number of tokens per step: batch_size = tokens_per_batch // context.
    Window starts come from the sampler's own seeded generator, so the same
    seed always gives the same batches.
    """

    def __init__(self, ids, context, tokens_per_batch, seed=0, device="cpu"):
        if tokens_per_batch % context != 0:
            raise ValueError(f"tokens_per_batch ({tokens_per_batch}) must be a multiple of context ({context})")
        if len(ids) <= context:
            raise ValueError("Need more than context tokens to sample a window")

        self.ids = torch.as_tensor(np.asarray(ids, dtype=np.int64))
        self.context = context
        self.batch_size = tokens_per_batch // context
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)

    def next_batch(self):
        """Return (inputs, targets), both of shape (batch_size, context)."""
        # A window of context + 1 ids can start anywhere up to len(ids) - context - 1.
        starts = torch.randint(len(self.ids) - self.context, (self.batch_size,), generator=self.generator)
        windows = self.ids[starts[:, None] + torch.arange(self.context + 1)]

        inputs, targets = windows[:, :-1], windows[:, 1:]
        return inputs.to(self.device), targets.to(self.device)
