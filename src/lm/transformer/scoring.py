"""Score a split with the Transformer, using a sliding window.

A model can't predict a token from nothing, so <eos> is put in front of the
split as a start marker: every split starts at a line boundary, and <eos> is
the token that ends every line. With context k, target i of the split is then
predicted from the last min(i + 1, k) tokens before it (the marker included):

1. The first window, the first k tokens, predicts its k targets in one pass.
2. Every later target sees exactly the k tokens before it. These windows are
   scored in batches, using only the logits after the last position.

A target's rank counts the tokens with a higher log-prob, plus the tokens
with the same log-prob and a lower id. Ids are in train-frequency order, so
ties are broken by unigram frequency, as for the n-gram models.
"""

import numpy as np
import torch
import torch.nn.functional as F

from lm.metrics import save_predictions


@torch.no_grad()
def score_split(model, ids, eos_id, batch_size=256, path=None):
    """Score every token of a split and return (logprob, rank).

    If path is given, the predictions are also saved in the lm.metrics format.
    """
    model.eval()
    device = next(model.parameters()).device
    context = model.config.context

    stream = torch.as_tensor(np.concatenate([[eos_id], np.asarray(ids, dtype=np.int64)]))
    targets = stream[1:]
    num_targets = len(targets)
    logprob = np.empty(num_targets, dtype=np.float64)
    rank = np.empty(num_targets, dtype=np.int64)

    # 1. Target i < k is predicted from stream[0 : i + 1], all in the first window.
    first = min(context, num_targets)
    logits = model(stream[None, :first].to(device))[0]
    _store(logprob, rank, 0, logits, targets[:first])

    # 2. Target i >= k is predicted from the window stream[i + 1 - k : i + 1].
    if num_targets > first:
        windows = stream.unfold(0, context, 1)  # windows[j] = stream[j : j + k], without copying
        for start in range(first, num_targets, batch_size):
            end = min(start + batch_size, num_targets)
            logits = model.next_token_logits(windows[start + 1 - context : end + 1 - context].to(device))
            _store(logprob, rank, start, logits, targets[start:end])

    if path is not None:
        save_predictions(path, logprob, rank)

    return logprob, rank


def _store(logprob, rank, start, logits, targets):
    """Write the log-prob and rank of each target, given its logits, from index start on."""
    log_probs = F.log_softmax(logits.double(), dim=-1)
    targets = targets.to(log_probs.device)[:, None]
    target_log_probs = log_probs.gather(1, targets)

    # Tokens ahead of the target: higher log-prob, or the same log-prob and a lower id.
    token_ids = torch.arange(log_probs.size(1), device=log_probs.device)
    ahead = (log_probs > target_log_probs) | ((log_probs == target_log_probs) & (token_ids < targets))

    end = start + len(targets)
    logprob[start:end] = target_log_probs[:, 0].cpu().numpy()
    rank[start:end] = ahead.sum(dim=1).cpu().numpy()
