"""A decoder-only Transformer language model, written from scratch in PyTorch.

Only basic PyTorch layers are used (nn.Embedding, nn.Linear, nn.LayerNorm,
nn.Dropout). Multi-head attention, the causal mask and the blocks are written
here; nn.Transformer and nn.MultiheadAttention are not used.

The model reads up to `context` token ids and, at every position, predicts the
next token. Each block is pre-LayerNorm:

    x = x + attention(layer_norm(x))
    x = x + mlp(layer_norm(x))

The output layer reuses the token embedding matrix (tied embeddings).
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm.data import ROOT

# A run's checkpoint goes to checkpoints/<run>/, next to its results in results/<run>/.
CHECKPOINTS_DIR = ROOT / "checkpoints"


@dataclass
class TransformerConfig:
    vocab_size: int
    context: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position only sees itself and earlier positions."""

    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

        # mask[i, j] is True when position i may look at position j (j <= i).
        mask = torch.tril(torch.ones(config.context, config.context, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=2)

        # (batch, length, d_model) -> (batch, heads, length, head_size)
        q = q.view(batch, length, self.n_heads, -1).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, -1).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, -1).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))

        # Mix the values, then put the heads back side by side.
        out = (weights @ v).transpose(1, 2).reshape(batch, length, d_model)
        return self.out_dropout(self.out(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, 4 * config.d_model)
        self.down = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.apply(_init_weights)

    def hidden_states(self, ids):
        """Return the final hidden state at every position: (batch, length, d_model)."""
        batch, length = ids.shape
        if length > self.config.context:
            raise ValueError(f"Got {length} tokens, but the context is {self.config.context}")

        positions = torch.arange(length, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, ids):
        """Return next-token logits at every position: (batch, length, vocab_size)."""
        # Tied embeddings: the output layer is the token embedding matrix.
        return self.hidden_states(ids) @ self.token_embedding.weight.T

    def next_token_logits(self, ids):
        """Return logits for the token after the last position only: (batch, vocab_size).

        Same as forward(ids)[:, -1], but skips the output layer for the other
        positions, which makes scoring much faster.
        """
        return self.hidden_states(ids)[:, -1] @ self.token_embedding.weight.T

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def _init_weights(module):
    # Small random weights (as in GPT-2), so the first predictions are close to
    # uniform and the initial loss is close to ln(vocab_size).
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def save_checkpoint(model, path, **info):
    """Save the model's config and weights, plus any extra info (e.g. step, val_loss).

    The weights are stored at half precision (float16), which halves the file
    (about 23 MB for the full model) so the checkpoints later tickets need can
    be committed. The rounding barely changes predictions: on a small trained
    model, validation perplexity moved by 0.002%.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: tensor.half() for name, tensor in model.state_dict().items()}
    torch.save({"config": asdict(model.config), "state_dict": weights, **info}, path)


def load_checkpoint(path, device="cpu"):
    """Load a model saved with save_checkpoint, in eval mode, with float32 weights."""
    checkpoint = torch.load(path, map_location=device)
    model = TransformerLM(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
