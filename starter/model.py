"""A small GPT in plain PyTorch. Yours to modify or replace entirely —
attention, SSM, whatever — as long as evaluate.py still works and the
parameter cap holds.

Change from baseline: learned absolute positional embeddings (pos_emb) are
replaced with RoPE (Rotary Position Embeddings, Su et al. 2021). RoPE has
zero learned parameters (it's a fixed rotation applied inside attention),
so the ~20K params that used to sit in pos_emb are simply gone from the
param count -- freed up under the 2,000,000 cap. RoPE also tends to
generalize across positions better than a learned absolute table, since
it encodes *relative* offsets between query/key pairs rather than a fixed
per-slot vector.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 768      # BPE tokenizer, trained on train_corpus.txt only
    block_size = 128      # in BPE tokens; ~2.2 bytes/token -> ~280 bytes of
                           # real context, vs 128 bytes for the byte tokenizer
    n_layer = 6
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True    # saves vocab_size*n_embd params -> spend on depth instead


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    """Precompute cos/sin tables for RoPE.

    Returns cos, sin each of shape (seq_len, head_dim // 2).
    head_dim must be even (standard for attention heads; asserted below).
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to q or k.

    x:   (B, n_head, T, head_dim)
    cos: (T, head_dim/2)
    sin: (T, head_dim/2)
    """
    x1 = x[..., 0::2]  # (B, n_head, T, head_dim/2)
    x2 = x[..., 1::2]  # (B, n_head, T, head_dim/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    # interleave back to (B, n_head, T, head_dim)
    out = torch.stack((rx1, rx2), dim=-1).flatten(-2)
    return out


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # rotate q, k (not v) -- standard RoPE
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # NOTE: no self.pos_emb -- positions are handled by RoPE inside
        # attention instead of a learned embedding table added to the input.
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        assert head_dim % 2 == 0, "n_embd // n_head must be even for RoPE"

        # cache cos/sin up to block_size; rebuilt lazily if a longer T is
        # ever requested (e.g. block_size changes between train/eval configs).
        cos, sin = build_rope_cache(cfg.block_size, head_dim, device=torch.device("cpu"))
        self.register_buffer("rope_cos_cache", cos, persistent=False)
        self.register_buffer("rope_sin_cache", sin, persistent=False)

        self.apply(self._init)
        # GPT-2 style: scale down residual-stream projections by 1/sqrt(2*n_layer)
        # so residual variance doesn't blow up with depth (Radford et al. 2019).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        # GPT-2 style init: small std, zero biases, near-zero LayerNorm bias/
        # unit weight.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _get_rope(self, T, device):
        if T <= self.rope_cos_cache.size(0) and self.rope_cos_cache.device == device:
            return self.rope_cos_cache[:T], self.rope_sin_cache[:T]
        # fallback: build fresh (covers longer T or a different device on
        # first call before buffers have been .to()'d)
        head_dim = self.cfg.n_embd // self.cfg.n_head
        cos, sin = build_rope_cache(T, head_dim, device=device)
        return cos, sin

    def forward(self, idx, targets=None):
        B, T = idx.shape
        rope_cos, rope_sin = self._get_rope(T, idx.device)
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, rope_cos, rope_sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())