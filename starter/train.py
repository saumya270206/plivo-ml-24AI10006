"""Training loop.

Change from previous version: batch_size default bumped 8 -> 64.
Batch size is NOT capped by the assignment (only optimizer `steps` and
`n_params` are), so this is a "free" change: total tokens actually seen by
the model = steps * batch_size * block_size, and going 8->64 gives the
model ~8x more tokens for the exact same 2000 gradient updates, with
lower-variance gradients as a side benefit. If you're memory-constrained,
drop --batch_size and/or add gradient accumulation (not implemented here
to keep the script simple -- ping me if you want that added).

Everything else (AdamW with weight decay only on matrices, linear warmup +
cosine decay to 10% of peak LR, grad-norm clipping at 1.0) is unchanged
from before. model.py no longer has pos_emb (RoPE lives inside attention
instead), so nothing here needs to change to account for that -- get_batch
and the training step are agnostic to how the model encodes position.
"""
import argparse
import json
import math
import time

import torch

from model import Config, GPT


def get_batch(data, block_size, batch_size, device):
    """Sample `batch_size` random block_size-length windows, with
    replacement, from the token id tensor `data`."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def lr_at_step(step, total_steps, peak_lr, warmup_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine


def build_param_groups(model, weight_decay):
    """Weight decay only on 2D+ params (matrices). Biases, LayerNorm
    weights, and (now unused, but kept generic) embeddings with ndim < 2
    are excluded -- decaying those fights normalization for no benefit."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_tokenized(data_path, block_size, merges_path="bpe_merges.json"):
    """Load raw text, encode with the trained BPE tokenizer, return a 1D
    LongTensor of token ids."""
    from tokenizer import BPETokenizer  # local module from starter/

    with open(merges_path) as f:
        saved = json.load(f)
    tok = BPETokenizer(saved["merges"])

    text = open(data_path, encoding="utf-8").read()
    ids = tok.encode(text)
    assert len(ids) > block_size + 1, "corpus too short for one training window"
    return torch.tensor(ids, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--peak_lr", type=float, default=3e-3)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config()
    model = GPT(cfg).to(device)
    n_params = model.n_params()
    print(f"n_params = {n_params:,}")
    assert n_params <= 2_000_000, f"over param cap: {n_params:,} > 2,000,000"

    data = load_tokenized(args.data, cfg.block_size).to(device)

    optim = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay),
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at_step(step, args.steps, args.peak_lr, args.warmup_steps)
        for g in optim.param_groups:
            g["lr"] = lr

        x, y = get_batch(data, cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        if step % 100 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

    config_dict = {
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "dropout": cfg.dropout,
        "tie_weights": cfg.tie_weights,
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config_dict,
        "steps": args.steps,
        "n_params": n_params,
    }, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()