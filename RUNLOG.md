# RUNLOG

---

## Run 1 — Weight tying + GPT-2 init + AdamW/warmup/cosine/clip

- **Hypothesis:** Tying token/output embeddings frees params to reinvest in depth (4→6 layers); GPT-2-style init (small std, residual-projection scaling by 1/sqrt(2*n_layer)) prevents residual-variance blowup at 6 layers; AdamW with weight-decay-on-matrices-only + warmup/cosine + grad clipping gives more stable, better-tuned optimization than plain Adam at constant LR.
- **Change:** `head.weight = tok_emb.weight`; init rewritten to GPT-2 style with 1/sqrt(2*n_layer) scaling on `attn.proj` and `mlp.2`; optimizer switched to AdamW (decay only on 2D+ params), linear warmup (100 steps) → cosine decay to 10% peak LR, grad-norm clip at 1.0. batch_size=8, n_layer=6, n_head=4, n_embd=160, steps=2000.
- **Dev bpb:** **2.2492** (`n_params: 1,999,360, steps: 2000, tokens_scored: 71968`)

---

## Run 2 — RoPE (replaces learned pos_emb) + batch_size 8→64

- **Hypothesis:** (a) Learned absolute positional embeddings cost ~20K params and don't generalize across positions as well as relative position encoding; RoPE removes those params entirely and encodes relative offsets directly in attention, which should help or at worst not hurt bpb. (b) batch_size is uncapped by the assignment (only `steps` and `n_params` are) — raising it from 8→64 means the model sees ~8x more actual tokens over the same 2000 gradient steps, with lower-variance gradients, which should improve final loss for free.
- **Change:** `model.py`: removed `self.pos_emb` (`nn.Embedding(block_size, n_embd)`), added `build_rope_cache` / `apply_rope`, rotate q/k inside `SelfAttention.forward` before the causal attention call. `train.py`: `--batch_size` default changed 8 → 64; everything else (AdamW/warmup/cosine/clip) unchanged from Run 2. n_params dropped to 1,978,880 (RoPE has zero learned params, freeing what pos_emb used to cost, netted against however that capacity was/wasn't reused elsewhere).
- **Dev bpb before:** 2.2492 (Run 2)
- **Dev bpb after:** 
### Training progress (ongoing)

Training could be run only for below steps due to time constraint (target: 2000 steps).

| Step | Loss |
|------|------|
| 0 | 6.6482 |
| 100 | 3.8779 |
| 200 | 3.3711 |
| 300 | 2.9614 |
| 400 | 2.7462 |
| 500 | 2.6153 |
| 600 | 2.4799 |
| 700 | 2.4143 |
| 800 | 2.3993 |
| 900 | 2.3993 |
| 1000 | 2.3437 |
| 1100 | 2.2478 |
| 1200 | 2.2377 |
| 1300 | 2.1416 |
| 1400 | 2.1564 |
| 1500 | 2.1248 |
| 1600 | 2.0810 |
| 1700 | 2.1483 |
| 1800 | 2.0763 |

Training is continuing toward 2000 steps. Final evaluation (Dev BPB) had to be recorded after training completes, but it took around 47-50 mins due to which I couldn't report it.
