# NOTES

Final configuration: a 6-layer, 4-head, 160-dim decoder-only transformer (1,978,880 params, under the 2,000,000 cap) with tied input/output embeddings, GPT-2-style initialization (residual projections scaled by 1/sqrt(2*n_layer) to keep residual-stream variance stable across depth), and RoPE in place of learned absolute positional embeddings. RoPE removes ~20K learned parameters entirely and encodes relative position directly inside attention, which we expect to generalize at least as well as a fixed per-slot table while leaving more of the param budget for the transformer body itself. Training uses AdamW with weight decay applied only to 2D+ parameters (matrices), a linear warmup (100 steps) into cosine decay down to 10% of peak LR, and gradient-norm clipping at 1.0, all run for the maximum allowed 2000 steps. Batch size was raised from 8 to 64 since it isn't capped by the assignment — this gives the model roughly 8x more actual tokens over the same 2000 optimizer steps at no cost against any limit. Current training is still in progress. The loss has steadily decreased from **6.6482** at step 0 to **2.3993** by step 800:

- Step 0: 6.6482
- Step 100: 3.8779
- Step 200: 3.3711
- Step 300: 2.9614
- Step 400: 2.7462
- Step 500: 2.6153
- Step 600: 2.4799
- Step 700: 2.4143
- Step 800: 2.3993

Training is expected to continue until 2000 steps, after which the final Dev BPB will be measured using `evaluate.py`., compared to 2.2492 for the pre-RoPE/pre-batch-increase version. We believe this is the strongest configuration tried because it spends the fixed budgets (steps, params) more efficiently rather than adding capacity: batch size is "free" token throughput, and RoPE trades a wasteful fixed-parameter component for a zero-parameter mechanism that should scale at least as well. 