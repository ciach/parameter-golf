# Parameter Golf — Experiment Log

> **Goal:** Achieve ≤ 1.1194 BPB on FineWeb validation set  
> **Constraints:** 16MB artifact, 10 min training on 8xH100, tokenizer-agnostic BPB  
> **Current Best:** 1.1529 BPB (Round 4 baseline, 6L/768d)

---

## Target Numbers (Honest)

| Metric | BPB | Source |
|--------|-----|--------|
| Best official non-TTT (seed 1337) | **1.1228** | PR #401, to reproduce |
| Mean over 3 seeds of that stack | **1.1233** | PR #401 README |
| Official world record (with TTT) | **1.1194** | PR #549 (LeakyReLU² + Legal TTT + Parallel Muon) |
| Our current best | **1.1529** | Round 4 baseline |
| **Gap to non-TTT frontier** | **0.0301** | |
| **Gap to world record** | **0.0335** | |

---

## Completed Experiments

### Round 4 — Multi-GPU Scaling (8xH100)

| # | Experiment ID | Config | BPB | Δ vs Best |
|---|--------------|--------|-----|-----------|
| 1 | `00_R4_8g_baseline` | 6L/768d, batch=524288, lr=0.02 | **1.1529** | — |
| 2 | `1_one_million_batch` | batch=1048576 | 1.1556 | +0.0027 |
| 3 | `2_hyper_frequency` | batch=262144 | 1.1559 | +0.0030 |
| 4 | `3_high_velocity_lrs` | lr=0.04 | 1.1586 | +0.0057 |

**Conclusion:** Hyperparams saturated. lr=0.02, batch=524288 optimal for 6L/768d Muon.

### Round 5 — LoRA TTT (8xH100 Spot)

| # | Experiment ID | Config | BPB | Δ vs Best |
|---|--------------|--------|-----|-----------|
| 1 | `1_lora_ttt_rank16_lr_0.01` | LoRA rank=16, lr=0.01 | 1.2358 | +0.0829 |
| 2 | `2_lora_ttt_rank16_lr_0.05` | LoRA rank=16, lr=0.05 | 1.2364 | +0.0835 |
| 3 | `3_lora_ttt_rank32_lr_0.01` | LoRA rank=32, lr=0.01 | ~1.275* | +0.12* |
| 4 | `4_lora_ttt_rank32_lr_0.05` | LoRA rank=32, lr=0.05 | — | — |

*Exp 3 at step 5000/20000 when spot pod reclaimed. Exp 4 never ran.

**Conclusion:** Naive LoRA TTT on our 6L/768d base is a dead end — it steals wall-clock from Muon and produces worse results. TTT remains a **late-stage optimization** to revisit only after matching the 1.1228 base with a proper architecture.

---

## Round 6 — Frontier Reproduction + Delta Experiments

### Run Order (strict sequence, one change at a time)

| # | Run ID | Change from Previous | Purpose | Expected BPB |
|---|--------|---------------------|---------|--------------|
| 1 | `frontier_reproduction` | Replace entire stack with PR #401 (1.1233) | Verify hardware reproduces within ±0.003 | ~1.123 |
| 2 | `leaky_relu2_delta` | Swap relu² → LeakyReLU(0.5)² | Test if our proven activation transfers | < 1.123? |
| 3 | `seq4096_delta` | seq_len 2048 → 4096 | Test longer context on frontier base | unknown |
| 4 | `bigram_4096_delta` | BigramHash 2048 → 4096 buckets | Scale local context cheaply | unknown |
| 5 | `bigram_8192_delta` | BigramHash 2048 → 8192 buckets | Further bucket scaling | unknown |
| 6+ | `cheap_ttt_*` | Rank-4, V-only, doc-isolated TTT | Only if base is at ~1.123 | ~1.119? |

> [!IMPORTANT]
> **Rule:** Each run changes exactly ONE variable from the previous best. No god-mode combos until individual effects are measured.

---

## The Frontier Stack (PR #401, 1.1228/1.1233)

### Architecture
- 11 layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP (1536 hidden), relu² activation
- U-Net skip connections (5 encoder, 6 decoder)
- XSA on last 4 layers (orthogonal projection, zero params)
- Partial RoPE (16/64 dims)
- LN Scale 1/sqrt(layer_idx+1)
- SmearGate (learned token blend, ~512 params)
- BigramHash (2048 buckets, dim=128, projected to 512)
- Shared Value Embedding (dim=128, layers 9,10)
- Tied embeddings, logit softcap=30.0

### Training
- Muon lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- AdamW embeddings lr=0.035, scalars lr=0.025, WD=0.04
- Grad clip: 0.3, Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iters (wallclock-based)
- EMA: decay=0.997, every step
- Tight SWA: every 50 steps when scale<0.2
- OrthoInit + muP-scaled output projections

### Post-Training & Eval
- GPTQ-lite: 5 clip percentiles per row, pick min MSE (zero training cost)
- Int6 per-row MLP+attn, Int8 embeddings
- zstd level 22 compression
- **Sliding window eval stride=64** (~0.034 BPB improvement over chunked)

> [!WARNING]
> Eval correctness is a first-class concern. Sliding window stride=64 is NOT optional — it's worth ~0.034 BPB. Must confirm: doc isolation, stride=64, quantized artifact eval, scoring path all match frontier exactly.

---

## Cost Tracking

| Date | Pod Type | Duration | Cost | Experiments |
|------|----------|----------|------|-------------|
| Mar 28 | 8xH100 on-demand | ~1.5hr | ~$30 | R4 (4 runs) |
| Mar 28 | 8xH100 on-demand | ~0.5hr | ~$10 | R5 partial (2 runs) |
| Mar 29 | 8xH100 spot | ~0.5hr | ~$7 | R5 continued (interrupted) |
| **Total spent** | | | **~$47** | **6 completed experiments** |
| **Remaining** | | | **~$22** | ~1.5 hrs spot @ $14/hr |

---

## Key Learnings

1. **The gap is 0.03 BPB = ~11 compounding tricks.** No silver bullets exist.
2. **Architecture is wrong.** 6L/768d cannot match 11L/512d. Must switch.
3. **Hyperparams are saturated** on the 6L architecture.
4. **Naive LoRA TTT is a dead end** on a weak base. Revisit only after matching frontier.
5. **Eval correctness matters enormously.** Sliding window stride=64 is ~0.034 free BPB.
6. **LeakyReLU² is the cleanest first innovation** — aligns with the world record stack.
7. **BigramHash bucket scaling** (2048→4096→8192) is a proven incremental lever.
8. **One variable at a time.** No god-mode combos until effects are isolated.

---

## Submission & PR Preparation

> [!IMPORTANT]
> **To submit 1.1529 (00_R4_8g_baseline) to the leaderboard:**
> You must re-run this config once to get the `train.log`, then use these commands:

1. **Create the folder:**
   `mkdir -p records/track_10min_16mb/$(date +%Y-%m-%d)_GPT_6L_768d_1.1529`
2. **Setup submission.json:** 
   (Copy from the snippet in the chat or use the template in `records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json`)
3. **Capture the files:**
   ```bash
   cp train_gpt.py records/track_10min_16mb/YYYY-MM-DD_.../
   cp experiment_logs_r4/log_00_R4_8g_baseline.txt records/track_10min_16mb/YYYY-MM-DD_.../train.log
   ```

---

## Immediate Next Steps (The Research Path)

- [ ] **1. Spin up new 8xH100 Spot Pod** (Keep note of the IP/Port).
- [ ] **2. Frontier Reproduction (Round 6, Exp 1)**: Run `train_gpt_frontier.py` to hit ~1.123 and verify hardware parity.
- [ ] **3. Delta #1 (LeakyReLU²)**: If reproduction is successful, immediately run with the LeakyReLU improvement.
- [ ] **4. PR the Winner**: If Round 6 hits ~1.123, skip the 1.1529 PR and submit the record-breaking stack instead.
