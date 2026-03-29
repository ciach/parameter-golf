# Parameter Golf — Experiment Log & Next Steps

> **Goal:** Achieve ≤ 1.1194 BPB on FineWeb validation set  
> **Constraints:** 16MB artifact (16,000,000 bytes), 10 min training on 8xH100, tokenizer-agnostic BPB  
> **Current Best (valid):** 1.1278 BPB (`02_quality_push_mlp2p875_bg1536`, cap PASS)  
> **Current Best (ignoring cap):** 1.1233 BPB (frontier reproduction)

---

## Complete Results (All Rounds)

### Round 4 — 6L/768d Baseline (8xH100)

| Exp | Config | BPB | Size | Cap? |
|-----|--------|-----|------|------|
| `00_R4_8g_baseline` | 6L/768d, LeakyReLU, seq=4096 | **1.1529** | 36.0MB | ❌ FAIL (2.25x over) |

### Round 5 — LoRA TTT (Dead End)

| Exp | Config | BPB | Notes |
|-----|--------|-----|-------|
| `1_lora_ttt_rank16_lr0.01` | LoRA r=16 on baseline | 1.2358 | Worse than no TTT |
| `2_lora_ttt_rank16_lr0.05` | LoRA r=16 on baseline | 1.2364 | Worse than no TTT |

**Conclusion:** Naive LoRA TTT is a dead end.

### Round 7 — Frontier Reproduction + Deltas

| Exp | Config | BPB | Size | Cap? | Steps |
|-----|--------|-----|------|------|-------|
| `00_record_repeat` | 6L/768d baseline repeat | **1.1528** | 36.0MB | ❌ | 7143 |
| `01_frontier_repro` | 11L/512d frontier (default) | **1.1233** | 17.05MB | ❌ (-1.05MB) | 6990 |
| `02_frontier_leaky` | 11L/512d + LeakyReLU(0.5)² | **2.0388** | 17.12MB | ❌ | 6916 |
| `03_frontier_seq4096` | 11L/512d + seq=4096 | CRASHED | — | — | — |

### Round 8 — Size-Cap Experiments

| Exp | Config | BPB | Size | Cap? | Margin |
|-----|--------|-----|------|------|--------|
| `00_frontier_repeat_decimal` | 11L/512d (size check) | **1.1238** | 17,056,319 | ❌ FAIL | -1,056,319 |
| `01_frontier_mlp2.75` | 11L/512d, MLP mult 2.75 | **1.1279** | 16,266,009 | ❌ FAIL | -266,009 |
| `02_frontier_10_layers` | 10L/512d | CRASHED | — | — | DDP unused param |

### Round 9 — Cap-First Recovery (Successful)

| Exp | Config | BPB | Size | Cap? | Margin |
|-----|--------|-----|------|------|--------|
| `00_cap_anchor_mlp2p75_bg1024` | MLP 2.75, Bigram 1024, int6 embed, auto compressor | **1.1318** | 15,152,079 | ✅ PASS | +847,921 |
| `01_quality_step_mlp2p875_bg1024` | MLP 2.875, Bigram 1024, int6 embed, auto compressor | **1.1282** | 14,946,074 | ✅ PASS | +1,053,926 |
| `02_quality_push_mlp2p875_bg1536` | MLP 2.875, Bigram 1536, int6 embed, auto compressor | **1.1278** | 15,006,755 | ✅ PASS | +993,245 |

---

## Critical Findings

### 🟢 Finding 1: CAP BLOCKER RESOLVED

Round 9 produced three cap-valid runs under `16,000,000` bytes.

Key mechanism:
- `QUANT_COMPRESSOR=auto` selected `zstd` in all successful runs.
- `INT6_PACK=0` outperformed packed int6 after compression on this model family.

### 🔴 Finding 2: LeakyReLU² CATASTROPHICALLY FAILED

> [!CAUTION]
> `02_frontier_leaky_relu2` scored **2.0388 BPB** — a catastrophic regression from 1.1233.
> Pre-quant EMA was fine (1.1367), meaning the quantization **destroyed** the LeakyReLU model.
> The negative activations from LeakyReLU create weight distributions that the int6 quantizer cannot handle.

This is critical: LeakyReLU² is not free — it requires quantization-aware training (QAT) or a different quantization scheme.

### 🟡 Finding 3: MLP 2.75 Trades Size for BPB

Shrinking MLP mult from 3.0 → 2.75 saved **790KB** but cost **+0.0041 BPB** (1.1238 → 1.1279).

### 🔴 Finding 4: 10-Layer Crash Root Cause Is Known

The 10-layer crash is not random NCCL instability. The model was run with `NUM_LAYERS=10` but default `VE_LAYERS=9,10`.  
Layer `10` does not exist when layers are `0..9`, so one VE scale parameter is never used, and DDP aborts with "unused parameter".

This is fixable offline: clamp/filter `ve_layer_indices` to `< num_layers` (or run `VE_LAYERS=9` for 10-layer experiments).

### 🟢 Finding 5: Frontier Reproduction is Rock Solid

Both R7 and R8 reproduced the frontier at 1.1233-1.1238 BPB. The training loop is deterministic and reliable.

### 🟢 Finding 6: Best Valid Quality So Far Is 1.1278

`02_quality_push_mlp2p875_bg1536` currently dominates the valid frontier:
- `val_bpb: 1.12777470`
- `total bytes: 15,006,755`
- `cap margin: +993,245`

---

## The 16MB Budget Breakdown

The cap is `16,000,000 bytes = code + compressed_model`.

Reference points:
```
R8 default frontier: 17,056,319 (FAIL, -1,056,319)
R9 best valid run:   15,006,755 (PASS, +993,245)
```

Remaining optimization space is now about **~0.99MB headroom** on top of the best valid run.

Current size/quality levers:

| Strategy | Est. Bytes Saved | BPB Impact | Difficulty |
|----------|-----------------|-----------|------------|
| `QUANT_COMPRESSOR=auto` | **validated** | 0 | Done |
| `INT6_PACK=0` (vs packed) | **validated** | 0 | Done |
| Minify code (69KB→30KB) | ~40KB | 0 | Easy |
| MLP mult 3.0 → 2.75 | **790KB observed** | +0.0041 observed | Done |
| MLP mult 3.0 → 2.875 | ~395KB est | likely < +0.004 | Easy |
| Reduce KV heads 4→2 | ~5.46% params (~1.47M) | unknown | Medium |
| Reduce to 10 layers (with VE fix) | ~8.75% params (~2.36M) | unknown | Medium |
| Reduce BigramHash 2048→1024 | ~0.49% params (~131k) | low | Easy |
| Disable VE entirely | ~0.61% params (~164k) | low-medium | Easy |
| Lower `MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL` | small-medium | low | Easy |
| Mixed int5/int6 quantization | medium-large | unknown | Hard |

---

## Offline-First Prep (No Pod Needed)

> [!IMPORTANT]
> Goal: eliminate avoidable runtime failures and pre-rank architecture candidates before spending GPU minutes.

1. **Fix 10-layer VE indexing issue offline**  
   - Make VE layer parsing robust to `num_layers` changes (`VE_LAYERS` filtered to valid indices).
2. **Lock a static architecture shortlist from parameter deltas**  
   - `MLP_MULT=2.875` (moderate size cut, likely gentler BPB hit than 2.75)  
   - `NUM_KV_HEADS=2` (large size cut)  
   - `NUM_LAYERS=10` + `VE_LAYERS=9` (large size cut, now runnable)
3. **Prepare quantization-only sweeps (no retrain) if a checkpoint is available**  
   - Sweep `MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL` and compression backend (`zlib` vs `zstd`) on the same weights.
4. **Add a strict preflight for every run config**  
   - Validate constraints before launch (`num_heads % num_kv_heads == 0`, valid `VE_LAYERS`, cap set to `16_000_000`).
5. **Prepare run scripts and parsing ahead of time**  
   - Make result summaries include BPB + total bytes + cap PASS/FAIL to stop bad runs early.

---

## Next Steps (Round 9 Pod Plan)

> [!IMPORTANT]
> **Primary Goal:** Get the 11L/512d frontier model under 16,000,000 bytes while keeping BPB ≤ 1.125.

### Priority 1: Build on the Best Valid Run (1.1278)
1. Use `02_quality_push_mlp2p875_bg1536` as the new baseline.
2. Keep `QUANT_COMPRESSOR=auto` and `INT6_PACK=0` fixed.

### Priority 2: Architecture Deltas With High Size Leverage
3. Increase `BIGRAM_VOCAB_SIZE` gradually (`1536 -> 1792 -> 2048`) while monitoring cap margin.
4. Test `MLP_MULT` micro-steps above `2.875` only if margin stays healthy.
5. Revisit `NUM_LAYERS=10` with corrected `VE_LAYERS`, only if it can beat 1.1278 at similar margin.

### Priority 3: Smaller Additive Trims (if still over cap)
6. `BIGRAM_VOCAB_SIZE=1024`
7. `VE_ENABLED=0` (or smaller `VE_DIM`)
8. lower `MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL`

### LeakyReLU² Recovery Plan
9. **Do NOT use LeakyReLU without QAT** — the quantization destroys the signal
10. If we want LeakyReLU, we need to add late-stage quantization-aware fine-tuning

### 10-Layer Fix
11. Keep `VE_LAYERS` consistent with `NUM_LAYERS` in every run manifest

---

## Resolved Questions

> [!IMPORTANT]
> 1. **zstd support exists in code** (`_COMPRESSOR` switch), but runtime falls back to zlib unless `zstandard` is installed.
> 2. **Architecture trims beat code minification on impact** (40KB code savings is too small by itself).
> 3. **Offline-first is the right move**: patch config safety + pre-rank variants, then spend pods only on top candidates.

---

## Offline Change Log

### 2026-03-29 — Int6 Bit-Packing Added

- Added true int6 packing path in `train_gpt_frontier.py` (4 quantized values packed into 3 bytes).
- Added env toggle `INT6_PACK` (default `1`) to keep compatibility with old storage (`INT6_PACK=0`).
- Quantization metadata now records whether an int6 tensor is packed, and dequantization supports both formats.
- Run logs now print `int6_pack` in the configuration line so we can audit which artifact format produced each result.
- Added configurable int6 category set via `MIXED_QUANT_INT6_CATS` (default `mlp,attn`) so we can include `embed`.

Expected impact:
- Lower exported artifact size at the same architecture/hyperparameters.
- No intended BPB change versus previous int6 path (representation changed, quantized values unchanged).

### 2026-03-29 — Cap-First Round 9 Batch Prepared

Prepared `run_experiments_r9.sh` with a strict size-first baseline:
- `INT6_PACK=1`
- `MIXED_QUANT_INT6_CATS=mlp,attn,embed`
- `MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=16384`
- `SUBMISSION_SIZE_CAP_BYTES=16000000`

Run order:
1. `00_cap_anchor_mlp2p75_bg1024`
2. `01_quality_step_mlp2p875_bg1024`
3. `02_quality_push_mlp2p875_bg1536`

Goal:
- Keep every run under cap first.
- Recover BPB from the cap-safe anchor by gradually increasing capacity.

### 2026-03-29 — Compression Selection Fix

- Added `QUANT_COMPRESSOR` in `train_gpt_frontier.py` (`auto|zlib|zstd`).
- `auto` now compresses with both codecs and selects the smaller artifact.
- Updated `run_experiments_r9.sh` baseline to:
  - `QUANT_COMPRESSOR=auto`
  - `INT6_PACK=0` (packed int6 increased compressed size on real runs)

### 2026-03-29 — Round 9 Completed

- `00_cap_anchor_mlp2p75_bg1024`: `1.13182362`, `15,152,079`, PASS
- `01_quality_step_mlp2p875_bg1024`: `1.12824321`, `14,946,074`, PASS
- `02_quality_push_mlp2p875_bg1536`: `1.12777470`, `15,006,755`, PASS (**current best valid**)

### 2026-03-29 — Round 9 Rerun Notes (Why It Started Failing, Then Passed)

- First R9 attempt used `INT6_PACK=1` and all runs failed cap/parse checks in early logs; one observed anchor run produced `17,108,867` bytes (cap FAIL).
- Final R9 batch switched to `QUANT_COMPRESSOR=auto` + `INT6_PACK=0`, and all three runs passed cap.
- In successful runs, `quant_codec_selected` resolved to `zstd`; this combination gave both better compression and stable cap margin under `16,000,000`.

### 2026-03-29 — Round 10 Prepared (5-Run Capacity Ladder, Quant-Quality First)

Base env kept fixed for cap safety:
- `SUBMISSION_SIZE_CAP_BYTES=16000000`
- `QUANT_COMPRESSOR=auto`
- `INT6_PACK=0`
- `EVAL_STRIDE=64`
- `MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=16384`

Execution order (requested: run former "5th idea" first):
1. `00_quant_quality_tweak_mlp3_bg2048_no_embed_int6`  
   `MIXED_QUANT_INT6_CATS=mlp,attn MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048`
2. `01_repeat_best_mlp2p875_bg1536`  
   `MIXED_QUANT_INT6_CATS=mlp,attn,embed MLP_MULT=2.875 BIGRAM_VOCAB_SIZE=1536`
3. `02_capacity_up_mlp3_bg1536`  
   `MIXED_QUANT_INT6_CATS=mlp,attn,embed MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=1536`
4. `03_capacity_up_plus_mlp3_bg2048`  
   `MIXED_QUANT_INT6_CATS=mlp,attn,embed MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048`
5. `04_no_size_arch_tweak_mlp3_bg2048_ve_8_9_10`  
   `MIXED_QUANT_INT6_CATS=mlp,attn,embed MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048 VE_LAYERS=8,9,10`

Supporting code hardening:
- `train_gpt_frontier.py` now clamps invalid `VE_LAYERS` indices to `[0, num_layers-1]` and logs dropped layers, preventing the known DDP unused-parameter crash when `NUM_LAYERS` changes.

### 2026-03-29 — Round 11 Prepared (Phase 1 Squeeze Around Winner)

Base winner locked:
- `MIXED_QUANT_INT6_CATS=mlp,attn`
- `MLP_MULT=3.0`
- `BIGRAM_VOCAB_SIZE=2048`
- `QUANT_COMPRESSOR=auto`
- `INT6_PACK=0`

R11 run order:
1. `00_seq4096_ve_expand_late_qat` (`TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`, `TRAIN_BATCH_TOKENS=393216`, `VE_LAYERS=8,9,10`, `LATE_QAT_START_FRAC=0.90`)
2. `01_ve_expand_8_9_10`
3. `02_late_qat_last10pct` (`LATE_QAT_THRESHOLD=0`, `LATE_QAT_START_FRAC=0.90`)
4. `03_ve_expand_plus_late_qat`
5. `04_seq4096_safe_batch` (`TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`, `TRAIN_BATCH_TOKENS=393216`)

Training code updates for this phase:
- Added explicit late-QAT wallclock control via `LATE_QAT_START_FRAC` in `train_gpt_frontier.py`.
- Late-QAT now enables when either scale threshold or start fraction trigger fires, and logs trigger reason.

### 2026-03-29 — Systems Track To Try (Memory + Throughput)

Goal:
- Recover 4096-context viability without halving effective optimization throughput.
- Reduce wallclock overhead so eval-time/quantization experiments fit inside the 10-minute budget.

Planned experiments:
1. **Gradient checkpointing (activation checkpointing) on transformer blocks**
   - Approach: wrap block forwards with `torch.utils.checkpoint.checkpoint(...)`.
   - Expected effect: significantly lower activation VRAM at long sequence length; allows `TRAIN_SEQ_LEN=4096` with less aggressive `TRAIN_BATCH_TOKENS` reduction.
   - Tradeoff: extra compute from recomputation; must verify net wallclock still acceptable.
2. **Asynchronous batch prefetch for token loader**
   - Approach: prefetch next batch in a background worker/thread so host-side shard reads are overlapped with GPU compute.
   - Expected effect: lower per-step idle time and better utilization; improves effective training steps under the same 600s cap.
   - Tradeoff: slightly more loader complexity; must verify deterministic behavior and stable DDP synchronization.

Validation checklist before keeping these changes:
- Compare `step_avg` and final step count at 600s vs current baseline.
- Confirm no DDP unused-parameter or synchronization regressions.
- Confirm cap metrics and final `val_bpb` are unaffected by loader-only changes.

### 2026-03-29 — Architectural Track To Try (BPB per Byte)

Context:
- Current best valid leaves roughly `~438KB` cap margin.
- Priority is to increase modeling power per stored byte, not raw parameter count.

Planned experiments:
1. **Cross-layer parameter sharing (depth recurrence)**
   - Approach: route activations through selected mid/deep blocks multiple times in forward pass while keeping a shared weight set.
   - Expected effect: deeper effective computation with little/no added artifact bytes.
   - Tradeoff: added compute per token can reduce total steps before wallclock cap; must measure step count and final BPB jointly.
2. **SwiGLU/GeGLU MLP replacement**
   - Approach: replace `relu^2` MLP with gated MLP (`SwiGLU` or `GeGLU`) and retune `MLP_MULT` downward to stay under cap.
   - Expected effect: better loss/BPB per parameter on modern decoder stacks.
   - Tradeoff: introduces extra projection path; requires cap-aware width rebalancing and may need fresh LR tuning.
3. **Hybrid local + global attention**
   - Approach: apply sliding-window/local attention in earlier layers (e.g. first 6), keep full causal attention in final layers.
   - Expected effect: lower memory/compute pressure at longer sequence lengths while preserving global integration near the output.
   - Tradeoff: implementation complexity and risk of quality loss if local window is too small; must sweep window size with fixed budget.

Validation checklist before keeping these changes:
- Confirm artifact stays below `16,000,000` bytes with margin.
- Compare BPB gain against wallclock cost (step throughput and final evaluated BPB).
- Run at least one repeat on any promising variant to rule out run-to-run variance.

### 2026-03-29 — Code & Compute Efficiency Track

Goal:
- Improve effective throughput and free byte budget without changing core model behavior.

Planned improvements:
1. **DDP overlap/sync handling cleanup**
   - Approach: replace manual `model.require_backward_grad_sync` toggling with `model.no_sync()` for gradient-accumulation microsteps.
   - Expected effect: safer synchronization semantics and potentially better DDP overlap behavior under accumulation.
   - Tradeoff: requires careful parity testing to ensure identical effective gradients and no hidden sync regressions.
2. **Script-size minimization for artifact budget**
   - Approach: remove dead code/imports/comments and produce a submission-minified trainer snapshot.
   - Expected effect: recover roughly `10KB-20KB` of code budget that can be reallocated to small quality-positive knobs.
   - Tradeoff: readability/maintainability of the minified snapshot; keep the readable source as canonical and only minify submission copy.

Validation checklist before keeping these changes:
- Verify identical final metrics on a short A/B smoke run for DDP sync refactor.
- Recompute `bytes_code`, `bytes_total`, and cap margin after minified snapshot export.
- Confirm no runtime failures from minified script in the isolated records-folder execution path.
