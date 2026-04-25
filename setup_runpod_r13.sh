#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh RunPod setup/validation for Parameter Golf R13 experiments.
# Run from the patched repo on the pod, or set REPO_DIR=/workspace/parameter-golf.

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
REPO_URL="${REPO_URL:-https://github.com/ciach/parameter-golf.git}"
DATA_VARIANT="${DATA_VARIANT:-sp1024}"
EXTRA_DATA_VARIANTS="${EXTRA_DATA_VARIANTS:-sp2048}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MIN_WORKSPACE_GB="${MIN_WORKSPACE_GB:-30}"
RUN_DATA_DOWNLOAD="${RUN_DATA_DOWNLOAD:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"
SMOKE_WALLCLOCK_SECONDS="${SMOKE_WALLCLOCK_SECONDS:-90}"
SMOKE_MAX_STEP_MS="${SMOKE_MAX_STEP_MS:-100}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HOME

log() { printf '[setup] %s\n' "$*"; }
fail() { printf '[setup][ERROR] %s\n' "$*" >&2; exit 1; }

run_py() {
    python3 - "$@"
}

pip_install() {
    if python3 -m pip install --disable-pip-version-check --break-system-packages "$@"; then
        return 0
    fi
    python3 -m pip install --disable-pip-version-check "$@"
}

find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -x "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

ensure_repo() {
    mkdir -p "$(dirname "${REPO_DIR}")"
    if [ -f "${REPO_DIR}/train_gpt_frontier.py" ]; then
        log "repo found: ${REPO_DIR}"
        return
    fi
    if [ -e "${REPO_DIR}" ] && [ -n "$(find "${REPO_DIR}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1 || true)" ]; then
        local backup="${REPO_DIR}.skeleton.$(date +%s)"
        log "moving non-repo/non-patched directory to ${backup}"
        mv "${REPO_DIR}" "${backup}"
    fi
    log "cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
}

validate_system() {
    log "system preflight"
    command -v python3 >/dev/null 2>&1 || fail "python3 not found"
    command -v git >/dev/null 2>&1 || fail "git not found"
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
    command -v torchrun >/dev/null 2>&1 || fail "torchrun not found"

    python3 --version
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    nvidia-smi --query-gpu=name --format=csv,noheader
    [ "${gpu_count}" = "8" ] || fail "expected 8 GPUs, got ${gpu_count}"

    local avail_gb
    avail_gb="$(df -BG /workspace 2>/dev/null | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
    [ -n "${avail_gb}" ] || fail "could not read /workspace free disk"
    log "/workspace free: ${avail_gb}G"
    [ "${avail_gb}" -ge "${MIN_WORKSPACE_GB}" ] || fail "need at least ${MIN_WORKSPACE_GB}G free on /workspace"
}

install_python_deps() {
    log "checking Python deps"
    if python3 - <<'PY'
mods = ["torch", "numpy", "sentencepiece", "huggingface_hub", "zstandard"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("missing", " ".join(missing))
    raise SystemExit(1)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
PY
    then
        return
    fi
    log "installing missing lightweight deps"
    pip_install --upgrade numpy sentencepiece huggingface-hub zstandard packaging wheel setuptools ninja einops
}

ensure_flash_attention() {
    log "checking FlashAttention"
    if python3 - <<'PY'
import importlib
for name in ("flash_attn_interface", "flash_attn.flash_attn_interface"):
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__file__", ""))
        raise SystemExit(0)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:160])
raise SystemExit(1)
PY
    then
        return
    fi

    local nvcc_path
    nvcc_path="$(find_nvcc || true)"
    [ -n "${nvcc_path}" ] || fail "FlashAttention missing and nvcc not found; use a CUDA devel/H100 image"
    export CUDA_HOME
    CUDA_HOME="$(dirname "$(dirname "${nvcc_path}")")"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    log "building flash-attn with CUDA_HOME=${CUDA_HOME}"
    nvcc --version | tail -n 1 || true
    pip_install --upgrade packaging wheel setuptools ninja einops
    MAX_JOBS="${MAX_JOBS:-8}" pip_install flash-attn==2.8.3 --no-build-isolation

    python3 - <<'PY'
import importlib
ok = False
for name in ("flash_attn_interface", "flash_attn.flash_attn_interface"):
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__file__", ""))
        ok = True
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:160])
raise SystemExit(0 if ok else 1)
PY
}

validate_repo_files() {
    cd "${REPO_DIR}"
    log "validating patched repo files in ${REPO_DIR}"
    [ -f train_gpt_frontier.py ] || fail "missing train_gpt_frontier.py"
    [ -f train_gpt_record1060.py ] || fail "missing train_gpt_record1060.py"
    [ -f data/cached_challenge_fineweb.py ] || fail "missing data downloader"
    [ -f tools/minify_train_script.py ] || fail "missing tools/minify_train_script.py; copy patched repo files first"
    [ -f tools/autoresearch_loop.py ] || fail "missing tools/autoresearch_loop.py; copy patched repo files first"
    [ -f tools/parse_experiment_log.py ] || fail "missing tools/parse_experiment_log.py; copy patched repo files first"
    [ -f experiment_queue.json ] || fail "missing experiment_queue.json; copy patched repo files first"
    [ -f research_state.json ] || fail "missing research_state.json; copy patched repo files first"
    [ -f run_experiments_r13_fast_tree.sh ] || fail "missing run_experiments_r13_fast_tree.sh; copy patched repo files first"

    grep -q '_FLASH_ATTN_BACKEND' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks FlashAttention backend patch"
    grep -q 'torch.is_grad_enabled()' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks RoPE grad-cache patch"
    grep -q 'fast_windows = my_windows' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks TTT fast-window fallback patch"
    grep -q 'MIXED_QUANT_INT8_PATTERNS' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks targeted mixed-quant override patch"
    grep -q 'QAT_BITS' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks targeted QAT bits patch"
    grep -q 'EMA_DECAY' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks EMA_DECAY patch"
    grep -q 'MATRIX_LR_LAYER_MULTS' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks MATRIX_LR_LAYER_MULTS patch"
    grep -q 'SPECTRAL_MODE' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks spectral-mode patch"
    grep -q 'target_recency_unigram' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks sparse target-cache patch"
    grep -q 'FACTORIZED_EMB_DIM' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks factorized embedding patch"
    grep -q 'OUTLIER_AUDIT' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks outlier audit patch"
    grep -q 'MATFORMER_WIDTHS' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks MatFormer width patch"
    grep -q 'MATFORMER_START_FRAC' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks MatFormer late-start patch"
    grep -q 'MATFORMER_LAYERS' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks MatFormer layer-subset patch"
    grep -q 'MLP_EQ_MODE' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks MLP equalization patch"
    grep -q 'act_rms' train_gpt_frontier.py || fail "train_gpt_frontier.py lacks activation-aware MLP equalization"
    grep -q 'RUN_TIMEOUT_SECONDS' run_experiments_r13_fast_tree.sh || fail "R13 runner lacks timeout guard"

    python3 -m py_compile train_gpt_frontier.py train_gpt_record1060.py tools/autoresearch_loop.py tools/parse_experiment_log.py
    python3 -m json.tool experiment_queue.json >/tmp/experiment_queue.ok
    python3 -m json.tool research_state.json >/tmp/research_state.ok
    python3 tools/autoresearch_loop.py validate
    python3 tools/minify_train_script.py --input train_gpt_frontier.py --output train_gpt_frontier_min.py
    python3 -m py_compile train_gpt_frontier_min.py
    chmod +x run_experiments_r13_fast_tree.sh

    python3 - <<'PY'
import train_gpt_frontier as t
print("train_gpt_frontier backend", t._FLASH_ATTN_BACKEND)
if t._FLASH_ATTN_BACKEND == "sdpa":
    raise SystemExit("sdpa fallback selected; FlashAttention install is not ready")
PY
}

download_data() {
    cd "${REPO_DIR}"
    if [ "${RUN_DATA_DOWNLOAD}" != "1" ]; then
        log "skipping data download because RUN_DATA_DOWNLOAD=${RUN_DATA_DOWNLOAD}"
        return
    fi
    log "downloading data variant=${DATA_VARIANT} train_shards=${TRAIN_SHARDS} HF_HOME=${HF_HOME}"
    python3 data/cached_challenge_fineweb.py --variant "${DATA_VARIANT}" --train-shards "${TRAIN_SHARDS}"
    local extra variant
    IFS=',' read -ra extra <<< "${EXTRA_DATA_VARIANTS}"
    for variant in "${extra[@]}"; do
        variant="$(echo "${variant}" | xargs)"
        [ -n "${variant}" ] || continue
        [ "${variant}" != "${DATA_VARIANT}" ] || continue
        log "downloading optional extra data variant=${variant} train_shards=${TRAIN_SHARDS}"
        if ! python3 data/cached_challenge_fineweb.py --variant "${variant}" --train-shards "${TRAIN_SHARDS}"; then
            log "optional extra data variant=${variant} unavailable; continuing without it"
        fi
    done
}

validate_one_data_variant() {
    cd "${REPO_DIR}"
    local variant="$1"
    local data_dir="data/datasets/fineweb10B_${variant}"
    local tok_prefix="data/tokenizers/fineweb_${variant#sp}_bpe"
    [ -d "${data_dir}" ] || fail "missing dataset dir ${data_dir}"
    local file_count train_count val_count tok_count
    file_count="$(find "${data_dir}" -maxdepth 1 -type f -name 'fineweb_*.bin' | wc -l | tr -d ' ')"
    train_count="$(find "${data_dir}" -maxdepth 1 -type f -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
    val_count="$(find "${data_dir}" -maxdepth 1 -type f -name 'fineweb_val_*.bin' | wc -l | tr -d ' ')"
    tok_count="$(find data/tokenizers -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
    log "data variant=${variant} files=${file_count} train=${train_count} val=${val_count} tokenizer_files=${tok_count}"
    [ "${train_count}" -ge "${TRAIN_SHARDS}" ] || fail "expected at least ${TRAIN_SHARDS} train shards, got ${train_count}"
    [ "${val_count}" -ge 1 ] || fail "expected at least 1 val shard"
    if [[ "${variant}" == sp* ]]; then
        [ -f "${tok_prefix}.model" ] || fail "missing tokenizer model ${tok_prefix}.model"
        [ -f "${tok_prefix}.vocab" ] || fail "missing tokenizer vocab ${tok_prefix}.vocab"
    fi
}

validate_data() {
    validate_one_data_variant "${DATA_VARIANT}"
    local extra variant
    IFS=',' read -ra extra <<< "${EXTRA_DATA_VARIANTS}"
    for variant in "${extra[@]}"; do
        variant="$(echo "${variant}" | xargs)"
        [ -n "${variant}" ] || continue
        [ "${variant}" != "${DATA_VARIANT}" ] || continue
        if [ -d "data/datasets/fineweb10B_${variant}" ]; then
            validate_one_data_variant "${variant}"
        else
            log "optional extra data variant=${variant} not present; skipping validation"
        fi
    done
    du -sh data "${HF_HOME}" 2>/dev/null || true
    df -h /workspace
}

run_smoke() {
    cd "${REPO_DIR}"
    if [ "${RUN_SMOKE}" != "1" ]; then
        log "skipping smoke because RUN_SMOKE=${RUN_SMOKE}"
        return
    fi
    log "running bounded FA speed smoke; timeout=${SMOKE_TIMEOUT_SECONDS}s wallclock=${SMOKE_WALLCLOCK_SECONDS}s"
    mkdir -p experiment_logs_r13/setup_smoke
    local log_file="experiment_logs_r13/setup_smoke/log_smoke.txt"
    set +e
    timeout "${SMOKE_TIMEOUT_SECONDS}s" env \
        RUN_ID=setup_smoke \
        MAX_WALLCLOCK_SECONDS="${SMOKE_WALLCLOCK_SECONDS}" \
        VAL_LOSS_EVERY=0 \
        TRAIN_LOG_EVERY=100 \
        WARMUP_STEPS=0 \
        TTT_ENABLED=0 \
        SUBMISSION_SIZE_CAP_BYTES=16000000 \
        EVAL_STRIDE=64 \
        QUANT_COMPRESSOR=auto \
        INT6_PACK=0 \
        MIXED_QUANT_INT6_CATS=mlp,attn \
        MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=24576 \
        MLP_MULT=3.0 \
        BIGRAM_VOCAB_SIZE=2048 \
        LATE_QAT_THRESHOLD=0 \
        LATE_QAT_START_FRAC=0.90 \
        torchrun --standalone --nproc_per_node=8 train_gpt_frontier_min.py > "${log_file}" 2>&1
    local status=$?
    set -e
    if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ]; then
        tail -n 120 "${log_file}" || true
        fail "smoke failed with exit ${status}"
    fi
    python3 - "${log_file}" "${SMOKE_MAX_STEP_MS}" <<'PY'
import re, sys
from pathlib import Path
log_path = Path(sys.argv[1])
max_ms = float(sys.argv[2])
text = log_path.read_text(errors="replace")
if "flash_attention_backend:sdpa" in text:
    raise SystemExit("smoke selected sdpa backend")
if "flash_attention_backend:" not in text:
    raise SystemExit("smoke missing flash_attention_backend log")
steps = []
for m in re.finditer(r"step:(\d+)/\d+ train_loss:[^\n]*train_time:(\d+)ms", text):
    steps.append((int(m.group(1)), int(m.group(2))))
print("smoke_steps", steps[-8:])
by_step = dict(steps)
if 10 in by_step and 100 in by_step:
    steady = (by_step[100] - by_step[10]) / 90.0
else:
    if not steps:
        raise SystemExit("no smoke step logs found")
    deltas = [
        (ms1 - ms0) / (step1 - step0)
        for (step0, ms0), (step1, ms1) in zip(steps, steps[1:])
        if step0 >= 3 and step1 > step0 and ms1 > ms0
    ]
    if not deltas:
        step, ms = steps[-1]
        steady = ms / max(step, 1)
    else:
        tail = deltas[-10:]
        steady = sum(tail) / len(tail)
print(f"smoke_steady_ms:{steady:.2f}")
if steady > max_ms:
    raise SystemExit(f"smoke too slow: {steady:.2f}ms > {max_ms:.2f}ms")
PY
    log "smoke accepted; log=${log_file}; timeout exit 124 during serialization is acceptable"
}

main() {
    validate_system
    ensure_repo
    cd "${REPO_DIR}"
    install_python_deps
    ensure_flash_attention
    validate_repo_files
    download_data
    validate_data
    run_smoke
    log "setup complete"
}

main "$@"
