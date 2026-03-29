#!/bin/bash
set -uo pipefail

# Round 9: cap-first batch with packed int6 + embed int6.
# Goal: stay under 16,000,000 bytes while recovering BPB.

STARTUP_SLEEP_SECONDS="${STARTUP_SLEEP_SECONDS:-10}"
if [ "${STARTUP_SLEEP_SECONDS}" -gt 0 ]; then
    sleep "${STARTUP_SLEEP_SECONDS}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BATCH_NAME="${BATCH_NAME:-$(date +%Y-%m-%d_%H-%M-%S)_cap_first_int6_embed}"
LOG_ROOT="${LOG_ROOT:-experiment_logs_r9}"
BATCH_DIR="${LOG_ROOT}/${BATCH_NAME}"
RESULTS_LOG="${BATCH_DIR}/results_summary.txt"
MANIFEST_LOG="${BATCH_DIR}/run_manifest.tsv"

mkdir -p "${BATCH_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
    DETECTED_GPUS="8"
fi
N_GPUS="${N_GPUS:-${DETECTED_GPUS}}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

echo "Round 9: cap-first int6 packed+embed batch"
echo "Batch directory: ${BATCH_DIR}"
echo "Using ${N_GPUS} GPUs"
echo ""

# Preflight data/tokenizer checks to avoid burning failed runs.
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
if [ ! -f "${TOKENIZER_PATH}" ]; then
    echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"
    echo "Hint: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi
if ! ls "${DATA_PATH}"/fineweb_train_*.bin >/dev/null 2>&1; then
    echo "ERROR: no training shards found under: ${DATA_PATH}"
    echo "Hint: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi
if ! ls "${DATA_PATH}"/fineweb_val_*.bin >/dev/null 2>&1; then
    echo "ERROR: no validation shards found under: ${DATA_PATH}"
    echo "Hint: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi
echo "data_preflight:ok data_path=${DATA_PATH} tokenizer_path=${TOKENIZER_PATH}"
echo ""

# Ensure we use zstd path, not zlib fallback.
python3 - <<'PY'
import zstandard
print(f"zstandard:{zstandard.__version__}")
PY

printf "run_id\tscript\tdescription\tlog_file\tenv\n" > "${MANIFEST_LOG}"
printf "%-30s | %-12s | %-11s | %-6s | %s\n" "Run ID" "Final BPB" "Total Bytes" "Cap" "Log File" > "${RESULTS_LOG}"
printf "%s\n" "----------------------------------------------------------------------------------------------------" >> "${RESULTS_LOG}"

extract_metric() {
    local log_file="$1"
    local metric
    metric="$(
        grep -E "final_int6_sliding_window_exact|final_int8_zlib_roundtrip_exact" "${log_file}" 2>/dev/null \
            | tail -n 1 \
            | grep -oE 'val_bpb:[0-9.]+' \
            | cut -d: -f2
    )" || true
    if [ -z "${metric}" ]; then
        metric="$(
            grep -E "val_bpb:" "${log_file}" 2>/dev/null \
                | tail -n 1 \
                | grep -oE 'val_bpb:[0-9.]+' \
                | cut -d: -f2
        )" || true
    fi
    printf "%s" "${metric}"
}

extract_total_bytes() {
    local log_file="$1"
    local total_bytes
    total_bytes="$(
        grep -E "Total submission size int6\\+|Total submission size int8\\+zlib:" "${log_file}" 2>/dev/null \
            | tail -n 1 \
            | grep -oE '[0-9]+ bytes' \
            | cut -d' ' -f1
    )" || true
    printf "%s" "${total_bytes}"
}

extract_cap_status() {
    local log_file="$1"
    local status
    status="$(
        grep -E "Submission cap check:" "${log_file}" 2>/dev/null \
            | tail -n 1 \
            | grep -oE 'PASS|FAIL'
    )" || true
    printf "%s" "${status}"
}

run_exp() {
    local run_id="$1"
    local env_prefix="$2"
    local description="$3"
    local log_file="${BATCH_DIR}/log_${run_id}.txt"
    local exit_code
    local metric
    local total_bytes
    local cap_status

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${run_id}" "train_gpt_frontier.py" "${description}" "${log_file}" "${env_prefix}" >> "${MANIFEST_LOG}"

    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting ${run_id}"
    echo "Description: ${description}"
    echo "Env: ${env_prefix}"

    eval "${env_prefix} RUN_ID=\"${run_id}\" \"${TORCHRUN_BIN}\" --standalone --nproc_per_node=\"${N_GPUS}\" train_gpt_frontier.py" \
        > "${log_file}" 2>&1
    exit_code=$?

    metric="$(extract_metric "${log_file}")"
    total_bytes="$(extract_total_bytes "${log_file}")"
    cap_status="$(extract_cap_status "${log_file}")"

    if [ -z "${metric}" ]; then
        metric="FAILED"
    fi
    if [ -z "${total_bytes}" ]; then
        total_bytes="UNKNOWN"
    fi
    if [ -z "${cap_status}" ]; then
        cap_status="N/A"
    fi
    if [ "${exit_code}" -ne 0 ]; then
        metric="${metric} (exit:${exit_code})"
    fi

    echo "[$(date +'%H:%M:%S')] Finished ${run_id}: bpb=${metric} bytes=${total_bytes} cap=${cap_status}"
    echo "Log: ${log_file}"
    printf "%-30s | %-12s | %-11s | %-6s | %s\n" "${run_id}" "${metric}" "${total_bytes}" "${cap_status}" "${log_file}" >> "${RESULTS_LOG}"
}

BASE_ENV="SUBMISSION_SIZE_CAP_BYTES=16000000 EVAL_STRIDE=64 QUANT_COMPRESSOR=auto INT6_PACK=0 MIXED_QUANT_INT6_CATS=mlp,attn,embed MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=16384"

run_exp \
    "00_cap_anchor_mlp2p75_bg1024" \
    "${BASE_ENV} MLP_MULT=2.75 BIGRAM_VOCAB_SIZE=1024" \
    "Cap-anchor run expected to stay below cap with margin"

run_exp \
    "01_quality_step_mlp2p875_bg1024" \
    "${BASE_ENV} MLP_MULT=2.875 BIGRAM_VOCAB_SIZE=1024" \
    "Quality recovery attempt while preserving cap-first quantization settings"

run_exp \
    "02_quality_push_mlp2p875_bg1536" \
    "${BASE_ENV} MLP_MULT=2.875 BIGRAM_VOCAB_SIZE=1536" \
    "Add back capacity (BigramHash) and test if still below cap"

echo ""
echo "Batch complete. Summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Manifest: ${MANIFEST_LOG}"
