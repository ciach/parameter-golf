#!/bin/bash
set -uo pipefail

# Round 8: decimal-16MB frontier repeat + two size-focused discoveries.

STARTUP_SLEEP_SECONDS="${STARTUP_SLEEP_SECONDS:-10}"
if [ "${STARTUP_SLEEP_SECONDS}" -gt 0 ]; then
    sleep "${STARTUP_SLEEP_SECONDS}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BATCH_NAME="${BATCH_NAME:-$(date +%Y-%m-%d_%H-%M-%S)_decimal_16mb_frontier}"
LOG_ROOT="${LOG_ROOT:-experiment_logs_r8}"
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

echo "Round 8: decimal-16MB frontier repeat + size deltas"
echo "Batch directory: ${BATCH_DIR}"
echo "Using ${N_GPUS} GPUs"
echo ""

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

BASE_ENV="SUBMISSION_SIZE_CAP_BYTES=16000000 EVAL_STRIDE=64"

run_exp \
    "00_frontier_repeat_decimal_cap" \
    "${BASE_ENV}" \
    "Repeat the current best frontier reproduction and score it against the true decimal 16MB cap"

run_exp \
    "01_frontier_mlp_mult_2p75" \
    "${BASE_ENV} MLP_MULT=2.75" \
    "Discovery: shrink the dominant MLP block while keeping the rest of the frontier stack unchanged"

run_exp \
    "02_frontier_10_layers" \
    "${BASE_ENV} NUM_LAYERS=10" \
    "Discovery: remove one layer as a clean size-focused architecture trim"

echo ""
echo "Batch complete. Summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Manifest: ${MANIFEST_LOG}"
