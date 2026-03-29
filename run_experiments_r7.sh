#!/bin/bash
set -uo pipefail

# Round 7: one record-repeat + four frontier experiments.
# This preserves a submission-grade log for the current best 1.1529 run while
# using the same pod session to test the next frontier ideas.

STARTUP_SLEEP_SECONDS="${STARTUP_SLEEP_SECONDS:-10}"
if [ "${STARTUP_SLEEP_SECONDS}" -gt 0 ]; then
    sleep "${STARTUP_SLEEP_SECONDS}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BATCH_NAME="${BATCH_NAME:-$(date +%Y-%m-%d_%H-%M-%S)_record_plus_frontier}"
LOG_ROOT="${LOG_ROOT:-experiment_logs_r7}"
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

echo "Round 7: record repeat + frontier deltas"
echo "Batch directory: ${BATCH_DIR}"
echo "Using ${N_GPUS} GPUs"
echo ""

printf "run_id\tscript\tdescription\tlog_file\tenv\n" > "${MANIFEST_LOG}"
printf "%-32s | %-16s | %s\n" "Run ID" "Final BPB" "Log File" > "${RESULTS_LOG}"
printf "%s\n" "--------------------------------------------------------------------------------" >> "${RESULTS_LOG}"

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

run_exp() {
    local run_id="$1"
    local script_name="$2"
    local env_prefix="$3"
    local description="$4"
    local log_file="${BATCH_DIR}/log_${run_id}.txt"
    local exit_code
    local metric

    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${run_id}" "${script_name}" "${description}" "${log_file}" "${env_prefix}" >> "${MANIFEST_LOG}"

    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting ${run_id}"
    echo "Script: ${script_name}"
    echo "Description: ${description}"
    echo "Env: ${env_prefix:-<defaults>}"

    if [ -n "${env_prefix}" ]; then
        eval "${env_prefix} RUN_ID=\"${run_id}\" \"${TORCHRUN_BIN}\" --standalone --nproc_per_node=\"${N_GPUS}\" \"${script_name}\"" \
            > "${log_file}" 2>&1
    else
        RUN_ID="${run_id}" "${TORCHRUN_BIN}" --standalone --nproc_per_node="${N_GPUS}" "${script_name}" \
            > "${log_file}" 2>&1
    fi
    exit_code=$?

    metric="$(extract_metric "${log_file}")"
    if [ -z "${metric}" ]; then
        metric="FAILED"
    fi
    if [ "${exit_code}" -ne 0 ]; then
        metric="${metric} (exit:${exit_code})"
    fi

    echo "[$(date +'%H:%M:%S')] Finished ${run_id}: ${metric}"
    echo "Log: ${log_file}"
    printf "%-32s | %-16s | %s\n" "${run_id}" "${metric}" "${log_file}" >> "${RESULTS_LOG}"
}

R4_RECORD_ENV="NUM_LAYERS=6 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 TRAIN_SEQ_LEN=4096 TRAIN_BATCH_TOKENS=524288 GRAD_CLIP_NORM=1.0 SCALAR_LR=0.02 MLP_MULT=4 QK_GAIN_INIT=2.0 LEAKY_RELU=1"
FRONTIER_BASE_ENV="EVAL_STRIDE=64"

run_exp \
    "00_record_repeat_r4_baseline" \
    "train_gpt.py" \
    "${R4_RECORD_ENV}" \
    "Repeat the 1.1529 Round 4 winner and capture the log for records"

run_exp \
    "01_frontier_reproduction" \
    "train_gpt_frontier.py" \
    "${FRONTIER_BASE_ENV}" \
    "Frontier reproduction baseline from the Round 6 plan"

run_exp \
    "02_frontier_leaky_relu2" \
    "train_gpt_frontier.py" \
    "${FRONTIER_BASE_ENV} LEAKY_RELU=1 LEAKY_RELU_SLOPE=0.5" \
    "Swap frontier relu^2 for LeakyReLU(0.5)^2"

run_exp \
    "03_frontier_seq4096" \
    "train_gpt_frontier.py" \
    "${FRONTIER_BASE_ENV} TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096" \
    "Test longer-context frontier training/eval at sequence length 4096"

run_exp \
    "04_frontier_bigram4096" \
    "train_gpt_frontier.py" \
    "${FRONTIER_BASE_ENV} BIGRAM_VOCAB_SIZE=4096" \
    "Scale BigramHash buckets from 2048 to 4096"

echo ""
echo "Batch complete. Summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Manifest: ${MANIFEST_LOG}"
