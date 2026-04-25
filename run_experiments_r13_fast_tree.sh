#!/bin/bash
set -uo pipefail

# Round 13: gated fast tree from the two local best branches.
# Goal: find a <1.12 BPB path without spending runs on dead branches.

STARTUP_SLEEP_SECONDS="${STARTUP_SLEEP_SECONDS:-10}"
if [ "${STARTUP_SLEEP_SECONDS}" -gt 0 ]; then
    sleep "${STARTUP_SLEEP_SECONDS}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BATCH_NAME="${BATCH_NAME:-$(date +%Y-%m-%d_%H-%M-%S)_fast_tree_gated}"
LOG_ROOT="${LOG_ROOT:-experiment_logs_r13}"
BATCH_DIR="${LOG_ROOT}/${BATCH_NAME}"
RESULTS_LOG="${BATCH_DIR}/results_summary.txt"
MANIFEST_LOG="${BATCH_DIR}/run_manifest.tsv"
STATE_TSV="${BATCH_DIR}/state.tsv"
LOG_ARCHIVE="${LOG_ARCHIVE:-${LOG_ROOT}/${BATCH_NAME}.tar.gz}"
LOG_DOWNLOAD_DIR="${LOG_DOWNLOAD_DIR:-}"
LOG_SCP_DEST="${LOG_SCP_DEST:-}"
LOG_SCP_OPTS="${LOG_SCP_OPTS:-}"

mkdir -p "${BATCH_DIR}"

finalize_logs() {
    local exit_code=$?
    trap - EXIT
    if [ -d "${BATCH_DIR}" ]; then
        mkdir -p "$(dirname "${LOG_ARCHIVE}")"
        if tar -czf "${LOG_ARCHIVE}" -C "${LOG_ROOT}" "${BATCH_NAME}"; then
            echo "log_archive:${LOG_ARCHIVE}"
            if [ -n "${LOG_DOWNLOAD_DIR}" ]; then
                mkdir -p "${LOG_DOWNLOAD_DIR}"
                if cp "${LOG_ARCHIVE}" "${LOG_DOWNLOAD_DIR}/"; then
                    echo "log_download_copy:${LOG_DOWNLOAD_DIR}/$(basename "${LOG_ARCHIVE}")"
                else
                    echo "WARNING: failed to copy log archive to LOG_DOWNLOAD_DIR=${LOG_DOWNLOAD_DIR}"
                fi
            fi
            if [ -n "${LOG_SCP_DEST}" ]; then
                if scp ${LOG_SCP_OPTS} "${LOG_ARCHIVE}" "${LOG_SCP_DEST}"; then
                    echo "log_scp_dest:${LOG_SCP_DEST}"
                else
                    echo "WARNING: failed to scp log archive to LOG_SCP_DEST=${LOG_SCP_DEST}"
                fi
            fi
        else
            echo "WARNING: failed to create log archive ${LOG_ARCHIVE}"
        fi
    fi
    exit "${exit_code}"
}

trap finalize_logs EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
    DETECTED_GPUS="8"
fi
N_GPUS="${N_GPUS:-${DETECTED_GPUS}}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SOURCE_SCRIPT="${SOURCE_SCRIPT:-train_gpt_frontier.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt_frontier_min.py}"
MINIFIER_SCRIPT="${MINIFIER_SCRIPT:-tools/minify_train_script.py}"

# Local bests.
A_PARENT_BPB="${A_PARENT_BPB:-1.12273283}"
B_PARENT_BPB="${B_PARENT_BPB:-1.12276383}"
TARGET_BPB="${TARGET_BPB:-1.12000000}"

# Branch only if the first-pass candidate is both meaningfully better than its
# parent and close enough to plausibly reach target in one more run.
MIN_PARENT_IMPROVE="${MIN_PARENT_IMPROVE:-0.00030}"
BRANCH_GATE_BPB="${BRANCH_GATE_BPB:-1.12180}"
MAX_BRANCHES="${MAX_BRANCHES:-2}"
MAX_PROBES="${MAX_PROBES:-3}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-1200}"
STOP_ON_TARGET="${STOP_ON_TARGET:-1}"
PROBE_FAIL_FAST="${PROBE_FAIL_FAST:-1}"

echo "Round 13: gated fast tree"
echo "Batch directory: ${BATCH_DIR}"
echo "Using ${N_GPUS} GPUs"
echo "target_bpb:${TARGET_BPB} branch_gate_bpb:${BRANCH_GATE_BPB} min_parent_improve:${MIN_PARENT_IMPROVE} max_probes:${MAX_PROBES} max_branches:${MAX_BRANCHES} run_timeout_seconds:${RUN_TIMEOUT_SECONDS} probe_fail_fast:${PROBE_FAIL_FAST}"
echo ""

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

if ! python3 - <<'PY'
import zstandard
print(f"zstandard:{zstandard.__version__}")
PY
then
    echo "ERROR: zstandard is required for QUANT_COMPRESSOR=auto"
    echo "Install with: python3 -m pip install zstandard"
    exit 1
fi
echo ""

if [ ! -f "${SOURCE_SCRIPT}" ]; then
    echo "ERROR: source train script not found: ${SOURCE_SCRIPT}"
    exit 1
fi
if [ ! -f "${MINIFIER_SCRIPT}" ]; then
    echo "ERROR: minifier script not found: ${MINIFIER_SCRIPT}"
    exit 1
fi
python3 "${MINIFIER_SCRIPT}" --input "${SOURCE_SCRIPT}" --output "${TRAIN_SCRIPT}"
python3 -m py_compile "${TRAIN_SCRIPT}"
RAW_BYTES="$(wc -c < "${SOURCE_SCRIPT}" | tr -d ' ')"
MIN_BYTES="$(wc -c < "${TRAIN_SCRIPT}" | tr -d ' ')"
echo "minify:source=${SOURCE_SCRIPT} bytes=${RAW_BYTES}"
echo "minify:output=${TRAIN_SCRIPT} bytes=${MIN_BYTES}"
echo ""

printf "run_id\tphase\tparent\tparent_bpb\tlog_file\tenv\tdescription\n" > "${MANIFEST_LOG}"
printf "run_id\tphase\tparent\tparent_bpb\tmetric\ttotal_bytes\tcap\teval_ms\tenv\n" > "${STATE_TSV}"
printf "%-42s | %-8s | %-9s | %-12s | %-11s | %-6s | %-10s | %s\n" "Run ID" "Phase" "Parent" "Final BPB" "Total Bytes" "Cap" "Eval ms" "Log File" > "${RESULTS_LOG}"
printf "%s\n" "--------------------------------------------------------------------------------------------------------------------------------" >> "${RESULTS_LOG}"

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

extract_eval_time_ms() {
    local log_file="$1"
    local eval_ms
    eval_ms="$(
        grep -E "final_int6_sliding_window .*eval_time:[0-9]+ms" "${log_file}" 2>/dev/null \
            | tail -n 1 \
            | grep -oE 'eval_time:[0-9]+ms' \
            | cut -d: -f2 \
            | tr -d 'ms'
    )" || true
    printf "%s" "${eval_ms}"
}

target_hit() {
    local metric="$1"
    python3 - "${metric}" "${TARGET_BPB}" <<'PY'
import sys
try:
    sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
except ValueError:
    sys.exit(1)
PY
}

last_probe_bad() {
    python3 - "${STATE_TSV}" <<'PY'
import sys
from pathlib import Path

rows = Path(sys.argv[1]).read_text().strip().splitlines()
if len(rows) <= 1:
    sys.exit(1)
run_id, phase, parent, parent_bpb, metric, total, cap, eval_ms, env = rows[-1].split("\t", 8)
if phase != "probe":
    sys.exit(1)
try:
    bpb = float(metric.split()[0])
    pbpb = float(parent_bpb)
except ValueError:
    sys.exit(0)
sys.exit(0 if cap != "PASS" or bpb >= pbpb else 1)
PY
}

maybe_stop_after_probe() {
    if [ "${PROBE_FAIL_FAST}" != "1" ]; then
        return
    fi
    if last_probe_bad; then
        echo "probe_fail_fast: last probe failed, missed cap, or did not beat parent; stopping"
        cat "${RESULTS_LOG}"
        exit 2
    fi
}

run_exp() {
    local run_id="$1"
    local phase="$2"
    local parent="$3"
    local parent_bpb="$4"
    local env_prefix="$5"
    local description="$6"
    local log_file="${BATCH_DIR}/log_${run_id}.txt"
    local exit_code
    local metric
    local total_bytes
    local cap_status
    local eval_ms
    local run_cmd

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${run_id}" "${phase}" "${parent}" "${parent_bpb}" "${log_file}" "${env_prefix}" "${description}" >> "${MANIFEST_LOG}"

    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting ${run_id}"
    echo "Phase: ${phase} Parent: ${parent} parent_bpb:${parent_bpb}"
    echo "Description: ${description}"
    echo "Env: ${env_prefix}"

    run_cmd="${env_prefix} RUN_ID=\"${run_id}\" \"${TORCHRUN_BIN}\" --standalone --nproc_per_node=\"${N_GPUS}\" \"${TRAIN_SCRIPT}\""
    if [ "${RUN_TIMEOUT_SECONDS}" -gt 0 ]; then
        run_cmd="timeout ${RUN_TIMEOUT_SECONDS}s ${run_cmd}"
    fi
    eval "${run_cmd}" > "${log_file}" 2>&1
    exit_code=$?

    metric="$(extract_metric "${log_file}")"
    total_bytes="$(extract_total_bytes "${log_file}")"
    cap_status="$(extract_cap_status "${log_file}")"
    eval_ms="$(extract_eval_time_ms "${log_file}")"

    if [ -z "${metric}" ]; then
        metric="FAILED"
    fi
    if [ -z "${total_bytes}" ]; then
        total_bytes="UNKNOWN"
    fi
    if [ -z "${cap_status}" ]; then
        cap_status="N/A"
    fi
    if [ -z "${eval_ms}" ]; then
        eval_ms="N/A"
    fi
    if [ "${exit_code}" -ne 0 ]; then
        metric="${metric} (exit:${exit_code})"
    fi

    echo "[$(date +'%H:%M:%S')] Finished ${run_id}: bpb=${metric} bytes=${total_bytes} cap=${cap_status} eval_ms=${eval_ms}"
    echo "Log: ${log_file}"
    printf "%-42s | %-8s | %-9s | %-12s | %-11s | %-6s | %-10s | %s\n" "${run_id}" "${phase}" "${parent}" "${metric}" "${total_bytes}" "${cap_status}" "${eval_ms}" "${log_file}" >> "${RESULTS_LOG}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${run_id}" "${phase}" "${parent}" "${parent_bpb}" "${metric}" "${total_bytes}" "${cap_status}" "${eval_ms}" "${env_prefix}" >> "${STATE_TSV}"
}

# A = R11-01, current best-valid. B = R10-00, same BPB, lower keep-float.
BASE_A="SUBMISSION_SIZE_CAP_BYTES=16000000 EVAL_STRIDE=64 QUANT_COMPRESSOR=auto INT6_PACK=0 MIXED_QUANT_INT6_CATS=mlp,attn MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=24576 MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048 LATE_QAT_THRESHOLD=0 LATE_QAT_START_FRAC=0.90"
BASE_B="SUBMISSION_SIZE_CAP_BYTES=16000000 EVAL_STRIDE=64 QUANT_COMPRESSOR=auto INT6_PACK=0 MIXED_QUANT_INT6_CATS=mlp,attn MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=16384 MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048"
TTT_GUARD="TTT_WEIGHT_DECAY=0.0 TTT_MAX_UPDATES=256 TTT_GRAD_CLIP_NORM=1.0"

run_exp "01_A_ttt_lr001" "probe" "A" "${A_PARENT_BPB}" \
    "${BASE_A} TTT_ENABLED=1 TTT_LAYERS=9,10 TTT_LR=0.001 TTT_CHUNK_TOKENS=128 TTT_MAX_TOKENS_PER_DOC=512 ${TTT_GUARD}" \
    "A: conservative V-only TTT"

if [ "${STOP_ON_TARGET}" = "1" ]; then
    last_metric="$(tail -n 1 "${STATE_TSV}" | cut -f5)"
    if target_hit "${last_metric}"; then
        echo "target_hit:${last_metric}; stopping"
        cat "${RESULTS_LOG}"
        exit 0
    fi
fi
maybe_stop_after_probe

if [ "${MAX_PROBES}" -ge 2 ]; then
    run_exp "02_A_ttt_lr005" "probe" "A" "${A_PARENT_BPB}" \
        "${BASE_A} TTT_ENABLED=1 TTT_LAYERS=9,10 TTT_LR=0.005 TTT_CHUNK_TOKENS=128 TTT_MAX_TOKENS_PER_DOC=512 ${TTT_GUARD}" \
        "A: stronger V-only TTT"

    if [ "${STOP_ON_TARGET}" = "1" ]; then
        last_metric="$(tail -n 1 "${STATE_TSV}" | cut -f5)"
        if target_hit "${last_metric}"; then
            echo "target_hit:${last_metric}; stopping"
            cat "${RESULTS_LOG}"
            exit 0
        fi
    fi
    maybe_stop_after_probe
else
    echo "max_probes:${MAX_PROBES}; skipping 02_A_ttt_lr005"
fi

if [ "${MAX_PROBES}" -ge 3 ]; then
    run_exp "03_B_ttt_lr005" "probe" "B" "${B_PARENT_BPB}" \
        "${BASE_B} TTT_ENABLED=1 TTT_LAYERS=9,10 TTT_LR=0.005 TTT_CHUNK_TOKENS=128 TTT_MAX_TOKENS_PER_DOC=512 ${TTT_GUARD}" \
        "B: stronger V-only TTT from lower keep-float branch"

    if [ "${STOP_ON_TARGET}" = "1" ]; then
        last_metric="$(tail -n 1 "${STATE_TSV}" | cut -f5)"
        if target_hit "${last_metric}"; then
            echo "target_hit:${last_metric}; stopping"
            cat "${RESULTS_LOG}"
            exit 0
        fi
    fi
    maybe_stop_after_probe
else
    echo "max_probes:${MAX_PROBES}; skipping 03_B_ttt_lr005"
fi

BRANCH_FILE="${BATCH_DIR}/branch_candidates.tsv"
python3 - "${STATE_TSV}" "${BRANCH_FILE}" "${BRANCH_GATE_BPB}" "${MIN_PARENT_IMPROVE}" "${MAX_BRANCHES}" <<'PY'
import sys
from pathlib import Path

state_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
gate = float(sys.argv[3])
min_improve = float(sys.argv[4])
max_branches = int(sys.argv[5])
cands = []
with state_path.open() as f:
    next(f)
    for line in f:
        run_id, phase, parent, parent_bpb, metric, total, cap, eval_ms, env = line.rstrip("\n").split("\t", 8)
        if phase != "probe" or cap != "PASS":
            continue
        try:
            bpb = float(metric)
            pbpb = float(parent_bpb)
        except ValueError:
            continue
        if bpb <= gate and (pbpb - bpb) >= min_improve:
            cands.append((bpb, run_id, parent, parent_bpb, env))
cands.sort()
with out_path.open("w") as out:
    out.write("metric\trun_id\tparent\tparent_bpb\tenv\n")
    for bpb, run_id, parent, parent_bpb, env in cands[:max_branches]:
        out.write(f"{bpb:.8f}\t{run_id}\t{parent}\t{parent_bpb}\t{env}\n")
print(len(cands[:max_branches]))
PY
branch_count="$(($(wc -l < "${BRANCH_FILE}") - 1))"
if [ "${branch_count}" -le 0 ]; then
    echo "No branch candidates crossed gate. Fast fail."
    echo "Gate: metric <= ${BRANCH_GATE_BPB} and parent improvement >= ${MIN_PARENT_IMPROVE}"
    cat "${RESULTS_LOG}"
    exit 2
fi

echo "Branching ${branch_count} candidate(s):"
cat "${BRANCH_FILE}"

branch_idx=0
while IFS=$'\t' read -r metric parent_run parent parent_bpb parent_env; do
    branch_idx=$((branch_idx + 1))
    base_id="$(printf "%02d_%s" "${branch_idx}" "${parent_run}")"

    # Same candidate, more adaptation budget.
    budget_env="$(echo "${parent_env}" \
        | sed -E 's/TTT_CHUNK_TOKENS=[^ ]+/TTT_CHUNK_TOKENS=256/g; s/TTT_MAX_TOKENS_PER_DOC=[^ ]+/TTT_MAX_TOKENS_PER_DOC=1024/g; s/TTT_MAX_UPDATES=[^ ]+/TTT_MAX_UPDATES=512/g')"
    run_exp "${base_id}_budget1024" "branch" "${parent_run}" "${metric}" \
        "${budget_env}" \
        "Branch from ${parent_run}: spend more eval-time adaptation"

    if [ "${STOP_ON_TARGET}" = "1" ]; then
        last_metric="$(tail -n 1 "${STATE_TSV}" | cut -f5)"
        if target_hit "${last_metric}"; then
            echo "target_hit:${last_metric}; stopping"
            cat "${RESULTS_LOG}"
            exit 0
        fi
    fi

    # Deeper V-only adaptation. Use slightly lower LR to avoid destabilizing 4 layers.
    deep_env="$(echo "${parent_env}" \
        | sed -E 's/TTT_LAYERS=[^ ]+/TTT_LAYERS=7,8,9,10/g; s/TTT_LR=[^ ]+/TTT_LR=0.003/g; s/TTT_MAX_UPDATES=[^ ]+/TTT_MAX_UPDATES=512/g')"
    run_exp "${base_id}_deep4" "branch" "${parent_run}" "${metric}" \
        "${deep_env}" \
        "Branch from ${parent_run}: deeper V-only adaptation"

    if [ "${STOP_ON_TARGET}" = "1" ]; then
        last_metric="$(tail -n 1 "${STATE_TSV}" | cut -f5)"
        if target_hit "${last_metric}"; then
            echo "target_hit:${last_metric}; stopping"
            cat "${RESULTS_LOG}"
            exit 0
        fi
    fi
done < <(tail -n +2 "${BRANCH_FILE}")

echo ""
echo "Batch complete. Summary:"
cat "${RESULTS_LOG}"
echo ""
echo "Manifest: ${MANIFEST_LOG}"
