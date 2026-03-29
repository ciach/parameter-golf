#!/bin/bash
set -e

# Round 6: Frontier Reproduction + Delta Experiments
# Strategy: one variable at a time, strict sequence

echo "Round 6: Frontier Catch-Up + Innovation"
echo "========================================"

RESULTS_LOG="experiment_logs_r6/results_summary_r6.txt"
mkdir -p experiment_logs_r6
echo "Experiment ID                 | Final BPB" > $RESULTS_LOG
echo "----------------------------------------" >> $RESULTS_LOG

# Install zstandard if missing (needed for zstd-22 compression)
pip install -q zstandard 2>/dev/null || true

run_exp() {
    exp_id=$1
    shift
    extra_env=$1
    shift

    echo "[$(date +'%H:%M:%S')] Starting Exp $exp_id"

    # Run with the frontier train_gpt.py
    eval "$extra_env torchrun --standalone --nproc_per_node=8 train_gpt_frontier.py" \
        > experiment_logs_r6/log_${exp_id}.txt 2>&1 || true

    # Extract final sliding window BPB (the real score)
    final_val=$(grep "final_int6_sliding_window_exact\|final_int8_zlib_roundtrip_exact" \
        experiment_logs_r6/log_${exp_id}.txt | tail -n 1 | \
        grep -oP 'val_bpb:\K[0-9.]+')
    
    # Fallback to last val_bpb if sliding window didn't run
    if [ -z "$final_val" ]; then
        final_val=$(grep "val_bpb:" experiment_logs_r6/log_${exp_id}.txt | \
            tail -n 1 | grep -oP 'val_bpb:\K[0-9.]+')
    fi

    if [ -z "$final_val" ]; then
        final_val="FAILED"
    fi

    echo "[$(date +'%H:%M:%S')] Finished $exp_id: val_bpb = $final_val"
    echo "========================================"
    printf "%-30s | %s\n" "$exp_id" "$final_val" >> $RESULTS_LOG
}

# -------------------------------------------------------------------
# Run 1: Faithful reproduction of PR #401 (1.1233 target)
# All defaults in train_gpt_frontier.py are already set to the frontier config.
# We just need to run it as-is.
# -------------------------------------------------------------------
run_exp "1_frontier_reproduction" ""

# -------------------------------------------------------------------
# Run 2: LeakyReLU² delta
# Swap relu² → LeakyReLU(0.5)² — our proven activation improvement
# (Requires LEAKY_RELU=1 env var — we'll add this toggle to the script)
# -------------------------------------------------------------------
# NOTE: LeakyReLU toggle needs to be added to train_gpt_frontier.py first
# run_exp "2_leaky_relu2_delta" "LEAKY_RELU=1"

echo ""
echo "Round 6 Run 1 complete. Check results:"
cat $RESULTS_LOG
