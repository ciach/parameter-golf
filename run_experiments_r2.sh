#!/bin/bash

# run_experiments_r2.sh
# Automation script for Round 2 of OFAT parameter-golf experiments.

mkdir -p experiment_logs_r2
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs. Running experiments using torchrun --standalone --nproc_per_node=${N_GPUS}"
echo ""

run_exp() {
    local exp_id=$1
    local extra_args=$2
    local log_file="experiment_logs_r2/log_${exp_id}.txt"
    
    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting Exp ${exp_id}"
    echo "Config: ${extra_args}"
    
    # Run the experiment
    env RUN_ID="${exp_id}" ${extra_args} torchrun --standalone --nproc_per_node="${N_GPUS}" train_gpt.py > "${log_file}" 2>&1
    
    # Parse the precise BPB
    local bpb=$(grep "final_int8_zlib_roundtrip val_loss:" "${log_file}" | tail -n 1 | awk -F'val_bpb:' '{print $2}' | awk '{print $1}')
    if [ -z "$bpb" ]; then bpb="FAILED/OOM"; fi
    
    echo "[$(date +'%H:%M:%S')] Finished ${exp_id}: val_bpb = ${bpb}"
    printf "%-27s | %s\n" "${exp_id}" "${bpb}" >> experiment_logs_r2/results_summary_r2.txt
}

echo "Experiment ID                 | Final BPB" > experiment_logs_r2/results_summary_r2.txt
echo "----------------------------------------" >> experiment_logs_r2/results_summary_r2.txt

# ----------------------------------------
# 0. The Round 1 WINNER Baseline String
# ----------------------------------------
BASE_ARGS="NUM_LAYERS=6 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 TRAIN_SEQ_LEN=2048 GRAD_CLIP_NORM=1.0 SCALAR_LR=0.02"

# ----------------------------------------
# NEW BASELINE TEST
# ----------------------------------------
run_exp "00_R2_baseline" "${BASE_ARGS}"

# ----------------------------------------
# Phase 1: Context Limits
# ----------------------------------------
run_exp "1.1_extreme_context_4096" "${BASE_ARGS} TRAIN_SEQ_LEN=4096"
run_exp "1.2_hybrid_context_3072" "${BASE_ARGS} TRAIN_SEQ_LEN=3072"

# ----------------------------------------
# Phase 2: Stacking Parameters
# ----------------------------------------
run_exp "2.1_the_fat_mlp" "${BASE_ARGS} MLP_MULT=4"
run_exp "2.2_standard_attention" "${BASE_ARGS} NUM_KV_HEADS=12"

# ----------------------------------------
# Phase 3: Initialization Variances
# ----------------------------------------
run_exp "3.1_high_embed_init" "${BASE_ARGS} TIED_EMBED_INIT_STD=0.01"
run_exp "3.2_high_qk_gain" "${BASE_ARGS} QK_GAIN_INIT=2.0"
run_exp "3.3_low_qk_gain" "${BASE_ARGS} QK_GAIN_INIT=1.0"

echo "========================================"
echo "Round 2 experiments completed!"
cat experiment_logs_r2/results_summary_r2.txt
