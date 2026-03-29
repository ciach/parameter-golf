#!/bin/bash

# run_experiments_r4.sh
# Round 4 OFAT testing 8-GPU Scaling Variables and Test-Time Training (TTT).

mkdir -p experiment_logs_r4
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs. Running exactly on 8x Multi-Node Distributed Architecture."
echo ""

run_exp() {
    local exp_id=$1
    local extra_args=$2
    local log_file="experiment_logs_r4/log_${exp_id}.txt"
    
    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting Exp ${exp_id}"
    echo "Config: ${extra_args}"
    
    # Run the experiment over all allocated GPUs natively
    env RUN_ID="${exp_id}" ${extra_args} \
    torchrun --standalone --nproc_per_node="${N_GPUS}" train_gpt.py > "${log_file}" 2>&1
    
    # Parse the precise BPB
    local bpb=$(grep "final_int8_zlib_roundtrip val_loss:" "${log_file}" | tail -n 1 | awk -F'val_bpb:' '{print $2}' | awk '{print $1}')
    if [ -z "$bpb" ]; then bpb="FAILED/OOM"; fi
    
    echo "[$(date +'%H:%M:%S')] Finished ${exp_id}: val_bpb = ${bpb}"
    printf "%-27s | %s\n" "${exp_id}" "${bpb}" >> experiment_logs_r4/results_summary_r4.txt
}

echo "Experiment ID                 | Final BPB" > experiment_logs_r4/results_summary_r4.txt
echo "----------------------------------------" >> experiment_logs_r4/results_summary_r4.txt

# ----------------------------------------
# 0. The Round 3 1-GPU WINNER Structure
# ----------------------------------------
# Optimal Shape: Leaky ReLU + Fat MLP + 4096 Context
BASE_ARGS="NUM_LAYERS=6 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 TRAIN_SEQ_LEN=4096 GRAD_CLIP_NORM=1.0 SCALAR_LR=0.02 MLP_MULT=4 QK_GAIN_INIT=2.0 LEAKY_RELU=1"

run_exp "00_R4_8g_baseline" "${BASE_ARGS}"

# ----------------------------------------
# Phase 1: Batch Geometry Optimization
# ----------------------------------------
# How does the 8-GPU cluster handle massive synchronized batch volumes vs tiny update volumes?
run_exp "1_one_million_batch" "${BASE_ARGS} TRAIN_BATCH_TOKENS=1048576"
run_exp "2_hyper_frequency" "${BASE_ARGS} TRAIN_BATCH_TOKENS=262144"

# ----------------------------------------
# Phase 2: Learning Rate Recalibration
# ----------------------------------------
# Doubling global learning rates to match multi-gpu parallelism
run_exp "3_high_velocity_lrs" "${BASE_ARGS} SCALAR_LR=0.04 MATRIX_LR=0.08"
# Warming up Muon Momentum 5x faster
run_exp "4_rapid_muon_warmup" "${BASE_ARGS} MUON_MOMENTUM_WARMUP_STEPS=100"

# ----------------------------------------
# Phase 3: Legal Test-Time Training (TTT)
# ----------------------------------------
# Dynamically updating weights backwards through the testing evaluation loop.
run_exp "5_ttt_initialization" "${BASE_ARGS} TTT_LR=0.001"
run_exp "6_ttt_aggressive" "${BASE_ARGS} TTT_LR=0.005"

echo "========================================"
echo "Round 4 experiments completed!"
cat experiment_logs_r4/results_summary_r4.txt
