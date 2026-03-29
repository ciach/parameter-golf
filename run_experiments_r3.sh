#!/bin/bash

# run_experiments_r3.sh
# Round 3 OFAT testing advanced algorithmic SOTA architectures on 1 GPU.

mkdir -p experiment_logs_r3
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs. Running experiments using torchrun --standalone --nproc_per_node=${N_GPUS}"
echo ""

run_exp() {
    local exp_id=$1
    local extra_args=$2
    local log_file="experiment_logs_r3/log_${exp_id}.txt"
    
    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting Exp ${exp_id}"
    echo "Config: ${extra_args}"
    
    # Run the experiment
    env RUN_ID="${exp_id}" ${extra_args} torchrun --standalone --nproc_per_node="${N_GPUS}" train_gpt.py > "${log_file}" 2>&1
    
    # Parse the precise BPB
    local bpb=$(grep "final_int8_zlib_roundtrip val_loss:" "${log_file}" | tail -n 1 | awk -F'val_bpb:' '{print $2}' | awk '{print $1}')
    if [ -z "$bpb" ]; then bpb="FAILED/OOM"; fi
    
    echo "[$(date +'%H:%M:%S')] Finished ${exp_id}: val_bpb = ${bpb}"
    printf "%-27s | %s\n" "${exp_id}" "${bpb}" >> experiment_logs_r3/results_summary_r3.txt
}

echo "Experiment ID                 | Final BPB" > experiment_logs_r3/results_summary_r3.txt
echo "----------------------------------------" >> experiment_logs_r3/results_summary_r3.txt

# ----------------------------------------
# 0. The Round 2 WINNER Baseline String
# ----------------------------------------
# Standard Llama Architecture Math: relu^2 + unet_skip=1
BASE_ARGS="NUM_LAYERS=6 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 TRAIN_SEQ_LEN=4096 GRAD_CLIP_NORM=1.0 SCALAR_LR=0.02 MLP_MULT=4 QK_GAIN_INIT=2.0"

run_exp "00_R3_baseline" "${BASE_ARGS}"

# ----------------------------------------
# Algorithmic Manipulations
# ----------------------------------------
run_exp "1_leaky_relu_sq" "${BASE_ARGS} LEAKY_RELU=1"
run_exp "2_strict_transformer" "${BASE_ARGS} UNET_SKIP=0"
run_exp "3_swiglu_gate" "${BASE_ARGS} SWIGLU=1"

# ----------------------------------------
# The Hybrid Stack
# ----------------------------------------
run_exp "4_the_hybrid_model" "${BASE_ARGS} LEAKY_RELU=1 UNET_SKIP=0"

echo "========================================"
echo "Round 3 experiments completed!"
cat experiment_logs_r3/results_summary_r3.txt
