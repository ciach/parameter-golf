#!/bin/bash

# run_experiments.sh
# Automation script to run the OFAT parameter-golf experiments sequentially.

# Create logs directory
mkdir -p experiment_logs

# Auto-detect number of GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${N_GPUS} GPUs. Running experiments using torchrun --standalone --nproc_per_node=${N_GPUS}"
echo ""

# Helper function to run a single experiment
run_exp() {
    local exp_id=$1
    local extra_args=$2
    local log_file="experiment_logs/log_${exp_id}.txt"
    
    echo "========================================"
    echo "[$(date +'%H:%M:%S')] Starting Exp ${exp_id}"
    echo "Config: ${extra_args}"
    echo "Tail Logs: tail -f ${log_file}"
    
    # Run the script with the provided environment variables
    # We use 'env' to pass the variables inline before the command
    env RUN_ID="${exp_id}" ${extra_args} torchrun --standalone --nproc_per_node="${N_GPUS}" train_gpt.py > "${log_file}" 2>&1
    
    # Extract the final val_bpb from the log
    local bpb=$(grep "final_int8_zlib_roundtrip val_bpb:" "${log_file}" | tail -n 1 | awk -F'val_bpb:' '{print $2}' | awk '{print $1}')
    
    if [ -z "$bpb" ]; then
        bpb="FAILED/OOM"
    fi
    
    echo "[$(date +'%H:%M:%S')] Finished ${exp_id}: val_bpb = ${bpb}"
    echo "${exp_id} | ${extra_args} | ${bpb}" >> experiment_logs/results_summary.txt
}

# Clear old summary
echo "Experiment ID | Config | Final BPB" > experiment_logs/results_summary.txt
echo "----------------------------------------" >> experiment_logs/results_summary.txt

# ----------------------------------------
# 0. Base configuration
# ----------------------------------------
run_exp "00_baseline" ""

# ----------------------------------------
# 1. Phase 1: Learning Rates
# ----------------------------------------
run_exp "1.1_matrix_lr_high" "MATRIX_LR=0.08"
run_exp "1.2_matrix_lr_low" "MATRIX_LR=0.02"
run_exp "1.3_scalar_lr_high" "SCALAR_LR=0.08"
run_exp "1.4_scalar_lr_low" "SCALAR_LR=0.02"
run_exp "1.5_tied_embed_lr_high" "TIED_EMBED_LR=0.10"

# ----------------------------------------
# 2. Phase 2: Architecture Sizing
# ----------------------------------------
run_exp "2.1_deeper_narrower" "NUM_LAYERS=12 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3"
run_exp "2.2_shallower_wider" "NUM_LAYERS=6 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6"
run_exp "2.3_bigger_mlp" "NUM_LAYERS=7 MLP_MULT=4"

# ----------------------------------------
# 3. Phase 3: Token Sizing
# ----------------------------------------
run_exp "3.1_shorter_context" "TRAIN_SEQ_LEN=512"
run_exp "3.2_longer_context" "TRAIN_SEQ_LEN=2048"

# ----------------------------------------
# 4. Phase 4: Regularization
# ----------------------------------------
run_exp "4.1_grad_clip" "GRAD_CLIP_NORM=1.0"
run_exp "4.2_high_softcap" "LOGIT_SOFTCAP=50.0"
run_exp "4.3_low_softcap" "LOGIT_SOFTCAP=15.0"
run_exp "4.4_high_muon_mom" "MUON_MOMENTUM=0.98"

echo "========================================"
echo "All experiments completed!"
echo "Summary of results:"
cat experiment_logs/results_summary.txt
