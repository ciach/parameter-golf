#!/bin/bash
set -e

# Wait for 10 seconds before starting to give enough headstart to the pod boot process
sleep 10

# Master Round 5: The 1.1194 World Record Attempt
echo "Detected 8 GPUs. Running exactly on 8x Multi-Node Distributed Architecture."
echo ""
echo "========================================"

RESULTS_LOG="experiment_logs_r5/results_summary_r5.txt"
mkdir -p experiment_logs_r5
echo "Experiment ID                 | Final BPB" > $RESULTS_LOG
echo "----------------------------------------" >> $RESULTS_LOG

BASE_ARGS="--num_layers 6 --model_dim 768 --num_heads 12 --num_kv_heads 6 --train_seq_len 4096 --train_batch_tokens 524288 --grad_clip_norm 1.0 --scalar_lr 0.02 --mlp_mult 4 --qk_gain_init 2.0"

export LEAKY_RELU=1

run_exp() {
    exp_id=$1
    shift
    extra_env=$1
    shift
    extra_args="$@"

    echo "[$(date +'%H:%M:%S')] Starting Exp $exp_id"
    echo "Config: $BASE_ARGS $extra_args EV($extra_env)"

    # Execute torchrun inside the subshell handling dynamic env vars
    eval "$extra_env torchrun --nproc_per_node=8 train_gpt.py $BASE_ARGS $extra_args" > experiment_logs_r5/log_${exp_id}.txt 2>&1 || true

    # Extract final BPB cleanly from the log file tail
    final_val=$(grep "val_bpb:" experiment_logs_r5/log_${exp_id}.txt | tail -n 1 | awk -F'val_bpb:' '{print $2}' | awk '{print $1}')
    if [ -z "$final_val" ]; then
        final_val="FAILED/OOM"
    fi

    echo "[$(date +'%H:%M:%S')] Finished $exp_id: val_bpb = $final_val"
    echo "========================================"
    printf "%-30s | %s\n" "$exp_id" "$final_val" >> $RESULTS_LOG
}

# The 4 LoRA Configurations (Targeting 1.119 BPB)
run_exp "1_lora_ttt_rank16_lr_0.01" "TTT_LORA_RANK=16" "--ttt_lr 0.01"
run_exp "2_lora_ttt_rank16_lr_0.05" "TTT_LORA_RANK=16" "--ttt_lr 0.05"
run_exp "3_lora_ttt_rank32_lr_0.01" "TTT_LORA_RANK=32" "--ttt_lr 0.01"
run_exp "4_lora_ttt_rank32_lr_0.05" "TTT_LORA_RANK=32" "--ttt_lr 0.05"

echo "Round 5 experiments completed!"
cat $RESULTS_LOG
