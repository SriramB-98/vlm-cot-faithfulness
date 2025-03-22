#!/bin/bash
#SBATCH --qos=cml-medium
#SBATCH --partition=cml-dpart
#SBATCH --account=cml-sfeizi
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=cvbench_analysis_%j.out
#SBATCH --error=cvbench_analysis_%j.err

# time: ~4:00:00 for 72B (100 samples), 
# time: ~1:20:00 for 7B (100 samples)
hostname

# Base parameters
# MODEL="omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps"
# MODEL="OpenGVLab/InternVL2_5-78B-MPO-AWQ"
# MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
# MODEL="OpenGVLab/InternVL2_5-78B-MPO-AWQ"
MODEL="Xkev/Llama-3.2V-11B-cot"
NUM_CONTEXT_SAMPLES=8
NUM_CONTEXTS=1
SCALE_FACTOR=0.25
TOTAL_SAMPLES=100
SEED=20
GIVE_REASONING=1
SERVER="VLLM"

DESCRIPTION_FLAG=""
REASONING_FLAG=""
if [ "${GIVE_REASONING}" = "1" ]; then
    REASONING_FLAG="--give_reasoning"
fi
SERVER_FLAG=""
if [ -n "${SERVER}" ] && [ "${SERVER}" != "OPENAI" ]; then
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_' | tr ':' '_')
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "${SERVER}" = "VLLM" ]; then
        echo "Starting VLLM server..."

        if [ "${MODEL}" = "Xkev/Llama-3.2V-11B-cot" ]; then
            SERVER_FLAG="--server vllm --server_batch_size $NUM_GPUS"
            vllm serve $MODEL --max-model-len 6144 --tensor-parallel-size=$NUM_GPUS --limit-mm-per-prompt "image=1" --chat-template-content-format "openai" --mm-processor-kwargs '{}' --gpu-memory-utilization 0.9 --port 27182 --max-num-seqs 4 > ./server_logs/vllm_server_${SAFE_MODEL_NAME}.log 2>&1 &
        else
            SBATCH_SIZE=$(echo "$NUM_GPUS * 2" | bc)
            SERVER_FLAG="--server vllm --server_batch_size $SBATCH_SIZE"
            MAX_PIXELS=$(echo "1003520 * $SCALE_FACTOR" | bc | cut -d. -f1)
            vllm serve $MODEL --max-model-len 32768 --tensor-parallel-size=$NUM_GPUS --limit-mm-per-prompt "image=9" --chat-template-content-format "openai" \
            --mm-processor-kwargs '{"images_kwargs.do_resize":true, "images_kwargs.size.shortest_edge": 3136, "images_kwargs.size.longest_edge": '"$MAX_PIXELS"'}' --port 27182 > ./server_logs/vllm_server_${SAFE_MODEL_NAME}.log 2>&1 &
        fi
    elif [ "${SERVER}" = "LMDEPLOY" ]; then
        echo "Starting lmdeploy server..."
        MAX_PATCHES=$(echo "12 * $SCALE_FACTOR" | bc | cut -d. -f1)
        SBATCH_SIZE=$(echo "$NUM_GPUS * 2" | bc)
        SERVER_FLAG="--server lmdeploy --server_batch_size $SBATCH_SIZE"
        lmdeploy serve api_server $MODEL --server-port 27182 --tp 2 --session-len 32768 --max-dynamic-patch $MAX_PATCHES > ./server_logs/lmdeploy_server_${SAFE_MODEL_NAME}.log 2>&1 &
    fi
    
    echo "Waiting for server to start..."
    sleep 120
    # Wait for server to be ready by checking if it responds to requests
    MAX_RETRIES=30
    RETRY_INTERVAL=20
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Check if the server is responding to health checks
        if curl -s -o /dev/null http://localhost:27182/health; then
            echo "Server is ready!"
            break
        else
            echo "Server not ready yet. Waiting ${RETRY_INTERVAL}s... (${RETRY_COUNT}/${MAX_RETRIES})"
            sleep $RETRY_INTERVAL
            RETRY_COUNT=$((RETRY_COUNT + 1))
        fi
        
        # Check if we've reached max retries
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo "Server failed to start after $((MAX_RETRIES * RETRY_INTERVAL)) seconds. Exiting."
            exit 1
        fi
        
        # Check if the server process is still running
        if ! ps -p $! > /dev/null; then
            echo "Server process has stopped. Check logs for errors."
            exit 1
        fi
    done
fi

if [ "${SERVER}" = "OPENAI" ]; then
    SERVER_FLAG="--server openai --server_batch_size $SERVER_BATCH_SIZE"
fi

python cvbench_analysis.py \
  --model_name $MODEL \
  --num_context_samples $NUM_CONTEXT_SAMPLES \
  --num_contexts $NUM_CONTEXTS \
  --randomize_contexts \
  --bias_type_list no_context,no_bias \
  --test_bias_type_list no_context,always_a,with_marking,always_left,bbox_thickened \
  --scale_factor $SCALE_FACTOR \
  --total_samples $TOTAL_SAMPLES \
  --seed $SEED \
  $REASONING_FLAG \
  $DESCRIPTION_FLAG \
  $SERVER_FLAG

python cvbench_analysis.py \
  --model_name $MODEL \
  --num_context_samples $NUM_CONTEXT_SAMPLES \
  --num_contexts $NUM_CONTEXTS \
  --randomize_contexts \
  --bias_type_list always_a,always_left,with_marking,bbox_thickened \
  --scale_factor $SCALE_FACTOR \
  --total_samples $TOTAL_SAMPLES \
  --seed $SEED \
  $REASONING_FLAG \
  $DESCRIPTION_FLAG \
  $SERVER_FLAG



##### TESTING ####

# echo "python cvbench_analysis.py \
#   --model_name $MODEL \
#   --num_context_samples $NUM_CONTEXT_SAMPLES \
#   --num_contexts $NUM_CONTEXTS \
#   --randomize_contexts \
#   --bias_type_list no_bias \
#   --scale_factor 0.25 \
#   --total_samples 25 \
#   --redo \
#   --seed $SEED \
#   $REASONING_FLAG \
#   $DESCRIPTION_FLAG \
#   $SERVER_FLAG"

# python cvbench_analysis.py \
#   --model_name $MODEL \
#   --num_context_samples $NUM_CONTEXT_SAMPLES \
#   --num_contexts $NUM_CONTEXTS \
#   --randomize_contexts \
#   --bias_type_list no_context \
#   --scale_factor 0.25 \
#   --total_samples 25 \
#   --redo \
#   --seed $SEED \
#   $REASONING_FLAG \
#   $DESCRIPTION_FLAG \
#   $SERVER_FLAG

# python cvbench_analysis.py \
#   --model_name $MODEL \
#   --num_context_samples $NUM_CONTEXT_SAMPLES \
#   --num_contexts $NUM_CONTEXTS \
#   --randomize_contexts \
#   --bias_type_list no_bias \
#   --scale_factor 0.25 \
#   --total_samples 25 \
#   --seed $SEED \
#   --redo \
#   $REASONING_FLAG \
#   $DESCRIPTION_FLAG \
#   $SERVER_FLAG

# exit 0

#####################

