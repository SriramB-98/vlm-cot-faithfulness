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
# time: ~1:00:00 for 7B (100 samples)

hostname

# Base parameters
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
NUM_CONTEXT_SAMPLES=8
NUM_CONTEXTS=1
SCALE_FACTOR=0.25
TOTAL_SAMPLES=100
SEED=20
GIVE_REASONING=1
REASONING_FLAG=""
if [ "${GIVE_REASONING}" = "1" ]; then
    REASONING_FLAG="--give_reasoning"
fi

python cvbench_analysis.py \
  --model_name $MODEL \
  --num_context_samples $NUM_CONTEXT_SAMPLES \
  --num_contexts $NUM_CONTEXTS \
  --randomize_contexts \
  --bias_type_list no_context,no_bias \
  --test_bias_type_list no_context,always_a,with_marking,always_left,bbox_thickened \
  --scale_factor $SCALE_FACTOR \
  --description _scale-${SCALE_FACTOR} \
  --total_samples $TOTAL_SAMPLES \
  --seed $SEED \
  $REASONING_FLAG

python cvbench_analysis.py \
  --model_name $MODEL \
  --num_context_samples $NUM_CONTEXT_SAMPLES \
  --num_contexts $NUM_CONTEXTS \
  --randomize_contexts \
  --bias_type_list always_a,always_left,with_marking,bbox_thickened \
  --scale_factor $SCALE_FACTOR \
  --description _scale-${SCALE_FACTOR} \
  --total_samples $TOTAL_SAMPLES \
  --seed $SEED \
  $REASONING_FLAG
