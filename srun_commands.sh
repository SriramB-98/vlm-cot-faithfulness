srun --qos=cml-default --partition=cml-dpart --account=cml-sfeizi --time=00:30:00 --gres=gpu:rtxa4000:1 --cpus-per-task=4 --mem=32gb bash -c 'env MODEL="Qwen/Qwen2.5-VL-3B-Instruct" SERVER="NONE" ./run_all_settings.sh' > run_all_settings_20250328_1.out 2> run_all_settings_20250328_1.err &

srun --qos=cml-medium --partition=cml-dpart --account=cml-sfeizi --time=01:00:00 --gres=gpu:rtxa4000:2 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_5.out 2> run_all_settings_20250328_5.err &

srun --qos=cml-medium --partition=cml-dpart --account=cml-sfeizi --time=00:30:00 --gres=gpu:rtxa4000:2 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="Qwen/Qwen2.5-VL-7B-Instruct" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_2.out 2> run_all_settings_20250328_2.err &

srun --qos=cml-medium --partition=cml-dpart --account=cml-sfeizi --time=01:00:00 --gres=gpu:rtxa4000:2 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="OpenGVLab/InternVL2_5-8B-MPO" SERVER="NONE" bash run_all_settings.sh' > run_all_settings_20250328_5.out 2> run_all_settings_20250328_5.err &

srun --qos=cml-high --partition=cml-dpart --account=cml-sfeizi --time=01:00:00 --gres=gpu:rtxa4000:4 --cpus-per-task=4 --mem=128gb bash -c 'env MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_4.out 2> run_all_settings_20250328_4.err &

srun --qos=cml-high --partition=cml-dpart --account=cml-sfeizi --time=01:00:00 --gres=gpu:rtxa4000:4 --cpus-per-task=4 --mem=128gb bash -c 'env MODEL="Xkev/Llama-3.2V-11B-cot" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_3.out 2> run_all_settings_20250328_3.err &


srun --qos=cml-scavenger --partition=cml-scavenger --account=cml-scavenger --time=01:00:00 --gres=gpu:rtxa6000:2 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="Qwen/Qwen2.5-VL-72B-Instruct-AWQ" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_2.out 2> run_all_settings_20250328_2.err &

srun --qos=cml-scavenger --partition=cml-scavenger --account=cml-scavenger --time=01:00:00 --gres=gpu:rtxa6000:2 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="OpenGVLab/InternVL2_5-78B-MPO-AWQ" SERVER="NONE" bash run_all_settings.sh' > run_all_settings_20250328_5.out 2> run_all_settings_20250328_5.err &


srun --qos=cml-scavenger --partition=cml-scavenger --account=cml-scavenger --time=01:00:00 --gres=gpu:rtxa6000:4 --cpus-per-task=4 --mem=64gb bash -c 'env MODEL="Qwen/QVQ-72B-Preview" SERVER="VLLM" bash run_all_settings.sh' > run_all_settings_20250328_5.out 2> run_all_settings_20250328_5.err &


