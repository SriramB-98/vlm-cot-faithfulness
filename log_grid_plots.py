# %%
import json
import shutil
from matplotlib import pyplot as plt
import argparse
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from extract_answer_utils import extract_from_llava_cot, extract_from_qvq, extract_from_vlm_r1

# %%

try:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="OpenGVLab_InternVL2_5-26B-MPO")
    # parser.add_argument("--bias_type", type=str, default="always_a")
    # parser.add_argument("--test_bias_type", type=str, default=None)
    # parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("log_name_template", type=str)
    args = parser.parse_args()

    log_name_templates = args.log_name_template.split(",")
except:
    # log_name_templates = ["results/cvbench_Qwen_Qwen2.5-VL-7B-Instruct_MODELBIAS_test-TESTBIAS_no_hint_4samples.json",
    #   "results/cvbench_Qwen_Qwen2.5-VL-7B-Instruct_MODELBIAS_test-TESTBIAS_no_hint_separate_4samples.json"]
    # log_name_templates = ["results/cvbench_Qwen_Qwen2.5-VL-7B-Instruct_MODELBIAS_test-TESTBIAS_no_hint_separate_4samples_1contexts.json"]
    model_name = "OpenGVLab_InternVL2_5-78B-MPO-AWQ"  # 'Qwen_Qwen2.5-VL-7B-Instruct'
    reasoning = True
    log_name_templates = [
        f"results/cvbench_{model_name}_MODELBIAS_test-TESTBIAS_no_hint_{'reasoning_' if reasoning else ''}separate_8samples_100testsamples_1contexts_scale-0.25.json"
    ]
    copy_files = False
    
# %%

for model_name in [
    "OpenGVLab_InternVL2_5-78B-MPO-AWQ",
    "OpenGVLab_InternVL2_5-8B-MPO",
    "Qwen_Qwen2.5-VL-7B-Instruct",
    "Qwen_Qwen2.5-VL-72B-Instruct-AWQ",
    "Xkev_Llama-3.2V-11B-cot",
    "omlab_Qwen2.5VL-3B-VLM-R1-REC-500steps",
    "Qwen_QVQ-72B-Preview",
    "meta-llama_Llama-3.2-11B-Vision-Instruct",
    "Qwen_Qwen2.5-VL-3B-Instruct",
]:
    reasoning = True
    if model_name == "Xkev_Llama-3.2V-11B-cot" or model_name == "meta-llama_Llama-3.2-11B-Vision-Instruct":
        mode = "grid"
    else:
        mode = "separate"
    log_name_templates = [
        f"results_no_context/cvbench_{model_name}_MODELBIAS_test-TESTBIAS_no_hint_{'reasoning_' if reasoning else ''}{mode}_8samples_100testsamples_1contexts_scale-0.25{'_short_reason' if model_name == "Qwen_QVQ-72B-Preview" else ''}.json"
    ]
    # Extract the directory and filename pattern from the template
    matching_files = []
    model_biases = []
    for log_name_template in log_name_templates:
        directory = os.path.dirname(log_name_template)
        filename_pattern = os.path.basename(log_name_template)

        # Replace placeholders with wildcards for glob pattern
        glob_pattern = os.path.join(directory, filename_pattern.replace("MODELBIAS", "*").replace("TESTBIAS", "*"))

        # Find all matching files
        matched_files = glob.glob(glob_pattern)
        matching_files.extend(matched_files)

        # Extract MODELBIAS values
        for file_path in matched_files:
            filename = os.path.basename(file_path)
            # Extract the part that replaced MODELBIAS in the template
            # Find the position of MODELBIAS in the template
            template_parts = filename_pattern.split("_")
            modelbias_index = -1
            for i, part in enumerate(template_parts):
                if "MODELBIAS" in part:
                    modelbias_index = i
                    break

            if modelbias_index != -1:
                # Split the actual filename and extract the corresponding part
                parts = filename.split("_")
                # Extract the model bias, which may contain underscores
                test_index = -1
                for i, part in enumerate(parts):
                    if "test-" in part:
                        test_index = i
                        break

                if test_index != -1 and modelbias_index < test_index:
                    # The model bias is everything between modelbias_index and test_index
                    model_bias = "_".join(parts[modelbias_index:test_index])
                    model_biases.append(model_bias)

    print(f"Found {len(matching_files)} matching files:")
    for file_path in matching_files:
        print(f"  - {file_path}")

    print(f"\nFound {len(model_biases)} model bias types:")
    for bias in sorted(model_biases):
        print(f"  - {bias}")
    
    if copy_files:
        for file_path in matching_files:
            # copy to results
            new_file_path = file_path.replace("old_results/", "results/")
            # Create directory if it doesn't exist
            # os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            # Copy the file
            shutil.copy2(file_path, new_file_path)
            print(f"  - Copied to {new_file_path}")

    # %%
    all_data = {}
    for file_path, model_bias in zip(matching_files, model_biases):
        with open(file_path, "r") as f:
            data = json.load(f)

        if model_bias not in all_data:
            all_data[model_bias] = dict()

        for k, v in data.items():
            if k == "context_indices":
                continue
            if k in all_data[model_bias]:
                raise ValueError(f"Duplicate key {k} for model bias {model_bias}")
            all_data[model_bias][k] = v

    # %%

    answer_options = ["(A)", "(B)"]
    acc_dict = {}
    for model_bias, data in all_data.items():
        acc_dict[model_bias] = {}
        for k, v in data.items():
            if k == "context_indices":
                continue
            hits, total = 0, 0
            unk = 0
            for d in v:
                ans_ind = answer_options.index(d["answer"])
                ans = d["answer"]
                if isinstance(d["pred"], list):
                    pred_list = d["pred"]
                else:
                    pred_list = [d["pred"]]

                for pred in pred_list:
                    if model_name == "Xkev_Llama-3.2V-11B-cot":
                        pred = extract_from_llava_cot(pred)
                    elif model_name == "Qwen_QVQ-72B-Preview":
                        pred = extract_from_qvq(pred)
                    elif model_name == "omlab_Qwen2.5VL-3B-VLM-R1-REC-500steps":
                        pred = extract_from_vlm_r1(pred)

                    if pred == "":
                        unk += 1
                    elif (
                        pred.startswith(ans)
                        or pred.startswith(ans[1])
                        or (ans in pred and answer_options[1 - ans_ind] not in pred)
                        or pred.endswith(ans)
                    ):
                        hits += 1

                    total += 1

            if unk > 0:
                print(f"Unknown rate for {model_name} {model_bias} {k}: {unk / total}")

            acc_dict[model_bias][k] = hits / total

    # %%
    # Get all unique test settings across all model biases
    all_test_settings = set()
    for model_bias, test_dict in acc_dict.items():
        all_test_settings.update(test_dict.keys())

    test_settings_order = [
        "unchanged",
        "answer (A)",
        "answer (B)",
        "correctly marked",
        "incorrectly marked",
        "answer on left",
        "answer on right",
        "correctly thickened",
        "incorrectly thickened",
        "correctly colored",
        "incorrectly colored",
    ]
    all_test_settings = [ts for ts in test_settings_order if ts in all_test_settings]

    # Create a matrix to hold the accuracy values
    model_biases_order = [
        "bbox_colored",
        "bbox_thickened",
        "always_left",
        "with_marking",
        "always_a",
        "no_bias",
        "no_context",
    ]
    model_biases = [mb for mb in model_biases_order if mb in model_biases]
    matrix = np.zeros((len(model_biases), len(all_test_settings)))
    mask = np.ones((len(model_biases), len(all_test_settings)), dtype=bool)

    # Fill the matrix with accuracy values
    for i, model_bias in enumerate(model_biases):
        for j, test_setting in enumerate(all_test_settings):
            if test_setting in acc_dict[model_bias]:
                matrix[i, j] = acc_dict[model_bias][test_setting]
                mask[i, j] = False

    # Create a custom colormap: blue (0) -> white (0.5) -> red (1)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors, N=100)

    # Create the plot
    plt.figure(figsize=(max(12, len(all_test_settings) * 0.8), max(8, len(model_biases) * 0.6)))

    # Plot the heatmap
    ax = sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        annot=True,
        fmt=".2f",
        xticklabels=all_test_settings,
        yticklabels=model_biases,
        cbar_kws={"label": "Accuracy"},
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Add labels and title
    plt.title("Accuracy by Model Bias and Test Setting")
    plt.tight_layout()

    # Show the plot
    plt.savefig(
        f"figures/cvbench_{model_name}_no_hint_{'reasoning_' if reasoning else ''}{mode}_8samples_100testsamples_1contexts_scale-0.25.png"
    )

    # %%


