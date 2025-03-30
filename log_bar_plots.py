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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from extract_answer_utils import extract_from_default, extract_from_llava_cot, extract_from_qvq, extract_from_vlm_r1
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_test(x, y):
    """
    Run McNemar's test on paired binary vectors x and y
    
    Parameters:
    x, y: numpy arrays of 0s and 1s with the same length
    
    Returns:
    test statistic and p-value
    """
    x = np.array(x)
    y = np.array(y)
    # Create the contingency table
    a = sum((x == 0) & (y == 0))
    b = sum((x == 0) & (y == 1))
    c = sum((x == 1) & (y == 0))
    d = sum((x == 1) & (y == 1))
    
    contingency_table = np.array([[a, b], [c, d]])
    
    # Run McNemar's test
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    return result.statistic, result.pvalue
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
    MODEL_BIAS = 'no_context'
    if model_name == "Xkev_Llama-3.2V-11B-cot" or model_name == "meta-llama_Llama-3.2-11B-Vision-Instruct":
        mode = "grid"
    else:
        mode = "separate"
    log_name_templates = [
        f"results/cvbench_{model_name}_MODELBIAS_test-TESTBIAS_no_hint_{'reasoning_' if reasoning else ''}{mode}_8samples_100testsamples_1contexts_scale-0.25{'_short_reason' if model_name == 'Qwen_QVQ-72B-Preview' else ''}_forced.json"
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

    # print(f"\nFound {len(model_biases)} model bias types:")
    # for bias in sorted(model_biases):
    #     print(f"  - {bias}")
    
    if copy_files:
        for file_path in matching_files:
            # copy to results
            new_file_path = file_path.replace("old_results/", "results/")
            # Create directory if it doesn't exist
            # os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            # Copy the file
            shutil.copy2(file_path, new_file_path)
            print(f"  - Copied to {new_file_path}")
        quit()

    # %%
    all_data = {}
    for file_path, model_bias in zip(matching_files, model_biases):
        with open(file_path, "r") as f:
            data = json.load(f)

        if model_bias != MODEL_BIAS:
            continue

        for k, v in data.items():
            if k == "context_indices":
                continue
            if k in all_data:
                raise ValueError(f"Duplicate key {k} for model bias {model_bias}")
            all_data[k] = v

    # %%

    answer_options = ["(A)", "(B)"]
    acc_dict = {}
    hitlist_dict = {}
    for k, v in all_data.items():
        if k == "context_indices":
            continue
        hits, total = 0, 0
        hit_list = []
        unk = 0
        for d in v:
            ans_ind = answer_options.index(d["answer"])
            ans = d["answer"]
            other_ans = answer_options[1-ans_ind]
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
                    
                pred = extract_from_default(pred)

                if ans[1] in pred and other_ans[1] not in pred:
                    hit_list.append(1)
                    hits += 1
                elif not (('A' in pred) ^ ('B' in pred)):
                    unk += 1
                    hit_list.append(0)
                else:
                    hit_list.append(0)
                
                total += 1

        hitlist_dict[k] = hit_list
        if unk > 0:
            print(f"Unknown rate for {model_name} {model_bias} {k}: {unk / total}")

        acc_dict[k] = hits / total

    # %%
    test_settings_order = [
        "unchanged",
        "correctly hinted",
        "incorrectly hinted",
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

    setting_groups = [
        ["unchanged", "----"],
        ["correctly hinted", "incorrectly hinted"],
        ["answer (A)", "answer (B)"],
        ["correctly marked", "incorrectly marked"],
        ["answer on left", "answer on right"],
        ["correctly thickened", "incorrectly thickened"],
        ["correctly colored", "incorrectly colored"],
    ]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define colors for each group
    colors = ListedColormap(plt.cm.tab20.colors)

    # Group the test settings and accuracies
    grouped_test_settings = []
    grouped_accuracies = []
    grouped_p_values = []
    for group in setting_groups:
        if all([setting in hitlist_dict for setting in group]):
            statistic, p_val = run_mcnemar_test(hitlist_dict[group[0]], hitlist_dict[group[1]])
            grouped_p_values.append(p_val)
        else:
            grouped_p_values.append(None)

        for setting in group:
            if setting in acc_dict:
                grouped_test_settings.append(setting)
                grouped_accuracies.append(acc_dict[setting])

    # Calculate the width of each bar
    bar_width = 0.35

    # Create bar positions with smaller spacing between bars in a group
    group_spacing = 0.5
    bar_spacing = 0.05
    bar_positions = []
    current_position = 0
    for group in setting_groups:
        for setting in group:
            if setting in acc_dict:
                bar_positions.append(current_position)
                current_position += bar_width + bar_spacing
        current_position += group_spacing - bar_spacing  # Add extra space between groups

    # Plot bars with colors for each group and add a thin black border
    for i, group in enumerate(setting_groups):
        group_positions = [bar_positions[j] for j, setting in enumerate(grouped_test_settings) if setting in group]
        group_accuracies = [grouped_accuracies[j] for j, setting in enumerate(grouped_test_settings) if setting in group]
        
        # Plot the bars with the appropriate colors
        ax.bar(group_positions, group_accuracies, bar_width, color=colors(i), edgecolor='black')
        
        # If p-value is significant, add p-value text and highlight the x-axis labels
        if grouped_p_values[i] is not None and grouped_p_values[i] < 0.05:
            # Calculate the position for the p-value text
            p_val_position = sum(group_positions) / len(group_positions)
            # Display the p-value at the top of the bars
            ax.text(p_val_position, max(group_accuracies) + 0.02, f'p={grouped_p_values[i]:.3f}', 
                   ha='center', va='bottom', fontsize=8, color='black')

    # Add a dotted line at 0.5 for the random baseline
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)

    # Set labels and title
    ax.set_xlabel('Test Settings')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy by Test Setting')

    # Set x-ticks and labels
    ax.set_xticks(bar_positions)
    
    # Create tick labels with bolding for significant p-values
    tick_labels = []
    for i, setting in enumerate(grouped_test_settings):
        # Find which group this setting belongs to
        group_idx = next((j for j, group in enumerate(setting_groups) if setting in group), None)
        
        # Check if the p-value for this group is significant
        is_significant = group_idx is not None and grouped_p_values[group_idx] is not None and grouped_p_values[group_idx] < 0.05
        
        # Apply yellow highlighting for significant p-values
        if is_significant:
            tick_labels.append(f"$\\bf{{{setting}}}$")  # Bold text for significant settings
        else:
            tick_labels.append(setting)
    
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Adjust layout to make room for the rotated x-axis labels
    plt.tight_layout()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to make room for the rotated x-axis labels
    plt.tight_layout()
    

    # Show the plot
    plt.savefig(
        f"figures/cvbench_{model_name}_{MODEL_BIAS}_no_hint_{'reasoning_' if reasoning else ''}{mode}_8samples_100testsamples_1contexts_scale-0.25_barplot.png"
    )

    # %%


