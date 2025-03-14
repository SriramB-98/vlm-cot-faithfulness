# %%
import json
import argparse
from spur_datasets import load_dataset
from celebamask_hq import CelebAMaskHQ
import numpy as np
import matplotlib.pyplot as plt


try:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="OpenGVLab_InternVL2_5-26B-MPO")
    # parser.add_argument("--bias_type", type=str, default="always_a")
    # parser.add_argument("--test_bias_type", type=str, default=None)
    # parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("log_name", type=str)
    args = parser.parse_args()
except:
    log_name = "results/spur_Qwen_Qwen2.5-VL-7B-Instruct_celeba_blond_reasoning_860samples.json"

if 'celeba' in log_name:
    dataset = load_dataset('celeba_blond')
elif 'waterbirds' in log_name:
    dataset = load_dataset('waterbirds')
else:
    raise ValueError(f"Dataset not found: {log_name}")

with open(f"./{log_name}", "r") as f:
    data = json.load(f)

# %%

indices_dict = {}
output_dict = {}
for k, v in data.items():
    print(k)
    a_a = 0
    a_b = 0
    a_unk = 0
    b_a = 0
    b_b = 0
    b_unk = 0
    output_dict[k] = {('(A)', '(A)'): [], ('(A)', '(B)'): [], ('(A)', '(UNK)'): [], ('(B)', '(A)'): [], ('(B)', '(B)'): [], ('(B)', '(UNK)'): []}
    indices_dict[k] = {('(A)', '(A)'): [], ('(A)', '(B)'): [], ('(A)', '(UNK)'): [], ('(B)', '(A)'): [], ('(B)', '(B)'): [], ('(B)', '(UNK)'): []}
    for d in v:
        if d['answer'] == '(A)':
            if d['pred'].startswith('(A)') or d['pred'].startswith('A') or ('(A)' in d['pred'] and '(B)' not in d['pred']) or d['pred'].endswith('(A)'):
                output_dict[k][('(A)', '(A)')].append(d['pred'])
                indices_dict[k][('(A)', '(A)')].append(d['id'])
                a_a += 1
            elif d['pred'].startswith('(B)') or d['pred'].startswith('B') or ('(B)' in d['pred'] and '(A)' not in d['pred']) or d['pred'].endswith('(B)'):
                output_dict[k][('(A)', '(B)')].append(d['pred'])
                indices_dict[k][('(A)', '(B)')].append(d['id'])
                a_b += 1
            else:
                output_dict[k][('(A)', '(UNK)')].append(d['pred'])
                indices_dict[k][('(A)', '(UNK)')].append(d['id'])
                a_unk += 1
        elif d['answer'] == '(B)':
            if d['pred'].startswith('(B)') or d['pred'].startswith('B') or ('(B)' in d['pred'] and '(A)' not in d['pred']) or d['pred'].endswith('(B)'):
                output_dict[k][('(B)', '(B)')].append(d['pred'])
                indices_dict[k][('(B)', '(B)')].append(d['id'])
                b_b += 1
            elif d['pred'].startswith('(A)') or d['pred'].startswith('A') or ('(A)' in d['pred'] and '(B)' not in d['pred']) or d['pred'].endswith('(A)'):
                output_dict[k][('(B)', '(A)')].append(d['pred'])
                indices_dict[k][('(B)', '(A)')].append(d['id'])
                b_a += 1
            else:
                output_dict[k][('(B)', '(UNK)')].append(d['pred'])
                indices_dict[k][('(B)', '(UNK)')].append(d['id'])
                b_unk += 1
    
    print(f"G\\P \t A \t B \t Unk")
    print(f"A \t {a_a} \t {a_b} \t {a_unk}")
    print(f"B \t {b_a} \t {b_b} \t {b_unk}")
    print()


# %%
answer_options = ['(A)', '(B)']
mask_dataset = CelebAMaskHQ(root='/cmlscratch/sriramb/CelebAMask-HQ', split='ALL', label_type='all', classify='Blond_Hair', confounder='Male')
hair_ind = mask_dataset.label_setting['suffix'].index('hair')
num_samples = 50
hair_color_dict = {}

for true_answer in answer_options:
    for pred_answer in answer_options:
        for confounder in ['male', 'not male']:
            indices = indices_dict[confounder][(true_answer, pred_answer)]
            outputs = output_dict[confounder][(true_answer, pred_answer)]
            print(f'{confounder} {true_answer} {pred_answer} {len(indices)}')
            hair_pixels_mom = np.zeros((3,))
            total_samples = 0
            for ind in indices[:num_samples]:
                img, mask, attributes = mask_dataset[ind]
                img_array = np.array(img.resize((512, 512)))
                hair_mask = (mask == hair_ind + 1)
                if hair_mask.sum() > 0:
                    hair_pixels = img_array[hair_mask]
                    hair_pixels_mean = hair_pixels.mean(axis=0)
                    hair_pixels_mom += hair_pixels_mean
                    total_samples += 1
                    
            hair_pixels_mom /= total_samples
            hair_color_dict[(true_answer, pred_answer, confounder)] = hair_pixels_mom
# %%
def display_color_square(rgb_values):
    """
    Display a square of the given RGB color.

    Parameters:
    rgb_values (tuple): A tuple containing the RGB values (R, G, B).

    Returns:
    None
    """
    import numpy as np

    # Create an array of the given color
    color_square = np.zeros((10, 10, 3), dtype=np.uint8)
    color_square[:, :] = rgb_values

    # Display the color square
    plt.imshow(color_square)
    plt.axis('off')
    plt.show()

# %%
