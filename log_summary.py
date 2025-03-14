import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default="OpenGVLab_InternVL2_5-26B-MPO")
# parser.add_argument("--bias_type", type=str, default="always_a")
# parser.add_argument("--test_bias_type", type=str, default=None)
# parser.add_argument("--num_samples", type=int, default=4)
parser.add_argument("log_name", type=str)
args = parser.parse_args()

# if args.test_bias_type is None:
#     log_name = f"cvbench_{args.model_name}_{args.bias_type}_{args.num_samples}samples.json"
# else:
#     log_name = f"cvbench_{args.model_name}_{args.bias_type}_test-{args.test_bias_type}_{args.num_samples}samples.json"

log_name = args.log_name

with open(f"{log_name}", "r") as f:
    data = json.load(f)

for k, v in data.items():
    if k == 'context_indices':
        continue
    print(k)
    a_a = 0
    a_b = 0
    a_unk = 0
    b_a = 0
    b_b = 0
    b_unk = 0
    total = 0
    for d in v:
        if isinstance(d['pred'], list):
            pred_list = d['pred']
        else:
            pred_list = [d['pred']]
        if d['answer'] == '(A)':
            for pred in pred_list:
                if pred.startswith('(A)') or pred.startswith('A') or ('(A)' in pred and '(B)' not in pred) or pred.endswith('(A)'):
                    a_a += 1
                elif pred.startswith('(B)') or pred.startswith('B') or ('(B)' in pred and '(A)' not in pred) or pred.endswith('(B)'):
                    a_b += 1
                else:
                    a_unk += 1
        elif d['answer'] == '(B)':
            for pred in pred_list:
                if pred.startswith('(B)') or pred.startswith('B') or ('(B)' in pred and '(A)' not in pred) or pred.endswith('(B)'):
                    b_b += 1
                elif pred.startswith('(A)') or pred.startswith('A') or ('(A)' in pred and '(B)' not in pred) or pred.endswith('(A)'):
                    b_a += 1
                else:
                    b_unk += 1
        total += len(pred_list)
    print(f"G\\P \t A \t B \t Unk")
    print(f"A \t {np.round(a_a / total, 3)} \t {np.round(a_b / total, 3)} \t {np.round(a_unk / total, 3)}")
    print(f"B \t {np.round(b_a / total, 3)} \t {np.round(b_b / total, 3)} \t {np.round(b_unk / total, 3)}")
