import json
import argparse

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
    print(k)
    a_a = 0
    a_b = 0
    a_unk = 0
    b_a = 0
    b_b = 0
    b_unk = 0
    for d in v:
        if d['answer'] == '(A)':
            if d['pred'].startswith('(A)') or d['pred'].startswith('A') or ('(A)' in d['pred'] and 'B' not in d['pred']):
                a_a += 1
            elif d['pred'].startswith('(B)') or d['pred'].startswith('B') or ('(B)' in d['pred'] and 'A' not in d['pred']):
                a_b += 1
            else:
                a_unk += 1
        elif d['answer'] == '(B)':
            if d['pred'].startswith('(B)') or d['pred'].startswith('B') or ('(B)' in d['pred'] and 'A' not in d['pred']):
                b_b += 1
            elif d['pred'].startswith('(A)') or d['pred'].startswith('A') or ('(A)' in d['pred'] and 'B' not in d['pred']):
                b_a += 1
            else:
                b_unk += 1
    
    print(f"G\\P \t A \t B \t Unk")
    print(f"A \t {a_a} \t {a_b} \t {a_unk}")
    print(f"B \t {b_a} \t {b_b} \t {b_unk}")
    print()