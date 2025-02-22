import json

log_name = "cvbench_OpenGVLab_InternVL2_5-8B-MPO_always_a_4samples.json"

with open(f"results/{log_name}", "r") as f:
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
            if d['pred'].startswith('(A)') or d['pred'].startswith('A'):
                a_a += 1
            elif d['pred'].startswith('(B)') or d['pred'].startswith('B'):
                a_b += 1
            else:
                a_unk += 1
        elif d['answer'] == '(B)':
            if d['pred'].startswith('(B)') or d['pred'].startswith('B'):
                b_b += 1
            elif d['pred'].startswith('(A)') or d['pred'].startswith('A'):
                b_a += 1
            else:
                b_unk += 1
    
    print(f"G\\P \t A \t B \t Unk")
    print(f"A \t {a_a} \t {a_b} \t {a_unk}")
    print(f"B \t {b_a} \t {b_b} \t {b_unk}")
    print()