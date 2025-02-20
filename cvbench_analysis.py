# %%

from datasets import load_dataset
# import torchvision
from cvbench_biased_contexts import answer_always_a, answer_with_marking, answer_always_left, bbox_colored, bbox_thickened, no_bias
from cvbench_biased_contexts import switch_order, answers_with_marking, mirror, colored_bbox, thickened_bbox
from model_utils import get_model_tokenizer
import torch
import json
# from PIL import Image
from collections import defaultdict
# %%

model_name = "OpenGVLab/InternVL2_5-1B"
bias_type = 'no_bias'
num_samples = 4
val_size = 20

model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)
cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']

# %%

biased_context_fn_dict = {
    'no_bias': lambda x: x,
    'always_a': answer_always_a,
    'with_marking': answer_with_marking,
    'always_left': answer_always_left,
    'bbox_colored': bbox_colored,
    'bbox_thickened': bbox_thickened,
}
biased_sample_fn_dict = {
    'no_bias': lambda x: [(x, 'unchanged')],
    'always_a': switch_order,
    'with_marking': answers_with_marking,
    'always_left': mirror,
    'bbox_colored': colored_bbox,
    'bbox_thickened': thickened_bbox,
}

biased_context = []

if bias_type != 'no_context':
    biased_context_gen = biased_context_fn_dict[bias_type](no_bias(cv_bench[:val_size]))
    index_list = []
    for i, s in biased_context_gen:
        biased_context.append(s)
        index_list.append(i)
        if len(biased_context) >= num_samples:
            break
    

# %%
outputs = defaultdict(list)

for i, s in no_bias(cv_bench):
    if i in index_list:
        continue

    sb_list = biased_sample_fn_dict[bias_type](s)
    for sb, desc in sb_list:
        out = predict_fn(model, tokenizer, sb, biased_context)
        outputs[desc].append(
            {
                'pred': out,
                'answer': s['answer'],
                'id': i
            }
        )

# %%
fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_{num_samples}samples.json'
with open(fname, 'w') as f:
    json.dump(outputs, f)

# %%
