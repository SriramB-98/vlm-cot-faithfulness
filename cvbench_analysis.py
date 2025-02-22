# %%
# import os
# os.environ['HOME'] = '/workspace'
from datasets import load_dataset
# import torchvision
from cvbench_biased_contexts import answer_always_a, answer_with_marking, answer_always_left, bbox_colored, bbox_thickened, no_bias
from cvbench_biased_contexts import switch_order, answers_with_marking, mirror, colored_bbox, thickened_bbox
from model_utils import get_model_tokenizer
import torch
import json
# from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm
# %%

model_name = "OpenGVLab/InternVL2_5-8B-MPO"
# model_name = "Qwen/QVQ-72B-Preview"
bias_type = 'always_a'
num_samples = 4
val_size = 20

cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']
model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)


if 'InternVL2' in model_name:
    predict_kwargs = {
        'max_num' : 4
    }
else:
    predict_kwargs = {}

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
    'no_context': lambda x: [(x, 'unchanged')],
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
    print("In-context samples:", index_list)

# %%
outputs = defaultdict(list)

num_samples_processed = 0
total_samples = 20#len(cv_bench[val_size:])
batch_size = 5
for i, s in tqdm(no_bias(cv_bench[val_size:]), total=total_samples):

    sb_list = biased_sample_fn_dict[bias_type](s)
    for sb, desc in sb_list:
        out = predict_fn(model, tokenizer, sb, biased_context, **predict_kwargs)
        outputs[desc].append(
            {
                'pred': out,
                'answer': sb['answer'],
                'id': i
            }
        )
    num_samples_processed += 1
    if num_samples_processed >= total_samples:
        break

# %%
fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_{num_samples}samples.json'
with open(fname, 'w') as f:
    json.dump(outputs, f)

# %%
