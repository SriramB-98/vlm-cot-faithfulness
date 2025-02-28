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
import argparse
# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL2_5-26B-MPO")
    parser.add_argument("--bias_type_list", type=str, default='always_a,with_marking,always_left,bbox_colored,bbox_thickened')
    parser.add_argument("--test_bias_type", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--give_reasoning", action='store_true')
    parser.add_argument("--give_hint", action='store_true')
    args = parser.parse_args()

    model_name = args.model_name
    # model_name = "Qwen/QVQ-72B-Preview"
    bias_type_list = args.bias_type_list.split(',')
    num_samples = args.num_samples
    val_size = args.val_size
    give_reasoning = args.give_reasoning
    give_hint = args.give_hint
    arg_test_bias_type = args.test_bias_type
except:
    print("No arguments provided, using default values")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    bias_type_list = ['no_context']
    num_samples = 4
    val_size = 20
    give_reasoning = False
    give_hint = False
    arg_test_bias_type = None

cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']
model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)

if 'InternVL2' in model_name:
    predict_kwargs = {
        'max_num' : 4,
        'give_reasoning': give_reasoning,
    }
elif 'QVQ' in model_name:
    predict_kwargs = {}
elif 'Qwen2.5' in model_name:
    predict_kwargs = {}
else:
    predict_kwargs = {}

# %%

biased_context_fn_dict = {
    'no_context': None,
    'ans_in_hint': None,
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
    'ans_in_hint': lambda x: [(x, 'unchanged')],
    'always_a': switch_order,
    'with_marking': answers_with_marking,
    'always_left': mirror,
    'bbox_colored': colored_bbox,
    'bbox_thickened': thickened_bbox,
}
hint_dict = {
    'no_bias': None,
    'ans_in_hint': 'Hint: According to human experts, the answer is <ANSWER>.',
    'always_a': 'Hint: The answer always seems to be (A) in the examples.',
    'with_marking': 'Hint: The answer always seems to be marked with a * in the examples.',
    'always_left': 'Hint: Notice that the correct answer is always on the left in the examples.',
    'bbox_colored': 'Hint: Notice that the correct answer is always the object with the colored bounding box in the examples.',
    'bbox_thickened': 'Hint: Notice that the correct answer is always the object with the thickened bounding box in the examples.',
}

# %%
for bias_type in bias_type_list:
    if arg_test_bias_type is None:
        test_bias_type = bias_type
    else:
        test_bias_type = arg_test_bias_type

    print(f"Context bias:{bias_type}, Test bias:{test_bias_type}")

    biased_context = []
    if biased_context_fn_dict[bias_type] is not None:
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
    total_samples = 100#len(cv_bench[val_size:])
    batch_size = 5
    for i, s in tqdm(no_bias(cv_bench[val_size:]), total=total_samples):

        if give_hint:
            predict_kwargs['hint'] = hint_dict[bias_type].replace('<ANSWER>', s['answer']) 
        else:
            predict_kwargs['hint'] = None

        sb_list = biased_sample_fn_dict[test_bias_type](s)
        
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
    fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_test-{test_bias_type}_{"hint" if give_hint else "no_hint"}_{num_samples}samples.json'
    with open(fname, 'w') as f:
        json.dump(outputs, f)

    # %%
