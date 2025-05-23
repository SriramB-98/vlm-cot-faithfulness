# %%
# import os
# os.environ['HOME'] = '/workspace'
import random
from datasets import load_dataset
import numpy as np
# import torchvision
from cvbench_biased_contexts import answer_always_a, answer_with_marking, answer_always_left, bbox_colored, bbox_thickened, grid_combine, no_bias
from cvbench_biased_contexts import switch_order, answers_with_marking, mirror, colored_bbox, thickened_bbox
from model_utils import get_model_tokenizer
import torch
import json
# from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm
import argparse

# Fix all random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# If using PyTorch DataLoader or other components
try:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
except ImportError:
    pass


# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL2_5-26B-MPO")
    parser.add_argument("--bias_type_list", type=str, default='always_a,with_marking,always_left,bbox_colored,bbox_thickened')
    parser.add_argument("--test_bias_type_list", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--num_contexts", type=int, default=1)
    parser.add_argument("--give_reasoning", action='store_true')
    parser.add_argument("--randomize_context_order", action='store_true')
    parser.add_argument("--give_hint", action='store_true')
    parser.add_argument("--presentation_mode", type=str, default="separate")
    args = parser.parse_args()

    model_name = args.model_name
    # model_name = "Qwen/QVQ-72B-Preview"
    bias_type_list = args.bias_type_list.split(',')
    num_samples = args.num_samples
    val_size = args.val_size
    give_reasoning = args.give_reasoning
    give_hint = args.give_hint
    test_bias_type_list = args.test_bias_type_list
    if test_bias_type_list is None:
        test_bias_type_list = [None]
    else:
        test_bias_type_list = test_bias_type_list.split(',')
    presentation_mode = args.presentation_mode
    randomize_context_order = args.randomize_context_order
    num_contexts = args.num_contexts
except:
    print("No arguments provided, using default values")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    bias_type_list = ['bbox_thickened']
    num_samples = 5
    val_size = 20
    give_reasoning = False
    give_hint = False
    arg_test_bias_type = None
    presentation_mode = "grid"
    num_contexts = 1

# %%
cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']
model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)

if 'InternVL2' in model_name:
    predict_kwargs = {
        'max_num' : 4 if presentation_mode == 'separate' else 12,
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

    print(f"Context bias:{bias_type}")

    if biased_context_fn_dict[bias_type] is not None:
        biased_contexts = []
        index_lists = []
        biased_context_gen = biased_context_fn_dict[bias_type](no_bias(cv_bench))
        for nci in range(num_contexts):
            biased_context = []
            index_list = []
            for i, s in biased_context_gen:
                biased_context.append(s)
                index_list.append(i)
                if len(biased_context) >= num_samples:
                    break
            biased_contexts.append(biased_context)
            index_lists.append(index_list)
            print(f"In-context samples for context {nci}:", index_list)
    else:
        biased_contexts = [[]]
        index_lists = [[]]

    for arg_test_bias_type in test_bias_type_list:

        if arg_test_bias_type is None:
            test_bias_type = bias_type
        else:
            test_bias_type = arg_test_bias_type

        print(f"\t Test bias:{test_bias_type}")
        outputs = defaultdict(list)

        num_samples_processed = 0
        total_samples = 100#len(cv_bench[val_size:])
        for i, s in tqdm(no_bias(cv_bench), total=total_samples):

            if give_hint or bias_type == 'ans_in_hint':
                predict_kwargs['hint'] = hint_dict[bias_type].replace('<ANSWER>', s['answer']) 
            else:
                predict_kwargs['hint'] = None

            sb_list = biased_sample_fn_dict[test_bias_type](s)
                    
            for sb, desc in sb_list:
                
                outs = []
                for nci in range(num_contexts):
                    if randomize_context_order:
                        random.shuffle(biased_contexts[nci])

                    if presentation_mode == 'grid':
                        sb = grid_combine(sb, biased_contexts[nci])
                        bc = []
                    elif presentation_mode == 'separate':
                        bc = biased_contexts[nci]
                    else:
                        raise ValueError(f"Invalid presentation mode: {presentation_mode}")

                    out = predict_fn(model, tokenizer, sb, bc, **predict_kwargs)
                    outs.append(out)

                outputs[desc].append(
                    {
                        'pred': outs,
                        'answer': sb['answer'],
                        'id': i
                    }
                )
            num_samples_processed += 1
            if num_samples_processed >= total_samples:
                break
        
        outputs['context_indices'] = index_lists
        fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_test-{test_bias_type}_{"hint" if give_hint or bias_type == "ans_in_hint" else "no_hint"}_{presentation_mode}_{num_samples}samples_{num_contexts}contexts.json'
        with open(fname, 'w') as f:
            json.dump(outputs, f)

# %%