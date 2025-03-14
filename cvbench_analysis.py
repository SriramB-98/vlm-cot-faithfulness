# %%
# import os
# os.environ['HOME'] = '/workspace'
import os
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


# %%
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL2_5-26B-MPO")
    parser.add_argument("--bias_type_list", type=str, default='always_a,with_marking,always_left,bbox_colored,bbox_thickened')
    parser.add_argument("--test_bias_type_list", type=str, default=None)
    parser.add_argument("--num_context_samples", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=20)
    parser.add_argument("--num_contexts", type=int, default=1)
    parser.add_argument("--give_reasoning", action='store_true')
    parser.add_argument("--randomize_contexts", action='store_true')
    parser.add_argument("--give_hint", action='store_true')
    parser.add_argument("--presentation_mode", type=str, default="separate")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--remove_explicit_question", action='store_true')
    parser.add_argument("--redo", action='store_true')
    parser.add_argument("--wrong_examples", type=int, default=0)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--total_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = args.model_name
    bias_type_list = args.bias_type_list.split(',')
    num_context_samples = args.num_context_samples
    val_size = args.val_size
    give_reasoning = args.give_reasoning
    give_hint = args.give_hint
    test_bias_type_list = args.test_bias_type_list
    if test_bias_type_list is None:
        test_bias_type_list = [None]
    else:
        test_bias_type_list = test_bias_type_list.split(',')
    presentation_mode = args.presentation_mode
    randomize_contexts = args.randomize_contexts
    # randomize_context_order = args.randomize_context_order
    scale_factor = args.scale_factor
    num_contexts = args.num_contexts
    description = args.description
    seed = args.seed
    total_samples = args.total_samples
    wrong_examples = args.wrong_examples
    remove_explicit_question = args.remove_explicit_question
    redo = args.redo
except:
    print("No arguments provided, using default values")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    bias_type_list = ['bbox_thickened']
    num_context_samples = 5
    val_size = 20
    give_reasoning = False
    give_hint = False
    arg_test_bias_type = None
    presentation_mode = "grid"
    num_contexts = 1
    randomize_contexts = False
    seed = 42
    total_samples = 100
    wrong_examples = 0
    remove_explicit_question = False
# Fix all random seeds
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If using PyTorch DataLoader or other components
try:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
except ImportError:
    pass

# %%
cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']

if 'InternVL2' in model_name:
    predict_kwargs = {
        'max_num' : 4 if presentation_mode == 'separate' else 12,
        'give_reasoning': give_reasoning,
        'remove_explicit_question': remove_explicit_question,
    }
elif 'QVQ' in model_name:
    predict_kwargs = {
        'image_kwargs': {
            'do_resize': True,
            'size': {'shortest_edge': 28 * 28 * 4, 'longest_edge': 28 * 28 * max(4, int(1280 * scale_factor))},
        },
        'remove_explicit_question': remove_explicit_question,
    }
elif 'Qwen2.5' in model_name:
    predict_kwargs = {
        # 'min_pixels': 28 * 28 * 4,
        # 'max_pixels': 28 * 28 * max(4, int(1280 * scale_factor)),
        'image_kwargs': {
            'do_resize': True,
            'size': {'shortest_edge': 28 * 28 * 4, 'longest_edge': 28 * 28 * max(4, int(1280 * scale_factor))},
        },
        'give_reasoning': give_reasoning,
        'remove_explicit_question': remove_explicit_question,
    }
else:
    predict_kwargs = {}

# %%

biased_context_fn_dict = {
    'no_context': None,
    'ans_in_hint': None,
    'no_bias': lambda x, **kwargs: x,
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

def get_biased_context(bias_type, num_context_samples, randomize=False, exclude_index=None, wrong_examples=False):
    if biased_context_fn_dict[bias_type] is not None:
        biased_context_gen = biased_context_fn_dict[bias_type](no_bias(cv_bench, randomize=randomize), wrong_examples=wrong_examples)
        biased_context = []
        index_list = []
        for i, s in biased_context_gen:
            if i == exclude_index:
                continue
            if len(biased_context) >= num_context_samples:
                break
            biased_context.append(s)
            index_list.append(i)
            
    else:
        biased_context = []
        index_list = []
    return biased_context, index_list


# %%

model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)

# %%

for bias_type in bias_type_list:

    print(f"Context bias:{bias_type}")

    if not randomize_contexts:

        biased_context, index_list = get_biased_context(bias_type, num_context_samples, wrong_examples=False)
        if wrong_examples > 0:
            wrong_biased_context, wrong_index_list = get_biased_context(bias_type, num_context_samples, wrong_examples=True)
            wrong_indices_random = random.sample(range(num_context_samples), wrong_examples)
            for i in wrong_indices_random:
                biased_context[i] = wrong_biased_context[i]
                index_list[i] = wrong_index_list[i]
        print(f"In-context samples:", index_list)
        
    for arg_test_bias_type in test_bias_type_list:

        if arg_test_bias_type is None:
            test_bias_type = bias_type
        else:
            test_bias_type = arg_test_bias_type

        print(f"\t Test bias:{test_bias_type}")
        outputs = defaultdict(list)

        fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_test-{test_bias_type}_{"hint" if give_hint or bias_type == "ans_in_hint" else "no_hint"}{"_reasoning" if give_reasoning else ""}_{presentation_mode}_{num_context_samples}samples_{total_samples}testsamples_{num_contexts}contexts{description}.json'
        if (not redo) and os.path.exists(fname):
            print(f"Skipping {fname} because it already exists")
            continue

        num_samples_processed = 0
        for i, s in tqdm(no_bias(cv_bench), total=total_samples):

            if give_hint or bias_type == 'ans_in_hint':
                predict_kwargs['hint'] = hint_dict[bias_type].replace('<ANSWER>', s['answer']) 
            else:
                predict_kwargs['hint'] = None

            sb_list = biased_sample_fn_dict[test_bias_type](s)
                    
            for sb, desc in sb_list:
                
                outs = []
                index_lists = []
                for nci in range(num_contexts):
                    if randomize_contexts:
                        biased_context, index_list = get_biased_context(bias_type, num_context_samples, randomize=True, exclude_index=i, wrong_examples=False)  
                        if wrong_examples > 0:
                            wrong_biased_context, wrong_index_list = get_biased_context(bias_type, num_context_samples, wrong_examples=True)
                            wrong_indices_random = random.sample(range(num_context_samples), wrong_examples)
                            for i in wrong_indices_random:
                                biased_context[i] = wrong_biased_context[i]
                                index_list[i] = wrong_index_list[i]
                        

                    if presentation_mode == 'grid':
                        sb = grid_combine(sb, biased_context)
                        bc = []
                    elif presentation_mode == 'separate':
                        bc = biased_context
                    else:
                        raise ValueError(f"Invalid presentation mode: {presentation_mode}")

                    out = predict_fn(model, tokenizer, sb, bc, **predict_kwargs)
                    outs.append(out)
                    index_lists.append(index_list)

                outputs[desc].append(
                    {
                        'pred': outs,
                        'context_indices': index_lists,
                        'answer': sb['answer'],
                        'id': i
                    }
                )
            num_samples_processed += 1
            if num_samples_processed >= total_samples:
                break
        
        # outputs['context_indices'] = index_lists
        with open(fname, 'w') as f:
            json.dump(outputs, f)

# %%