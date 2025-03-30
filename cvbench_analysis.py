# %%
# import os
# os.environ['HOME'] = '/workspace'
import os
import random
from datasets import load_dataset
import numpy as np
# import torchvision
from cvbench_biased_contexts import answer_always_a, answer_with_marking, answer_always_left, bbox_colored, bbox_thickened, grid_combine, no_bias
from cvbench_biased_contexts import switch_order, answers_with_marking, mirror, colored_bbox, thickened_bbox, hints_in_question
from model_utils import get_model_tokenizer, get_client
import torch
import json
# from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm
import argparse
from server_utils import gather_async_results

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
    parser.add_argument("--presentation_mode", type=str, default="separate")
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--server_batch_size", type=int, default=1)
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
    server = args.server
    server_batch_size = args.server_batch_size
except:
    print("No arguments provided, using default values")
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    bias_type_list = ['bbox_thickened']
    num_context_samples = 5
    val_size = 20
    give_reasoning = False
    arg_test_bias_type = None
    presentation_mode = "grid"
    num_contexts = 1
    randomize_contexts = False
    seed = 42
    total_samples = 100
    wrong_examples = 0
    remove_explicit_question = False
    server = None
    server_batch_size = 4
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
        'max_num' : max(1, int(12 * scale_factor)),
        'give_reasoning': give_reasoning,
        'remove_explicit_question': remove_explicit_question,
    }
elif 'QVQ' in model_name or 'Qwen2.5' in model_name:
    predict_kwargs = {
        'image_kwargs': {
            'do_resize': True,
            'min_pixels': 28 * 28 * 4,
            'max_pixels': 28 * 28 * max(4, int(1280 * scale_factor)),
            'size': {'shortest_edge': 28 * 28 * 4, 'longest_edge': 28 * 28 * max(4, int(1280 * scale_factor))},
        },
        'give_reasoning': give_reasoning and 'QVQ' not in model_name,
        'remove_explicit_question': remove_explicit_question,
    }
elif 'Llama-3.2' in model_name:
    presentation_mode = 'grid'
    predict_kwargs = {}
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
    'ans_in_hint': hints_in_question,
    'always_a': switch_order,
    'with_marking': answers_with_marking,
    'always_left': mirror,
    'bbox_colored': colored_bbox,
    'bbox_thickened': thickened_bbox,
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
if server is None:
    model, tokenizer, predict_fn = get_model_tokenizer(model_name, use_flash_attn=True, dtype=torch.bfloat16)
else:
    model, predict_fn = get_client(model_name, backend=server)
    tokenizer = model_name

# %%

for bias_type in bias_type_list:

    print(f"Context bias:{bias_type}")

    if not randomize_contexts:

        biased_context, index_list = get_biased_context(bias_type, num_context_samples, wrong_examples=False)
        print(f"In-context samples:", index_list)
        
    for arg_test_bias_type in test_bias_type_list:

        if arg_test_bias_type is None:
            test_bias_type = bias_type
        else:
            test_bias_type = arg_test_bias_type

        print(f"\t Test bias:{test_bias_type}")
        outputs = defaultdict(list)

        fname = f'results/cvbench_{model_name.replace("/", "_")}_{bias_type}_test-{test_bias_type}_no_hint{"_reasoning" if give_reasoning else ""}_{presentation_mode}_{num_context_samples}samples_{total_samples}testsamples_{num_contexts}contexts_scale-{scale_factor}{description}.json'
        if (not redo) and os.path.exists(fname):
            print(f"Skipping {fname} because it already exists")
            continue

        num_samples_processed = 0
        for i, s in tqdm(no_bias(cv_bench), total=total_samples):

            sb_list = biased_sample_fn_dict[test_bias_type](s)
                    
            for sb, desc in sb_list:
                
                outs = []
                index_lists = []
                for nci in range(num_contexts):
                    if randomize_contexts:
                        biased_context, index_list = get_biased_context(bias_type, num_context_samples, randomize=True, exclude_index=i, wrong_examples=False)

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
        
        if server is not None:
            if bias_type == 'no_context':
                bs = 4*server_batch_size
            else:
                bs = server_batch_size
            outputs = gather_async_results(outputs, batch_size=bs)
        # outputs['context_indices'] = index_lists
        with open(fname, 'w') as f:
            json.dump(outputs, f)

# %%