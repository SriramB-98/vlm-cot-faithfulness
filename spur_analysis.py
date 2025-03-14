# %%
import json
from spur_datasets import load_dataset
from model_utils import get_model_tokenizer
import argparse
from tqdm.auto import tqdm

# %%

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="clean_waterbirds")
    parser.add_argument("--give_reasoning", action='store_true')
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    give_reasoning = args.give_reasoning
    num_samples = args.num_samples
except:
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    dataset_name = 'clean_waterbirds'
    give_reasoning = True
    num_samples = 860


if 'InternVL2' in model_name:
    predict_kwargs = {
        'max_num' : 4,
        'give_reasoning': give_reasoning,
    }
elif 'QVQ' in model_name:
    predict_kwargs = {}
elif 'Qwen2.5' in model_name:
    predict_kwargs = {
        'give_reasoning': give_reasoning,
    }
else:
    predict_kwargs = {}

if dataset_name == 'clean_waterbirds':
    prompt = 'Is the bird in this image a (A) landbird or (B) waterbird?'
    class_names = ['landbird', 'waterbird']
    options = ['(A)', '(B)']    
    dataset = load_dataset(dataset_name)

elif dataset_name == 'celeba_blond' or dataset_name == 'celeba_blond_corrected':
    prompt = 'Is the person in this image a (A) blond or (B) not blond?'
    class_names = ['blond hair', 'not blond hair']
    options = ['(A)', '(B)']
    dataset = load_dataset(dataset_name)
elif 'spurious_imagenet' in dataset_name:
    class_name = dataset_name[len('spurious_imagenet_'):].replace('_', ' ')
    prompt = f'Is there a {class_name} in this image? (A) yes or (B) no'
    class_names = ['yes', 'no']
    options = ['(A)', '(B)']
    dataset = load_dataset(dataset_name, class_name=class_name)
else:
    raise ValueError(f"Dataset {dataset_name} not supported")

biased_context = []

# %%

model, tokenizer, predict_fn = get_model_tokenizer(model_name)

# %%

results_dict = {}

for i, (id, image, class_name, confounder) in tqdm(enumerate(dataset), total=num_samples):
    if i >= num_samples:
        break

    sample_dict = { 'prompt': prompt, 'image': image }

    out = predict_fn(model, tokenizer, sample_dict, biased_context, **predict_kwargs)

    if confounder not in results_dict:
        results_dict[confounder] = []

    results_dict[confounder].append({
        'pred': out,
        'answer': options[class_names.index(class_name)],
        'id': id
    })


# %%
fname = f'results/spur_{model_name.replace("/", "_")}_{dataset_name}_{"reasoning_" if give_reasoning else ""}{num_samples}samples.json'
with open(fname, 'w') as f:
    json.dump(results_dict, f)

# %%
