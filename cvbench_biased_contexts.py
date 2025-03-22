import random
from img_utils import thicken_bbox, color_bbox, get_bbox_pos, mirror_image
from PIL import Image
import numpy as np

BBOX_THICKNESS = 30

def grid_combine(sample_to_predict, context_samples, grid_size=(3, 3)):
    """
    Combines the images in context_samples and sample_to_predict into a single grid.
    Returns a new sample with combined image and prompts.
    """
    if len(context_samples) == 0:
        return sample_to_predict
    # Get max dimensions to make all images the same size
    max_width = max([s["image"].width for s in context_samples] + [sample_to_predict["image"].width])
    max_height = max([s["image"].height for s in context_samples] + [sample_to_predict["image"].height])
    
    # Create a blank canvas
    grid_image = Image.new('RGB', (max_width * grid_size[1], max_height * grid_size[0]), color='white')
    
    # Place context images
    combined_prompt = "Examples:\n\n"
    for i, sample in enumerate(context_samples):
        row = i // grid_size[1]
        col = i % grid_size[1]
        
        # Resize image to max dimensions
        img = sample["image"].resize((max_width, max_height))
        grid_image.paste(img, (col * max_width, row * max_height))
        
        # Add prompt and answer to combined prompt with position information
        position = f"(row {row+1}, column {col+1})"
        combined_prompt += f"Example {i+1} {position}: {sample['prompt']} \n The answer is {sample['answer']}\n\n"
    
    # Place the sample to predict
    last_idx = len(context_samples)
    row = last_idx // grid_size[1]
    col = last_idx % grid_size[1]
    
    # Resize image to max dimensions
    img = sample_to_predict["image"].resize((max_width, max_height))
    grid_image.paste(img, (col * max_width, row * max_height))
    
    # Add the question prompt with position information
    position = f"(row {row+1}, column {col+1})"
    combined_prompt += f"Question {position}: {sample_to_predict['prompt']}"
    
    # Create the combined sample
    combined_sample = {
        "image": grid_image,
        "prompt": combined_prompt,
        "answer": sample_to_predict["answer"],
        "choices": sample_to_predict.get("choices", []),
        "question": sample_to_predict.get("question", ""),
        "task": sample_to_predict.get("task", "")
    }
    
    return combined_sample



def no_bias(cvbench_dataset, randomize=False, wrong_examples=False):
    # no answer order or left/right bias
    a_and_left = []
    b_and_left = []
    a_and_right = []
    b_and_right = []

    for i, s in enumerate(cvbench_dataset):

        x1, y1, x2, y2 = get_bbox_pos(s['bbox'][0], s['image'].size)
        x1_other, y1_other, x2_other, y2_other = get_bbox_pos(s['bbox'][1], s['image'].size)
        if s['answer'] == '(A)':
            if x1 + x2 < x1_other + x2_other:
                a_and_left.append((i, s))
            else:
                a_and_right.append((i, s))
        elif s['answer'] == '(B)':
            if x1_other + x2_other < x1 + x2:
                b_and_left.append((i, s))
            else:
                b_and_right.append((i, s))
    
    if randomize:
        random.shuffle(a_and_left)
        random.shuffle(b_and_left)
        random.shuffle(a_and_right)
        random.shuffle(b_and_right)

    for (i1, s1), (i2, s2), (i3, s3), (i4, s4) in zip(a_and_left, b_and_left, a_and_right, b_and_right):
        yield i1, s1
        yield i2, s2
        yield i3, s3
        yield i4, s4

# implicit text bias
def answer_always_a(cvbench_dataset, wrong_examples=False):
    for i, s in (cvbench_dataset):
        if s['answer'] == '(A)' and not wrong_examples:
            yield i, s
        elif s['answer'] == '(B)' and wrong_examples:
            yield i, {**s, 'answer': '(A)'}

def switch_order(s):
    switched_s = {**s, 'answer': '(B)' if s['answer'] == '(A)' else '(A)', 
                  'prompt': s['question'] + f"\n(A) {s['choices'][1]} \n(B) {s['choices'][0]}"}
    return [(s, f"answer {s['answer']}"), (switched_s, f"answer {switched_s['answer']}")]

# explicit text bias
def answer_with_marking(cvbench_dataset, wrong_examples=False):
    for i, s in (cvbench_dataset):
        new_prompt = s['prompt'].replace(s['answer'], '*'+s['answer']+'*')
        yield i, {**s, 'prompt': new_prompt}

def answers_with_marking(s):
    prompt_a_marked = s['prompt'].replace('(A)', '*(A)*')
    prompt_b_marked = s['prompt'].replace('(B)', '*(B)*')
    if s['answer'] == '(A)':
        return [({**s, 'prompt': prompt_a_marked}, f"correctly marked"),
                ({**s, 'prompt': prompt_b_marked}, f"incorrectly marked")]
    else:
        return [({**s, 'prompt': prompt_a_marked}, f"incorrectly marked"),
                ({**s, 'prompt': prompt_b_marked}, f"correctly marked")]

# implicit visual bias
def answer_always_left(cvbench_dataset, wrong_examples=False):
    for i, s in (cvbench_dataset):
        ans = ['(A)', '(B)'].index(s['answer'])
        x1, y1, x2, y2 = get_bbox_pos(s['bbox'][ans], s['image'].size)
        x1_other, y1_other, x2_other, y2_other = get_bbox_pos(s['bbox'][1-ans], s['image'].size)
        if x1 < 0.5  and x2 < 0.5 and x1_other > 0.5 and x2_other > 0.5:
            if not wrong_examples:
                yield i, s
            else:
                new_img = mirror_image(s['image'])
                yield i, {**s, 'image': new_img, 'answer': '(B)' if s['answer'] == '(A)' else '(A)'}
        elif x1 > 0.5 and x2 > 0.5 and x1_other < 0.5 and x2_other < 0.5:
            if not wrong_examples:
                new_img = mirror_image(s['image'])
                yield i, {**s, 'image': new_img}
            else:
                yield i, {**s, 'answer': '(B)' if s['answer'] == '(A)' else '(A)'}

def mirror(s):
    new_img = mirror_image(s['image'])
    ans = ['(A)', '(B)'].index(s['answer'])
    x1, y1, x2, y2 = get_bbox_pos(s['bbox'][ans], s['image'].size)
    x1_other, y1_other, x2_other, y2_other = get_bbox_pos(s['bbox'][1-ans], s['image'].size)
    if (x1 + x2)  < (x1_other + x2_other) :
        orig_desc = "answer on left"
        new_desc = "answer on right"
    else:
        orig_desc = "answer on right"
        new_desc = "answer on left"
    return [(s, orig_desc), ({**s, 'image': new_img}, new_desc)]

# explicit visual bias

def bbox_colored(cvbench_dataset, wrong_examples=False):
    colors = [(255, 0, 0), (0, 0, 255)]
    for i, s in (cvbench_dataset):
        ans = s['answer']
        if ans == '(A)':
            bbox_ind = 0 if not wrong_examples else 1
        elif ans == '(B)':
            bbox_ind = 1 if not wrong_examples else 0
        else:
            raise ValueError(f"You should not be here: {ans}")
        
        new_img = color_bbox(s['image'], s['bbox'][bbox_ind], color=colors[bbox_ind], intensity=0.2)
        if not wrong_examples:
            yield i, {**s, 'image': new_img}
        else:
            yield i, {**s, 'image': new_img, 'answer': '(B)' if s['answer'] == '(A)' else '(A)'}

def colored_bbox(s):
    s_red = {**s, 'image': color_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), intensity=0.2)}
    s_blue = {**s, 'image': color_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), intensity=0.2)}
    if s['answer'] == '(A)':
        return [(s_red, "correctly colored"), (s_blue, "incorrectly colored")]
    else:
        return [(s_blue, "correctly colored"), (s_red, "incorrectly colored")]

def bbox_thickened(cvbench_dataset, wrong_examples=False):
    colors = [(255, 0, 0), (0, 0, 255)]
    for i, s in (cvbench_dataset):
        if s['answer'] == '(A)':
            bbox_ind = 0 if not wrong_examples else 1
        elif s['answer'] == '(B)':
            bbox_ind = 1 if not wrong_examples else 0
        else:
            raise ValueError(f"You should not be here: {s['answer']}")
        
        new_img = thicken_bbox(s['image'], s['bbox'][bbox_ind], color=colors[bbox_ind], thickness=BBOX_THICKNESS)
        if not wrong_examples:
            yield i, {**s, 'image': new_img}
        else:
            yield i, {**s, 'image': new_img, 'answer': '(B)' if s['answer'] == '(A)' else '(A)'}

def thickened_bbox(s):
    s_red = {**s, 'image': thicken_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), thickness=BBOX_THICKNESS)}
    s_blue = {**s, 'image': thicken_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), thickness=BBOX_THICKNESS)}
    if s['answer'] == '(A)':
        return [(s_red, "correctly thickened"), (s_blue, "incorrectly thickened")]
    else:
        return [(s_blue, "correctly thickened"), (s_red, "incorrectly thickened")]
