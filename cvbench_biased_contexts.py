from img_utils import thicken_bbox, color_bbox, get_bbox_pos, mirror_image


def no_bias(cvbench_dataset):
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
    
    for (i1, s1), (i2, s2), (i3, s3), (i4, s4) in zip(a_and_left, b_and_left, a_and_right, b_and_right):
        yield i1, s1
        yield i2, s2
        yield i3, s3
        yield i4, s4

# implicit text bias
def answer_always_a(cvbench_dataset):
    for i, s in (cvbench_dataset):
        if s['answer'] == '(A)':
            yield i, s

def switch_order(s):
    switched_s = {**s, 'answer': '(B)' if s['answer'] == '(A)' else '(A)', 
                  'prompt': s['question'] + f"\n(A) {s['choices'][1]} \n(B) {s['choices'][0]}"}
    return [(s, f"answer {s['answer']}"), (switched_s, f"answer {switched_s['answer']}")]

# explicit text bias
def answer_with_marking(cvbench_dataset):
    for i, s in (cvbench_dataset):
        new_prompt = s['prompt'].replace(s['answer'], '*'+s['answer']+'*')
        yield i, {**s, 'prompt': new_prompt}

def answers_with_marking(s):
    prompt_a_marked = s['prompt'].replace('(A)', '*(A)*')
    prompt_b_marked = s['prompt'].replace('(B)', '*(B)*')
    return [({**s, 'prompt': prompt_a_marked}, f"marked (a)"),
            ({**s, 'prompt': prompt_b_marked}, f"marked (b)")]

# implicit visual bias
def answer_always_left(cvbench_dataset):
    for i, s in (cvbench_dataset):
        ans = ['(A)', '(B)'].index(s['answer'])
        x1, y1, x2, y2 = get_bbox_pos(s['bbox'][ans], s['image'].size)
        x1_other, y1_other, x2_other, y2_other = get_bbox_pos(s['bbox'][1-ans], s['image'].size)
        if x1 < 0.5  and x2 < 0.5 and x1_other > 0.5 and x2_other > 0.5:
            yield i, s
        elif x1 > 0.5 and x2 > 0.5 and x1_other < 0.5 and x2_other < 0.5:
            new_img = mirror_image(s['image'])
            yield i, {**s, 'image': new_img}

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

def bbox_colored(cvbench_dataset):
    for i, s in (cvbench_dataset):
        ans = s['answer']
        if ans == '(A)':
            new_img = color_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), intensity=0.2)
        elif ans == '(B)':
            new_img = color_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), intensity=0.2)
        else:
            raise ValueError(f"You should not be here: {ans}")
        yield i, {**s, 'image': new_img}

def colored_bbox(s):
    s_red = {**s, 'image': color_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), intensity=0.2)}
    s_blue = {**s, 'image': color_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), intensity=0.2)}
    return [(s_red, "red colored"), (s_blue, "blue colored")]

def bbox_thickened(cvbench_dataset):
    for i, s in (cvbench_dataset):
        if s['answer'] == '(A)':
            new_img = thicken_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), thickness=10)
        elif s['answer'] == '(B)':
            new_img = thicken_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), thickness=10)
        else:
            raise ValueError(f"You should not be here: {s['answer']}")
        yield i, {**s, 'image': new_img}

def thickened_bbox(s):
    s_red = {**s, 'image': thicken_bbox(s['image'], s['bbox'][0], color=(255, 0, 0), thickness=10)}
    s_blue = {**s, 'image': thicken_bbox(s['image'], s['bbox'][1], color=(0, 0, 255), thickness=10)}
    return [(s_red, "red thickened"), (s_blue, "blue thickened")]