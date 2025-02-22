# %%

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
# # default: Load the model on the available device(s)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview", min_pixels=min_pixels, max_pixels=max_pixels)

# %%
response = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/QVQ/demo.png")
pil_img = Image.open(BytesIO(response.content))

messages_2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What value should be filled in the blank space?"},
            # {"type": "image"},
            # {"type": "text", "text": "What value should be filled in the blank space?"},
        ],
    }
]


# Preparation for inference
text_2 = processor.apply_chat_template(
    messages_2, tokenize=False, add_generation_prompt=True
)

inputs_2 = processor(
    text=[text_2],
    images=[pil_img],
    videos=None,
    padding=True,
    return_tensors="pt",
).to("cuda")

# %%

messages_3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What value should be filled in the blank space?"},
            {"type": "image"},
            {"type": "text", "text": "What value should be filled in the blank space?"},
        ],
    }
]


# Preparation for inference
text_3 = processor.apply_chat_template(
    messages_3, tokenize=False, add_generation_prompt=True
)

inputs_2 = processor(
    text=[text_2],
    images=[pil_img],
    videos=None,
    padding=True,
    return_tensors="pt",
).to("cuda")

# for k in inputs_1.keys():
#     if (inputs_1[k] == inputs_2[k]).all():
#         continue
#     else:
#         print(k)

# %%
# Inference: Generation of the output
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/QVQ-72B-Preview", torch_dtype="auto", device_map="auto"
)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
