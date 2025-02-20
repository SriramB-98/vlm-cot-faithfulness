import torch
from internvl import internvl_preprocess, get_internvl_model_tokenizer
from llava_cot import get_llama_model_tokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
from qwen_vl_utils import process_vision_info


def internvl_cvbench_predict(
    model, tokenizer, sample_to_predict, context_samples, max_num=12, max_new_tokens=1024, do_sample=False
):

    question = "<image>\n" + sample_to_predict["prompt"]
    gen_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    history = []
    images = []
    for s in context_samples:
        history.append(("<image>\n" + s["prompt"], s["answer"].replace("(", "").replace(")", "")))
        images.append(s["image"])

    images.append(sample_to_predict["image"])
    pixel_values_list = [internvl_preprocess(img, max_num=max_num, dtype=torch.bfloat16)[0] for img in images]
    num_patches_list = [pv.size(0) for pv in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)

    response = model.chat(
        tokenizer, pixel_values, question, gen_config, num_patches_list=num_patches_list, history=history
    )
    return response


def llama_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False):

    content = [ {"type": "text", "text": "I will first give you a few examples, then I will ask you a question. Think step by step and reason through the question before giving the answer."}]
    images = []
    for i, s in enumerate(context_samples):
        images.append(s["image"])
        content.extend( [ {"type": "text", "text": "Example " + str(i) + ":"}, {"type": "image"}, {"type": "text", "text": s["prompt"] + ". The answer is " + s["answer"]}])

    images.append(sample_to_predict["image"])
    content.extend( [ {"type": "text", "text": "Question:"}, {"type": "image"}, {"type": "text", "text": sample_to_predict["prompt"]}])

    messages = [{"role": "user", "content": content}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return output


def qwen_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False):
    
    content = [ {"type": "text", "text": "I will first give you a few examples, then I will ask you a question. Think step by step and reason through the question before giving the answer."}]
    images = []
    for i, s in enumerate(context_samples):
        images.append(s["image"])
        content.extend( [ {"type": "text", "text": "Example " + str(i) + ":"}, {"type": "image"}, {"type": "text", "text": s["prompt"] + ". The answer is " + s["answer"]}])

    images.append(sample_to_predict["image"])
    content.extend( [ {"type": "text", "text": "Question:"}, {"type": "image"}, {"type": "text", "text": sample_to_predict["prompt"]}])

    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        # tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)



def get_model_tokenizer(model_name, *args, **kwargs):
    if "InternVL" in model_name:
        model, tokenizer = get_internvl_model_tokenizer(model_name, *args, **kwargs)
        return model, tokenizer
    elif "Llama" in model_name:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    elif 'Qwen' in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor
    else:
        raise ValueError(f"Model {model_name} not supported")
    
