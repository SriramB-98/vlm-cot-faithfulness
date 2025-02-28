import torch
from qwen_vl_utils import process_vision_info
from internvl import internvl_preprocess, get_internvl_model_tokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration


def internvl_cvbench_predict(
    model, tokenizer, sample_to_predict, context_samples, max_num=12, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None
):

    gen_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    # history = []
    images = []
    question = ""
    for i, s in enumerate(context_samples):
        question += "Example " + str(i) + ":\n<image>\n" + s["prompt"] +  ". The answer is " + s["answer"] + ".\n"
        images.append(s["image"])

    question += "Question: \n<image>\n" + sample_to_predict["prompt"] + "\n Conclude your answer with either (A) or (B)."
    if give_reasoning:
        question += "\n Think step by step and reason through the question before giving the answer."
    if hint:
        question += "\n " + hint
    images.append(sample_to_predict["image"])
    pixel_values_list = [internvl_preprocess(img, max_num=max_num, dtype=torch.bfloat16)[0] for img in images]
    num_patches_list = [pv.size(0) for pv in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)

    response = model.chat(
        tokenizer, pixel_values, question, gen_config, num_patches_list=num_patches_list, #history=history
    )
    return response


def llama_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False, give_reasoning=False):

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

def qwen_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, return_context=False):
    content = []
    images = []
    for i, s in enumerate(context_samples):
        images.append(s["image"])
        content.extend( [ {"type": "text", "text": "Example " + str(i) + ":"}, {"type": "image"}, {"type": "text", "text": s["prompt"] + ". The answer is " + s["answer"]}])

    images.append(sample_to_predict["image"])
    content.extend( [ {"type": "text", "text": "Question:"}, {"type": "image"}, {"type": "text", "text": sample_to_predict["prompt"]}])
    content.extend( [ {"type": "text", "text": "Conclude your answer with either (A) or (B)."}] )
    if give_reasoning:
        content.extend( [ {"type": "text", "text": "Think step by step and reason through the question before giving the answer."}])
    if hint:
        content.extend( [ {"type": "text", "text": hint}])

    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(images, text=[prompt_text], padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    if return_context:
        return output, (messages, images)
    else:
        return output
    

def qvq_cvbench_predict(model, processor, sample_to_predict, context_samples, force_answer=True, force_answer_phrase="\n\n**Final Answer**\n\n", end_phrase="\\]",max_new_tokens=1024, do_sample=True, hint=None):
    
    # content = [ {"type": "text", "text": "I will first give you a few examples, then I will ask you a question. Give your reasoning in the answer too."}]
    output, (messages, images) = qwen_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=max_new_tokens, do_sample=do_sample, give_reasoning=False, hint=hint, return_context=True)

    if force_answer:
        if force_answer_phrase in output:
            if end_phrase in output:
                return output
            else:
                replace_str = ""
        else:
            replace_str = force_answer_phrase

        messages.append({"role": "assistant", "content": output})
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # prompt_text = prompt_text.replace("<|im_end|>", force_answer_phrase, -1)
        prompt_text = replace_str.join(prompt_text.rsplit("<|im_end|>", 1))
        inputs = processor(images, text=[prompt_text], padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=20, do_sample=do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output


def get_model_tokenizer(model_name, *args, **kwargs):
    if "InternVL" in model_name:
        model, tokenizer = get_internvl_model_tokenizer(model_name, *args, **kwargs)
        return model, tokenizer, internvl_cvbench_predict
    elif "Llama" in model_name:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, llama_cvbench_predict
    elif 'Qwen2-' in model_name or 'QVQ' in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        predict_fn = qvq_cvbench_predict if 'QVQ' in model_name else qwen_cvbench_predict
        return model, processor, predict_fn
    elif 'Qwen2.5-' in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor, qwen_cvbench_predict
    else:
        raise ValueError(f"Model {model_name} not supported")
    
