from itertools import tee
import os
import tempfile
import torch
from functools import partial
from qwen_vl_utils import process_vision_info
from internvl import internvl_preprocess, get_internvl_model_tokenizer, get_internvl_pipeline
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLImageProcessor
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy import GenerationConfig
from transformers import pipeline
import vllm
import base64
from openai import AsyncOpenAI
import io
import atexit

VLM_R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
VLM_R1_REASONING_PROMPT = (
    "Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags"
)

# Keep track of temporary files
temp_files = []

def cleanup_temp_files():
    for file_path in temp_files:
        try:
            os.unlink(file_path)
            print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# Register the cleanup function
atexit.register(cleanup_temp_files)

def encode_image(image, img_encoding='base64'):
    if img_encoding == 'base64':
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_encoded = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        img_encoded_dict = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_encoded}"}}
    elif img_encoding == 'path':
        ## Save image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        image.save(temp_file.name)
        img_encoded = temp_file.name
        temp_files.append(temp_file.name)
        img_encoded_dict = {"type": "image_url", "image_url": {"url": f"file://{img_encoded}"}}
    elif img_encoding == 'blank':
        img_encoded_dict = {"type": "image"}
    else:
        raise ValueError(f"Invalid image encoding: {img_encoding}")
    return img_encoded_dict


def internvl_pipeline_cvbench_predict(
    pipe, tokenizer, sample_to_predict, context_samples, max_num=12, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, num_options=2, remove_explicit_question=False, 
):
    assert tokenizer is None
    if remove_explicit_question:
        raise NotImplementedError("Removing explicit question is not supported for InternVL")
    
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=do_sample)
    images = []
    question = ""
    for i, s in enumerate(context_samples):
        question += "Example " + str(i) + f":\n{IMAGE_TOKEN}\n" + s["prompt"] +  "\n The answer is " + s["answer"] + ".\n"
        images.append(s["image"])

    options = [ "(" + chr(65 + i) + ")" for i in range(num_options) ]
    option_str = ", ".join(options[:-1]) + " or " + options[-1]
    question += f"Question: \n{IMAGE_TOKEN}\n" + sample_to_predict["prompt"] + "\nEnd your answer with " + option_str + "."
    if give_reasoning:
        question += "\nThink step by step and carefully reason through the question before giving the answer."
    if hint:
        question += "\n" + hint
    images.append(sample_to_predict["image"])

    old_max_dynamic_patch = pipe.vl_encoder.model.config.max_dynamic_patch
    pipe.vl_encoder.model.config.max_dynamic_patch = max_num
    response = pipe((question, images), gen_config=gen_config)
    pipe.vl_encoder.model.config.max_dynamic_patch = old_max_dynamic_patch
    
    return response.text


def internvl_cvbench_predict(
    model, tokenizer, sample_to_predict, context_samples, max_num=12, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, num_options=2, remove_explicit_question=False
):
    if remove_explicit_question:
        raise NotImplementedError("Removing explicit question is not supported for InternVL")
    
    gen_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    # history = []
    images = []
    question = ""
    for i, s in enumerate(context_samples):
        question += "Example " + str(i) + ":\n<image>\n" + s["prompt"] +  "\n The answer is " + s["answer"] + ".\n"
        images.append(s["image"])

    options = [ "(" + chr(65 + i) + ")" for i in range(num_options) ]
    option_str = ", ".join(options[:-1]) + " or " + options[-1]
    question += "Question: \n<image>\n" + sample_to_predict["prompt"] + "\n End your answer with " + option_str + "."
    if give_reasoning:
        question += "\n Carefully reason through the question before giving the answer."
    if hint:
        question += "\n " + hint
    images.append(sample_to_predict["image"])
    pixel_values_list = [internvl_preprocess(img, max_num=max_num, dtype=torch.bfloat16)[0] for img in images]
    num_patches_list = [pv.size(0) for pv in pixel_values_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)

    print("Pixel values shape", pixel_values.shape)

    response = model.chat(
        tokenizer, pixel_values, question, gen_config, num_patches_list=num_patches_list, #history=history
    )
    return response


def llama_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, num_options=2):

    content = []
    images = []
    for i, s in enumerate(context_samples):
        images.append(s["image"])
        content.extend( [ {"type": "text", "text": "Example " + str(i) + ":"}, {"type": "image"}, {"type": "text", "text": s["prompt"] + ". The answer is " + s["answer"]}])

    images.append(sample_to_predict["image"])
    content.extend( [ {"type": "text", "text": "Question:"}, {"type": "image"}, {"type": "text", "text": sample_to_predict["prompt"]}])

    options = [ "(" + chr(65 + i) + ")" for i in range(num_options) ]
    content.extend( [ {"type": "text", "text": "End your answer with either " + ", ".join(options[:-1]) + " or " + options[-1] + "."}] )

    if give_reasoning:
        content.extend( [ {"type": "text", "text": "Carefully reason through the question before giving the answer."}])
    if hint:
        content.extend( [ {"type": "text", "text": hint}])

    messages = [{"role": "user", "content": content}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return output




def openai_template(sample_to_predict, context_samples, give_reasoning=False, hint=None, num_options=2, remove_explicit_question=False, reasoning_prompt=None, system_prompt=None, img_encoding='blank'):
    content = []
    images = []
   
    if remove_explicit_question:
        content.append({"type": "text", "text": "I will first give you a few examples, then I will ask you a question. Answer the question based on the examples."})


    for i, s in enumerate(context_samples):
        images.append(s["image"])
        if remove_explicit_question:
            prompt = s["prompt"].replace(s["question"], "Select the correct option: ")
        else:
            prompt = s["prompt"]
        content.extend( [ encode_image(s["image"], img_encoding=img_encoding), 
                         {"type": "text", "text": "\nExample " + str(i) + ":"},  
                         {"type": "text", "text": '\n'+ prompt + "\n The answer is " + s["answer"] + "\n"}])

    images.append(sample_to_predict["image"])
    if remove_explicit_question:
        # Assert that the question is a substring of the prompt
        assert sample_to_predict["question"] in sample_to_predict["prompt"], "Question must be a substring of the prompt"
        question = sample_to_predict["prompt"].replace(sample_to_predict["question"], "Select the correct option: ")
    else:
        question = sample_to_predict["prompt"]

    content.extend( [ encode_image(sample_to_predict["image"], img_encoding=img_encoding), 
                      {"type": "text", "text": "\nQuestion:"}, 
                      {"type": "text", "text": '\n' + question}])
    options = [ "(" + chr(65 + i) + ")" for i in range(num_options) ]
    content.extend( [ {"type": "text", "text": "\nEnd your answer with either " + ", ".join(options[:-1]) + " or " + options[-1] + "."}] )
    if give_reasoning:
        if reasoning_prompt is not None:
            content.extend( [ {"type": "text", "text": '\n' + reasoning_prompt}])
        else:
            content.extend( [ {"type": "text", "text": "\nThink step by step and carefully reason through the question before giving the answer."}])
        # content.extend( [ {"type": "text", "text": "Think step by step before giving the answer."}])
    if hint:
        content.extend( [ {"type": "text", "text": hint}])

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages, images



def qwen_cvbench_predict(model, processor, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, return_context=False, num_options=2, remove_explicit_question=False, image_kwargs=None, system_prompt=None, reasoning_prompt=None):
    
    messages, images = openai_template(sample_to_predict, context_samples, give_reasoning=give_reasoning, hint=hint, num_options=num_options, remove_explicit_question=remove_explicit_question, system_prompt=system_prompt, reasoning_prompt=reasoning_prompt, img_encoding="blank")

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    print("Prompt text", prompt_text)
    images_kwargs = {}
    if image_kwargs is not None:
        images_kwargs.update(image_kwargs)
        images_kwargs.pop("min_pixels", None)
        images_kwargs.pop("max_pixels", None)

    inputs = processor(images, text=[prompt_text], padding=True, return_tensors="pt", images_kwargs=images_kwargs).to(model.device)
    print("Input token length", [len(t) for t in inputs.input_ids])
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


async def openai_api_predict(client, model_name, sample_to_predict, context_samples, max_new_tokens=1024, do_sample=False, give_reasoning=False, hint=None, num_options=2, remove_explicit_question=False, system_prompt=None, reasoning_prompt=None, img_encoding='base64', **kwargs):

    for k, v in kwargs.items():
        print(f"Dropping {k} = {v} in openai_api_predict")

    messages, images = openai_template(sample_to_predict, context_samples, give_reasoning=give_reasoning, hint=hint, num_options=num_options, remove_explicit_question=remove_explicit_question, system_prompt=system_prompt, reasoning_prompt=reasoning_prompt, img_encoding=img_encoding)

    if 'QVQ' in model_name:
        messages[-1]["content"].append(
            {"type": "text", "text": "Be concise and to the point. You are LIMITED to think for at most 5 sentences before giving the answer. "}
        )

    chat_response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0 if not do_sample else 0.7,
        top_p=0.95,
    )
    output = chat_response.choices[0].message.content
    return output


def setup_openai_api(api_key, base_url):
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


def get_client(model_name, backend='vllm', api_key=None, base_url=None, **kwargs):
    if backend in ['vllm', 'lmdeploy', 'openai']:
        if api_key is None and base_url is None:
            if backend == 'vllm':
                api_key = "EMPTY"
                base_url = "http://localhost:27182/v1"
            elif backend == 'lmdeploy':
                api_key = "YOUR_API_KEY"
                base_url = "http://localhost:27182/v1"
            elif backend == 'openai':
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_BASE_URL")
        client = setup_openai_api(api_key, base_url)
        predict_fn = openai_api_predict
    else:
        raise ValueError(f"Invalid backend: {backend}")
    
    if 'VLM-R1' in model_name:
        predict_fn = partial(predict_fn, system_prompt=VLM_R1_SYSTEM_PROMPT, reasoning_prompt=VLM_R1_REASONING_PROMPT)
        
    return client, predict_fn

def get_model_tokenizer(model_name, *args, **kwargs):
    if "InternVL" in model_name:
        model = get_internvl_pipeline(model_name, *args, **kwargs)
        return model, None, internvl_pipeline_cvbench_predict
        if 'AWQ' in model_name:
            model = get_internvl_pipeline(model_name, *args, **kwargs)
            return model, None, internvl_pipeline_cvbench_predict
        else:
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
    elif 'Qwen2.5' in model_name:
        if 'AWQ' in model_name:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        if 'VLM-R1' in model_name:
            predict_fn = partial(qwen_cvbench_predict, system_prompt=VLM_R1_SYSTEM_PROMPT, reasoning_prompt=VLM_R1_REASONING_PROMPT)
        else:
            predict_fn = qwen_cvbench_predict
        return model, processor, predict_fn
    else:
        raise ValueError(f"Model {model_name} not supported")
    
