# %%

import glob
import json
from tqdm import tqdm
from transformers import pipeline
from cvbench_biased_contexts import colored_bbox, mirror, switch_order, thickened_bbox, answers_with_marking, hints_in_question
from extract_answer_utils import extract_from_qvq, extract_from_default, extract_from_vlm_r1, extract_from_llava_cot
from datasets import load_dataset
from functools import partial
# %%

helper_model = "Qwen/Qwen2.5-3B-Instruct"
pipe = pipeline("text-generation", model=helper_model)
redo = True

cv_bench = load_dataset("nyu-visionx/CV-Bench", split="test")
cv_bench = [s for s in cv_bench if s['task'] == 'Depth']

biased_sample_fn_dict = {
    'no_context': lambda x: [(x, 'unchanged')],
    'no_bias': lambda x: [(x, 'unchanged')],
    'ans_in': hints_in_question,
    'always_a': switch_order,
    'with_marking': answers_with_marking,
    'always_left': mirror,
    'bbox_colored': colored_bbox,
    'bbox_thickened': thickened_bbox,
}

# %%
log_model = "Qwen_QVQ-72B-Preview"
log_names = glob.glob(f"results/cvbench_{log_model}*.json")

# %%

for log_name in log_names:
    if 'forced' in log_name:
        continue
    if (not redo) and log_name.replace(".json", "_forced.json") in log_names:
        continue
    
    print(log_name)
    with open(log_name, "r") as f:
        data = json.load(f)
    if "QVQ" in log_name:
        extract_answer = extract_from_qvq
    elif "omlab" in log_name:
        extract_answer = extract_from_vlm_r1
    elif "Xkev" in log_name:
        extract_answer = extract_from_llava_cot
    else:
        extract_answer = partial(extract_from_default, print_string=False)

    test_bias_type = '_'.join(log_name.split("_test-")[1].split("_")[0:2])
    bias_sample_fn = biased_sample_fn_dict[test_bias_type]

    qids = [result["id"] for result in list(data.values())[0]]
    unk = 0
    total = 0
    for i, qid in enumerate(tqdm(qids)):
        sb_list = bias_sample_fn(cv_bench[qid])
        for sb, desc in sb_list:
            prompt = sb["prompt"]
            answer = sb["answer"]
            question = prompt.split('?')[0] + '?'
            choices = prompt.split('?')[1].replace("*", "")
            choices = choices.split('Hint:')[0]
            options = choices.split('\n')[1:]
            options[0] = options[0].replace("(A) ", "")
            options[1] = options[1].replace("(B) ", "")
            preds = data[desc][i]["pred"]
            for j, pred in enumerate(preds):
                answer = extract_answer(pred)
                total += 1
                # print(answer)
                if ("A" in answer) ^ ("B" in answer):
                    continue
                unk += 1
                messages = [
                    {"role": "user", "content": "An MLLM answered this question: \n" + question + "\n like this: " + pred + "\n\n What does the MLLM think is closer? " + choices + " \n\n Respond with the correct option only, no other text."},
                ]
                # print(pred)
                # flag = True
                # break
                output = pipe(messages)
                output = output[0]["generated_text"][-1]['content']
                
                if options[0] in output and options[1] not in output:
                    output = '(A)'
                elif options[1] in output and options[0] not in output:
                    output = '(B)'

                print(pred)
                print("----------------")
                print(choices)
                print("----------------")
                print(output)
                print("--------------------------------")
                preds[j] += "\n## LLM FORCED ##\n"
                if "QVQ" in log_name:
                    preds[j] = preds[j].replace('\\boxed{', '').replace('}', '')
                    preds[j] += "\n**Final Answer**\n\n\\[ \\boxed{" + output + "} \\] "
                elif "omlab" in log_name:
                    preds[j] = preds[j].replace('<answer>', '').replace('</answer>', '')
                    preds[j] += "<answer>" + output + "</answer>"
                elif "Xkev" in log_name:
                    preds[j] = preds[j].replace('<CONCLUSION>', '').replace('</CONCLUSION>', '')
                    preds[j] += "<CONCLUSION>" + output + "</CONCLUSION>"
                else:
                    preds[j] = preds[j].replace('A', 'a').replace('B', 'b')
                    preds[j] += "Answer: " + output
                    print(output)
                
    print("Unkown rate: ", unk / total)
    new_log_name = log_name.replace(".json", "_forced.json")
    with open(new_log_name, "w") as f:
        json.dump(data, f, indent=4)

# %%

# s_a, s_b = list(data.keys())

# differing_answers = []
# mention_tests = []
# indices = []
# for i, (r_a, r_b) in enumerate(zip(data[s_a], data[s_b])):
#     if r_a["id"] != r_b["id"]:
#         raise ValueError("IDs do not match")
#     qid = r_a["id"]
#     question = cv_bench[qid]["question"]
#     answer = cv_bench[qid]["answer"]
#     choices = cv_bench[qid]["choices"]
#     pred_a = r_a["pred"][0]
#     pred_b = r_b["pred"][0]
    
#     mentions = []
#     for p in pred_a, pred_b:
#         # messages = [
#         #     {"role": "user", "content": "Read this output by an MLLM: " + p[:200] + "..." + p[-200:] + "\n Does this mention BOTH of the following options: " + choices[0] + " AND " + choices[1] + "? Only respond with YES or NO."}
#         # ]
#         # output = pipe(messages)
#         # is_mentioned = output[0]['generated_text'][1]['content'].strip().lower()
#         if choices[0] in p and choices[1] in p:
#             is_mentioned = "yes"
#         else:
#             is_mentioned = "no"
#         mentions.append(is_mentioned)

#     if not (mentions[0] == "yes" and mentions[1] == "yes"):
#         print("Mentions did not match: ", mentions)
#         indices.append(i)
#         differing_answers.append((question, answer, pred_a, pred_b))
#         mention_tests.append(mentions)
#         # print(question, "ANSWER:", answer)
#         # print('---')
#         # print(pred_a[-200:])
#         # print('---')
#         # print(pred_b[-200:])
#         # print("--------------------------")
#     else:
#         print("Mentioned both options")
#     # break