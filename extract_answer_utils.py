import string as str_mod

def extract_from_llava_cot(string):
    """
    Extract answer within <CONCLUSION> and </CONCLUSION> tags
    """
    start = string.find('<CONCLUSION>')
    end = string.find('</CONCLUSION>')
    if start == -1 or end == -1:
        return ""
    return string[start+len('<CONCLUSION>'):end].strip()
    

def extract_from_qvq(string):
    """
    Extract answer in \\boxed{}
    """
    start = string.find('\\boxed{')
    end = string.find('}')
    if start == -1 or end == -1:
        return ""
    return string[start+len('\\boxed{'):end].strip()

def extract_from_vlm_r1(string):
    """
    Extract answer in <answer> and </answer> tags
    """
    start = string.find('<answer>')
    end = string.find('</answer>')
    if start == -1 or end == -1:
        return ""
    return string[start+len('<answer>'):end].strip()

def extract_from_default(string, answer_options=["(A)", "(B)"], print_string=True):

    # string = string[-50:]
    #replace punctuation with space
    # string_space = string.translate(str.maketrans(str_mod.punctuation, ' ' * len(str_mod.punctuation)))
    for ans in answer_options:
        other_ans = answer_options[1 - answer_options.index(ans)]
        if (
            string.endswith(ans[1])
            or (ans in string and other_ans not in string)
            # or (" " + ans[1] + " " in string_space and " " + answer_options[1 - answer_options.index(ans)] + " " not in string_space)
        ):
            return ans
    if print_string:
        print(string)
        print("########################")
    return ""

