from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_qvq_model_tokenizer(model_name, *args, **kwargs):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/QVQ-72B-Preview", torch_dtype="auto", device_map="auto"
    )
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")
    return model, processor

