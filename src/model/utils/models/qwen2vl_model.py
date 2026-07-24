from av import OutputChangedError
from .base_model import BaseModel
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from PIL import Image
import requests
import einops
from loguru import logger
import torch
import base64
from io import BytesIO, text_encoding
from qwen_vl_utils import process_vision_info
from icecream import ic

# AutoAWQ compatibility shim. Loading an AWQ checkpoint makes transformers import autoawq,
# and autoawq 0.2.9 (the only installable build, now discontinued) imports
# `PytorchGELUTanh` from transformers.activations — a symbol current transformers removed.
# Re-add it when missing so the AWQ backbone imports; a no-op on older transformers. Runs
# at module import, before any from_pretrained() triggers the autoawq import.
import transformers.activations as _tf_activations
if not hasattr(_tf_activations, "PytorchGELUTanh"):
    class PytorchGELUTanh(torch.nn.GELU):
        def __init__(self, *args, **kwargs):
            super().__init__(approximate="tanh")
    _tf_activations.PytorchGELUTanh = PytorchGELUTanh


def resize_image(image:Image.Image, max_size=640):
    """
    Resize an image while maintaining its aspect ratio, ensuring the longest side is max_size pixels.
    
    Args:
        image (PIL.Image.Image): The image to resize
        max_size (int): Maximum length of the longest dimension
        
    Returns:
        PIL.Image.Image: Resized image
    """
    # Get current dimensions
    width, height = image.size
    
    # Calculate the scaling factor
    if width >= height:
        # Width is the longest dimension
        scaling_factor = max_size / width
    else:
        # Height is the longest dimension
        scaling_factor = max_size / height
    
    # Calculate new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    # Resize the image
    return image.resize((new_width, new_height), Image.LANCZOS)


def pil_image_to_base64(image, image_format="PNG"):
    """
    Convert a PIL.Image.Image object to a Base64 string.

    Args:
        image (PIL.Image.Image): The image to be converted.
        image_format (str): Format to save the image, e.g., "PNG", "JPEG".

    Returns:
        str: The Base64-encoded string of the image.
    """
    # resize image
    image = resize_image(image)
    # Create a BytesIO object
    buffer = BytesIO()

    # Save the image to the buffer in the specified format
    image.save(buffer, format=image_format)

    # Ensure the pointer is at the beginning of the buffer
    buffer.seek(0)

    # Get the binary content of the image and encode it as Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Close the buffer
    buffer.close()

    return image_base64


class Qwen2VLModel(BaseModel):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-72B-Instruct-AWQ", params={}):
        super().__init__()
        # Prefer FlashAttention-2 when it is installed; otherwise fall back to the
        # PyTorch SDPA kernel so the local model still loads without flash-attn.
        try:
            import flash_attn  # noqa: F401
            attn_implementation = "flash_attention_2"
        except ImportError:
            attn_implementation = "sdpa"
        # AWQ GEMM kernels are float16-only. The checkpoint config declares bfloat16, and
        # transformers only casts it down when given an explicit torch.bfloat16 — passing
        # "auto" leaves the model in bfloat16 and the dequant kernel fails to compile at
        # generate() time. Pin float16 for AWQ weights; keep "auto" for the full model.
        torch_dtype = torch.float16 if "AWQ" in model_id else "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="cuda",
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.params = params if params else {}
        
        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")
        
        self.harmful_id = self.tokenizer.convert_tokens_to_ids("Ġharmful")
        self.harmless_id = self.tokenizer.convert_tokens_to_ids("Ġharmless")
        
    @staticmethod
    def model_list():
        # Local (GPU) LMM backbone, run via transformers. The same Qwen2.5-VL-72B
        # agent serves every stage: the label stage needs token-level logits for the
        # harmful/benign confidence, and the reference stage uses it text-only
        # (chat_text). The -AWQ variant is a lower-memory drop-in.
        return ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"]

    
    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 512) -> str:
        image = resize_image(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text[0]
    
    def chat_img_batch(self, prompts: list[str], images: list[Image.Image], max_tokens: int = 256) -> list[str]:
        conversations = [
            [{
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }] for prompt in prompts
        ]
        
        text_prompts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            ) for conversation in conversations
        ]
        
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text
    

    def chat_label(self, prompt: str, image: Image.Image) -> int:
        image = resize_image(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_dict = self.model.generate(
            **inputs,
            max_new_tokens=1,
            **self.params,
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=True
        )
        output_ids = output_dict.sequences
        logits = output_dict.logits
        
        # get logits for 1 and 0
        logits = logits[0][0, :]
        
        logits_1 = logits[self.one_id].unsqueeze(-1) 
        logits_0 = logits[self.zero_id].unsqueeze(-1)
        cls_logits = torch.cat([logits_0, logits_1], dim=-1)

        probs = torch.softmax(cls_logits, dim=-1)
        probs = probs.cpu().numpy()
        # to list
        probs = probs.tolist()
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(f"output_text: {output_text}")
        logger.debug(f"probs: {probs}")
        
        return output_text[0], probs
    
    
    def chat_multi_img(self, prompt: str, images: list[Image.Image]) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                ],
            }
        ]
        for i in range(len(images)):
            conversation[0]["content"].append({"type": "image", "image": f'data:image;base64,{pil_image_to_base64(images[i])}'})
        conversation[0]["content"].append({"type": "text", "text": prompt})
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            **self.params,
            max_new_tokens=1024
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text[0]

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        output = self.chat_text_batch([prompt], max_tokens)
        return output[0]
    
    def chat_text_batch(self, prompts: list[str], max_tokens: int = 2048) -> list[str]:
        conversations = [
            [{
                "role": "user", 
                "content": prompt
            }] for prompt in prompts
        ]
        
        texts = self.tokenizer.apply_chat_template(  
            conversations,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params
        )

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug(output_text)
        
        return output_text
    