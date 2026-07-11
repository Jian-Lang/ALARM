from PIL import Image
import base64
from io import BytesIO
import requests
import os
from typing import Optional
import dotenv
import time
from loguru import logger

# from .base_model import BaseModel


class APIModel():
    def __init__(self, model_id: str, params={}):
        """
        Initialize APIModel with model_id and API key.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4-vision-preview")
        """
        super().__init__()
        dotenv.load_dotenv()
        self.model_id = model_id
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable is not set")
        
        self.api_base = os.getenv("API_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.params = params

    @staticmethod
    def model_list():
        # Only GPT-4o is served through the OpenAI-compatible API (set API_URL /
        # API_KEY in .env). The Qwen backbone runs locally via Qwen2VLModel.
        return ["gpt-4o"]
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        image = image.convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def chat_img(self, prompt: str, image: Image.Image, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [image], max_tokens)

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.chat_multi_img(prompt, [], max_tokens)

    def chat_multi_img(self, prompt: str, images: list[Image.Image], max_tokens: int = 2048) -> str:
        """
        Generate text response for the given prompt and multiple images using OpenAI API.
        
        Args:
            prompt: Text prompt to process
            images: List of images to analyze (list of PIL Image objects)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        content = [{"type": "text", "text": prompt}]
        
        # Add each image to the content list
        for image in images:
            base64_image = self._encode_image(image)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            if 'Qwen' in self.model_id:
                image_content["image_url"]["detail"] = 'high'
            if 'gpt' in self.model_id:
                image_content["image_url"]["detail"] = 'low'
            content.append(image_content)

        payload = {
            "model": self.model_id,
            "messages": [
                # {
                #     "role": "system",
                #     "content": [
                #         {
                #             "type": "text",
                #             "text": "You are an expert in detecting hateful memes."
                #         }
                #     ]
                # },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.000001
        }
        # merage payload with params
        payload.update(self.params)
        
        # Try up to 3 times with 30 second sleep on 429 errors
        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                # If we get a 429 Too Many Requests error and have attempts left, sleep and retry
                if response.status_code == 429 and attempt < max_retries - 1:
                    retry_delay = 30 * (2 ** attempt)  # 30s, 60s, 120ss
                    logger.warning(f"Rate limit (429) hit. Retrying after {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"API request {payload['messages'][0]['content'][0]}\n Response: {result}")
                return result['choices'][0]['message']['content'].strip()
                
            except requests.exceptions.RequestException as e:
                # If this is our last attempt, raise the exception
                time.sleep(30)
                if attempt == max_retries - 1:
                    raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")

def test_multi_image_chat(image_paths: list[str], prompt: str = "What's in these images?", model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ") -> str:
    """
    Test function to verify if multiple images can be used for chat with the model.
    
    Args:
        image_paths: List of paths to the image files
        prompt: Text prompt to send with the images
        model_id: Model identifier to use for the test
        
    Returns:
        str: Response from the model
    """
    try:
        # Load the images and convert to RGB to ensure compatibility
        images = [Image.open(path).convert('RGB') for path in image_paths]
        
        # Create an instance of APIModel
        api_model = APIModel(model_id=model_id)
        
        # Send the prompt with multiple images
        response = api_model.chat_multi_img(prompt, images)
        
        return response
    except Exception as e:
        return f"Error during test: {str(e)}"

