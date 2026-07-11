from typing import Dict

from .models.base_model import BaseModel
from .models.qwen2vl_model import Qwen2VLModel
from .models.api_model import APIModel

def MLLMFactory(model_id: str, params=None|Dict):
    # Check each model class directly
    model_classes = [Qwen2VLModel, APIModel]
    for model_class in model_classes:
        if model_id in model_class.model_list():
            return model_class(model_id, params)
    raise ValueError(f"Model {model_id} not supported")
