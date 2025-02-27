from typing import Any

from src.models.model import Model
from src.models.hf_model import HFModel


def model_factory(model: str, **kwargs) -> Model:
    """
    Factory function to create a model object based on the model name.
    
    Args:
        model (str): The name of the model.
        
    Returns:
        Model: An instance of the model.
    """
    Model = {
        'hf_model': HFModel,
    }[model]
    return Model(**kwargs)
