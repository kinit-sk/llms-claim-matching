from typing import Any

from src.prompts.prompt import Prompt, UniversalPrompt, PointwisePrompt, FewShotPointwisePrompt, AggregationPrompt

def prompt_factory(name, **kwargs) -> Prompt:
    """
    Factory function to create a prompt object based on the prompt name.
    
    Args:
        name (str): The name of the prompt.
        
    Returns:
        Prompt: An instance of the prompt.
    """
    prompt = {
        'prompt': Prompt,
        'pointwise': PointwisePrompt,
        'universal': UniversalPrompt,
        'aggregation': AggregationPrompt,
        'fewshot_pointwise': FewShotPointwisePrompt,
    }[name]
    return prompt(**kwargs)