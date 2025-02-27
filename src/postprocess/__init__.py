from src.postprocess.postprocessor import Postprocessor
from src.postprocess.llm_postprocessor import LLMPostprocessor
from src.postprocess.retriever_postprocessor import RetrieverPostprocessor
from src.postprocess.pointwise_postprocessor import PointwisePostprocessor
from src.postprocess.context_postprocessors import ContextPostprocessor

def postprocessor_factory(name: str, **kwargs) -> Postprocessor:
    """
    Factory function to create a postprocessor object.
    
    Args:
        name (str): The name of the postprocessor.
        
    Returns:
        Postprocessor: An instance of the postprocessor.
    """
    potsprocessor = {
        'llm_postprocessor': LLMPostprocessor,
        'retriever_postprocess': RetrieverPostprocessor,
        'pointwise_postprocessor': PointwisePostprocessor,
        'context_postprocessor': ContextPostprocessor,
    }[name]
    return potsprocessor(**kwargs)