from dataclasses import dataclass, field
from typing import Any, Iterable, List, Union
import yaml


def _normalize(d: Union[dict, list, str]) -> Union[dict, list, Any]:
    """
    Normalize all 'None' strings into None in all depths of the dictionary
    
    Args:
        d: dict, list or string to normalize
        
    Returns:
        dict, list or element with all 'None' strings converted into None
    """
    if isinstance(d, dict):
        return {k: _normalize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_normalize(v) for v in d]
    elif d == 'None':
        return None
    else:
        return d


@dataclass
class Dataset:
    """
    name: identifier of the knowledge base
    crosslingual: whether we want to use a crosslingual pairs, specific for MultiClaim dataset
    fact_check_language: language of the fact checking dataset, specific for MultiClaim dataset
    fact_check_fields: fields to be used for fact checking, specific for MultiClaim dataset
    language: language of the knowledge base
    post_language: language of the postprocessing, specific for MultiClaim dataset
    split: which split of the dataset to use (train, dev, test)
    doc_sent_count: number of sentences that should be in one document
    overlap: number of sentences that should overlap between documents
    document_path: path to the document or directory with documents
    version: version of the dataset (original, english), specific for MultiClaim dataset
    """
    name: str = None
    crosslingual: bool = False
    fact_check_language: str = None
    fact_check_fields: Iterable[str] = ('claim', )
    language: str = None
    post_language: str = None
    split: str = None
    doc_sent_count: int = 1
    overlap: int = 0
    document_path: str = None
    version: str = 'original'  # original, english"


@dataclass
class Retriever:
    """
    name: identifier of the retriever
    model_name: name of the model to be used, e.g. 'intfloat/multilingual-e5-large'
    top_k: number of documents to retrieve
    use_unidecode: whether to use unidecode for the query, specific for bm25 retriever
    knowledge_base: knowledge base to be used for the retrieval
    cache: path to the cache
    """
    name: str = None
    model_name: str = None
    top_k: int = 5
    use_unidecode: bool = False
    knowledge_base: Dataset = field(default_factory=Dataset)
    cache: str = None


@dataclass
class Prompt:
    """
    type: type of the prompt
    template: template of the prompt, See: https://docs.python.org/3/library/stdtypes.html#str.format
    """
    type: str = None
    template: str = None
    strategy: str = None
    examples_path: str = None
    num_examples: int = 6
    english: bool = False


@dataclass
class LLM:
    """
    model: identifier of the model
    model_name: name of the model to be used, e.g. 'meta-llama/Meta-Llama-3-8B-Instruct'
    max_new_tokens: maximum number of tokens to generate
    do_sample: whether to sample from the model
    device_map: device to use
    load_in_4bit: whether to load the model in 4bit
    load_in_8bit: whether to load the model in 8bit
    offload_folder: folder to offload the model
    offload_state_dict: whether to offload the state dict
    max_memory: maximum memory to use
    system_prompt: system prompt to use, e.g. 'You are a helpful assistant.'
    """
    model: str = None
    model_name: str = None
    max_new_tokens: int = 128
    do_sample: bool = False
    device_map: str = 'auto'
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    offload_folder: str = None
    offload_state_dict: bool = False
    max_memory: Any = None
    system_prompt: str = None


@dataclass
class Postprocessor:
    """
    name: identifier of the postprocessor
    prefix_name: name of the prefix to use, specific for ContextPostprocessor
    separator: separator to use, specific for ContextPostprocessor
    """
    name: str = None
    prefix_name: str = None
    separator: str = ' '


@dataclass
class RAGConfig:
    """
    RAGConfig class to hold the configuration of the RAG pipeline
    
    steps: list of steps in the pipeline
    
    """
    steps: List[Union[Dataset, Prompt, LLM, Postprocessor]] = None

    @classmethod
    def from_dict(cls, config: dict) -> 'RAGConfig':
        """
        Create RAGConfig object from a dictionary
        
        Args:
            config: dictionary with the configuration of the RAG pipeline
            
        Returns:
            RAGConfig object
        """
        steps = []
        for step in config['steps']:
            step = _normalize(step)
            if 'retriever' in step.keys():
                knowledge_base = Dataset(
                    **step['retriever']['knowledge_base'])
                step = {k: v for k, v in step['retriever'].items(
                ) if k != 'knowledge_base'}
                retriever = Retriever(**step, knowledge_base=knowledge_base)
                steps.append(retriever)
            elif 'prompt' in step.keys():
                prompt = Prompt(**step['prompt'])
                steps.append(prompt)
            elif 'llm' in step.keys():
                llm = LLM(**step['llm'])
                steps.append(llm)
            elif 'postprocessor' in step.keys():
                postprocessor = Postprocessor(**step['postprocessor'])
                steps.append(postprocessor)

        return cls(steps=steps)

    @classmethod
    def load_config(cls, path: str) -> 'RAGConfig':
        """
        Load configuration from a yaml file
        
        Args:
            path: path to the yaml file with the configuration
            
        Returns:
            RAGConfig object
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        return cls.from_dict(config)
