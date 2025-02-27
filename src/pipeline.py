from src.module import Module
from src.config import RAGConfig
from typing import Any, Union, List

from src.retrievers import retriever_factory
from src.models import model_factory
from src.prompts import prompt_factory
from src.postprocess import postprocessor_factory
from src.datasets import dataset_factory
from src.datasets.dataset import Dataset
from src.datasets.retrieved_documents import RetrievedDocuments
from src.config import Retriever, Prompt, LLM, Postprocessor
from src.retrievers.retriever import Retriever as RetrieverModule

from copy import deepcopy


class Pipeline(Module):
    """
    Pipeline class to hold the RAG pipeline
    
    Args:
        path: path to the configuration file
        rag_config: configuration of the RAG pipeline
    """
    def __init__(self, path: str = None, rag_config: dict = None):
        super().__init__('Pipeline')
        self.path = path
        self.rag_config = rag_config
        self.modules = []
        self.load()
        
    def get_knowledge_base(self, knowledge_base: Any) -> Dataset:
        knowledge_base = knowledge_base.__dict__
        name = knowledge_base.pop('name')

        if name == 'retrieved_documents':
            return None

        return dataset_factory(
            name, **knowledge_base
        ).load()
        
    def get_module(self, module: Union[Retriever, Prompt, LLM, Postprocessor]) -> Module:
        """
        Get the module based on the module type
        
        Args:
            module: configuration of the module
            
        Returns:
            Module: the created module
        """
        if isinstance(module, Retriever):
            cache = module.cache
            knowledge_base = self.get_knowledge_base(module.knowledge_base)
            return retriever_factory(module.name, model_name=module.model_name, top_k=module.top_k, knowledge_base=knowledge_base, cache=cache)
        elif isinstance(module, Prompt):
            name = module.type
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['type']
            }
            return prompt_factory(name, **rest)
        elif isinstance(module, LLM):
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['model_name', 'model']
            }
            print('Rest:', rest)
            return model_factory(module.model, name=module.model_name, **rest)
        elif isinstance(module, Postprocessor):
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['name']
            }
            return postprocessor_factory(module.name, **rest)
        else:
            raise ValueError(f'Unknown module type: {type(module)}')

    def _convert_kwargs(self, module: Union[Retriever, Prompt, LLM, Postprocessor], kwargs: dict) -> dict:
        if isinstance(module, RetrieverModule):
            self.retrieved_documents = kwargs['documents']
            self.retrieved_documents_ids = kwargs['top_k']

        return kwargs

    def __call__(self, **kwargs) -> Any:
        for idx, module in enumerate(self.modules):
            if isinstance(module, RetrieverModule) and isinstance(self.modules[idx - 1], RetrieverModule):
                if len(kwargs['documents']) == 0:
                    continue
                knowledge_base = RetrievedDocuments(
                    name='retrieved_documents',
                    documents=kwargs['documents'],
                    ids=kwargs['top_k']
                )
                kwargs = {
                    "query": kwargs['query']
                }
                module.set_knowledge_base(knowledge_base)

            kwargs = module(**kwargs)
            kwargs = self._convert_kwargs(module, kwargs)

        return kwargs
    
    def _load_modules(self) -> None:
        """
        Load the modules from the configuration
        """
        print('Cofig:', self.config)
        steps = deepcopy(self.config.steps)

        self.modules = []
        for step in steps:
            module = self.get_module(step)
            self.modules.append(module)
            
    def set_modules(self, modules: List[Union[Retriever, Prompt, LLM, Postprocessor]]) -> None:
        """
        Set the modules of the pipeline manually.
        
        Args:
            modules: list of modules
        """
        self.modules = modules

    def load(self) -> None:
        if self.rag_config:
            self.config = RAGConfig.from_dict(self.rag_config)
            self._load_modules()
        elif self.path:
            self.config = RAGConfig.load_config(self.path)
            self._load_modules()
