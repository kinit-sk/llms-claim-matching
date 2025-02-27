import re
from src.postprocess.postprocessor import Postprocessor

class ContextPostprocessor(Postprocessor):
    """
    Postprocessor that formats the context for the language model.
    
    Attributes:
        prefix_name (str): The prefix name.
        separator (str): The separator.
    """
    def __init__(self, prefix_name: str = None, separator: str = '\n'):
        super().__init__('ContextPostprocessor')
        self.prefix_name = prefix_name
        self.separator = separator

    def postprocess(self, **kwargs) -> dict:
        """
        Postprocess the context based on the separator and prefix name.
        
        If the prefix name is not None, the context will be formatted as follows:
        {prefix_name} {idx + 1}: {doc}
        
        Returns:
            dict: The postprocessed context."""
        documents = kwargs['documents']
        query = kwargs['query']
        if self.prefix_name:
            context = f'{self.separator}'.join([f'{self.prefix_name} {idx + 1}: {doc}' for idx, doc in enumerate(documents)])
        else:
            context = f'{self.separator}'.join([f'{doc}' for doc in documents])
            
        return {
            'context': context,
            'query': query
        }