import re
from src.postprocess.postprocessor import Postprocessor

class PointwisePostprocessor(Postprocessor):
    """
    Postprocessor that formats the output of the language model.
    """
    def __init__(self, **kwargs):
        super().__init__('pointwise')

    def postprocess(self, **kwargs) -> tuple:
        """
        Postprocess the output of the language model.
        
        The output is split by '\n\n' and the Yes and No tokens are extracted for each document.

        Returns:
            tuple: The postprocessed output and the documents.
        """
        output = kwargs['output']
        yes_probs = kwargs['yes_probs']
        no_probs = kwargs['no_probs']
        try:
            documents = re.findall(r'(Yes|No|yes|no)(\.|,|#####|\s?)', output)
            documents = [f'{i + 1}' for i, d in enumerate(documents) if d[0].capitalize() == "Yes"]
        except:
            documents = []

        return output, documents, yes_probs, no_probs
        