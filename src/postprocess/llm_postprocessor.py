import re
from src.postprocess.postprocessor import Postprocessor

class LLMPostprocessor(Postprocessor):
    """
    Postprocessor that formats the output of the language model.
    """
    def __init__(self, **kwargs):
        super().__init__('LLMPostprocessor')

    def postprocess(self, **kwargs) -> tuple:
        """
        Postprocess the output of the language model.
        
        The output is split by '\n\n' and the document ids are extracted.
        
        Returns:
            tuple: The postprocessed output and the documents.
        """
        output = kwargs['output']
        try:
            processed_output = output.split('\n\n')[0]
            documents = re.findall(r'(\d*),?', processed_output)
            documents = list(filter(None, documents))
        except Exception as e:
            documents = []

        return {
            'generated_text': output,
            'documents': documents
        }