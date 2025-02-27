from src.postprocess.postprocessor import Postprocessor


class RetrieverPostprocessor(Postprocessor):
    def __init__(self, **kwargs):
        super().__init__('retriever_postprocess')

    def postprocess(self, **kwargs) -> tuple:
        """
        Postprocess the output of the language model.

        Returns:
            tuple: The postprocessed output and the documents.
        """
        return (
            kwargs['documents'],
            kwargs['top_k'],
            None,
            None
        )