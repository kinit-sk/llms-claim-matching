from typing import List

from sentence_transformers import SentenceTransformer
import torch

from src.retrievers.vectorizers.vectorizer import Vectorizer


class SentenceTransformerVectorizer(Vectorizer):
    """
    Vectorizer for `SentenceTransformer` models compatible with `sentence_transformers` library.
    """
    
    def __init__(
        self,
        dir_path: str,
        model_handle: str = None,
        model: SentenceTransformer = None,
        batch_size: int = 32
    ):
        """
        Attributes:
            dir_path: str  Path to cached vectors and vocab files.
            model_handle: str  Name of the model, either a HuggingFace repository handle or path to a local model.
            model: SentenceTransformer  A loaded model -- this option can be used during fine-tuning.
        """
        
        super().__init__(dir_path)
        
        if model_handle:
            self.model = SentenceTransformer(model_handle)
        else:
            self.model = model
            
        self.batch_size = batch_size
            
        assert self.model
        
        
    def _calculate_vectors(self, texts: List[str]) -> torch.tensor:
        
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=False,
        )