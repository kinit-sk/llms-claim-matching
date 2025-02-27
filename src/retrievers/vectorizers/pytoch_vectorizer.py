from typing import List

import torch

from src.retrievers.vectorizers.vectorizer import Vectorizer


class PytorchVectorizer(Vectorizer):
    """
    Vectorizer for `Pytorch` models.
    """
    
    def __init__(
        self,
        dir_path: str,
        model_handle: str = None,
        model: torch.nn.Module = None,
        tokenizer = None,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        port_embeddings_to_cpu: bool = True
    ):
        """
        Attributes:
            dir_path: str  Path to cached vectors and vocab files.
            model_handle: str  Name of the model, either a HuggingFace repository handle or path to a local model.
            model: SentenceTransformer  A loaded model -- this option can be used during fine-tuning.
            tokenizer: AutoTokenizer  A tokenizer for the model.
            batch_size: int  Batch size for inference
            dtype: torch.dtype  Inference dtype
            port_embeddings_to_cpu: bool  Whether to move the embeddings to CPU after inference.
        """
        
        super().__init__(dir_path)
        
        if model_handle:
            self.model = torch.load(model_handle)
            self.model.eval()        
        else:
            self.model = model

        assert self.model

        self.device = next(self.model.parameters()).device.type
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dtype = dtype
        self.port_embeddings_to_cpu = port_embeddings_to_cpu

        
    def _calculate_vectors(self, texts: List[str]) -> torch.tensor:

        @torch.autocast(device_type=self.device.split(':')[0], dtype=self.dtype)
        @torch.no_grad()
        def embedding_pipeline(text: List[str], tokenizer, model, device, max_length = 512):
            tokenized = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            embeddings = model(**tokenized)
            return embeddings.cpu() if self.port_embeddings_to_cpu else embeddings

        return torch.vstack(
                [
                    embedding_pipeline(
                        texts[i:i+self.batch_size], 
                        self.tokenizer, 
                        self.model, 
                        device=self.device, 
                        max_length=512
                    ) 
                    for i in range(0, len(texts), self.batch_size)
                ]
            )
            