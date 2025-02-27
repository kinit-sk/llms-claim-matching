import json
import logging
import os
from typing import List

import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vectorizer:
    """
    An abstract class for vectorizers. Vectorizers calculate vectors for texts and handle thir caching so that we do not have to calculate
    the vector for the same text more than once.
    
    Currently supported:
            `SentenceTransformerVectorizer` - Used for models using `sentence_transformers` library.
            `PytorchVectorizer` - Used for pytorch models (nn.Module).
            
    The main call for `Vectorizer` is `vectorize`. This will calculate the appropriate vectors and store them in the `dict` attribute. The class also
    supports `save` and `load`. `dir_path` is used as path to a folder where there is a `vocab.json` stored with the collection of texts and `vectors.py`
    with a torch tensor of vectors for the texts.
    """
    
    def __init__(self, dir_path: str):
        self.dict = {}
        
        self.dir_path = dir_path
        if dir_path:
            self.vectors_path = os.path.join(dir_path, 'vectors.pt')
            self.vocab_path = os.path.join(dir_path, 'vocab.json')
        
            try:
                self.load()
                logger.info(f'Vector database with {len(self.dict)} records loaded')
            except FileNotFoundError:
                pass
                logger.info(f'No previous database found')

        
    def vectorize(self, texts: List[str], save_if_missing: bool = False, normalize: bool = False) -> torch.tensor:
        """
        The main API point for the users. Try to find the vectors in the existing database. For the missing texts, the vectors will be calculated
        and saved in the `self.dict`. 
        
        Attributes:
            save_if_missing: bool  Should the vectors in `dict` be saved after new vectors are calculated? This makes sense for models that will
                be used more than once.
            normalize: bool  Should the vectors be normalized. Useful for cosine similarity calculations.
        """
        
        missing_texts = list(set(texts) - set(self.dict.keys()))
        
        if missing_texts:
            
            # logger.info(f'Calculating {len(missing_texts)} vectors.')
            missing_vectors = self._calculate_vectors(missing_texts)
            for text, vector in zip(missing_texts, missing_vectors):
                self.dict[text] = vector
            
            if save_if_missing:
                self.save()
            
        vectors = torch.vstack([
            self.dict[text]
            for text in texts
        ])
        
        if normalize:
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            
        return vectors

    
    def _calculate_vectors(self, txts: List[str]) -> torch.tensor:
        """
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError

            
    def load(self):
        """
        Load vocab and vectors from appropriate files
        """
        with open(self.vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        
        vectors = torch.load(self.vectors_path)
        
        assert len(vocab) == len(vectors)

        self.dict = {
            text: vector
            for text, vector in zip(vocab, vectors)
        }
        
        
    def save(self):
        """
        Save vocab and vectors to appropriate files
        """
        os.makedirs(self.dir_path, exist_ok=True)
            
        vocab = list(self.dict.keys())        
        with open(self.vocab_path, 'w', encoding='utf8') as f:
            json.dump(vocab, f)
            
        vectors = torch.vstack(list(self.dict.values()))
        torch.save(vectors, self.vectors_path)
            