import torch
from typing import Any

from src.retrievers.vectorizers.sentence_transformer_vectorizer import SentenceTransformerVectorizer
from src.retrievers.retriever import Retriever
from src.retrievers.bm25 import BM25
from src.retrievers.embedding import Embedding


def retriever_factory(name, **kwargs: Any) -> Retriever:
    if name == 'embedding':
        vct = SentenceTransformerVectorizer(dir_path=kwargs['cache'], model_handle=kwargs['model_name'])
        kwargs['vectorizer_document'] = vct
        kwargs['vectorizer_query'] = vct
        kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        kwargs['save_if_missing'] = True

    retriever = {
        'bm25': BM25,
        'embedding': Embedding,
    }[name]
    return retriever(**kwargs)
