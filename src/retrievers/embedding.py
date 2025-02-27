import logging
from typing import Any, List, Generator, Tuple
from nltk.tokenize import sent_tokenize
import torch
from torch_scatter import scatter
import nltk

from src.datasets.cleaning import replace_stops, replace_whitespaces
from src.retrievers.retriever import Retriever
from src.retrievers.vectorizers.vectorizer import Vectorizer

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def slice_text(text, window_type, window_size, window_stride=None) -> List[str]:
    """
    Split a `text` into parts using a sliding window. The windows slides either across characters or sentences, based on the value of `window_tyoe`.

    Attributes:
        text: str  Text that is to be splitted into windows.
        window_type: str  Either `sentence` or `character`. The basic unit of the windows.
        window_size: int  How many units are in a window.
        window_stride: int  How many units are skipped each time the window moves.
    """

    text = replace_whitespaces(text)

    if window_stride is None:
        window_stride = window_size

    if window_size < window_stride:
        logger.warning(
            f'Window size ({window_size}) is smaller than stride length ({window_stride}). This will result in missing chunks of text.')

    if window_type == 'sentence':
        text = replace_stops(text)
        sentences = sent_tokenize(text)
        return [
            ' '.join(sentences[i:i+window_size])
            for i in range(0, len(sentences), window_stride)
        ]

    elif window_type == 'character':
        return [
            text[i:i+window_size]
            for i in range(0, len(text), window_stride)
        ]


def gen_sliding_window_delimiters(post_lengths: List[int], max_size: int) -> Generator[Tuple[int, int], None, None]:
    """
    Calculate where to split the sequence of `post_lenghts` so that the individual batches do not exceed `max_size`
    """
    range_length = start = cur_sum = 0

    for post_length in post_lengths:
        if (range_length + post_length) > max_size:  # exceeds memory
            yield (start, start + range_length)
            start = cur_sum
            range_length = post_length
        else:  # memory still avail in current split
            range_length += post_length
        cur_sum += post_length

    if range_length > 0:
        yield (start, start + range_length)


class Embedding(Retriever):
    def __init__(
            self,
            name: str = 'embedding',
            top_k: int = 5,
            vectorizer_document: Vectorizer = None,
            vectorizer_query: Vectorizer = None,
            sliding_window: bool = False,
            sliding_window_pooling: str = 'max',
            sliding_window_size: int = None,
            sliding_window_stride: int = None,
            sliding_window_type: str = None,
            query_split_size: int = 1,
            dtype: torch.dtype = torch.float32,
            device: str = 'cpu',
            save_if_missing: bool = False,
            knowledge_base: Any = None,
            **kwargs: Any
    ):
        super().__init__(name, top_k, knowledge_base=knowledge_base)
        self.vectorizer_document = vectorizer_document
        self.vectorizer_query = vectorizer_query
        self.sliding_window = sliding_window
        self.sliding_window_pooling = sliding_window_pooling
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.sliding_window_type = sliding_window_type
        self.query_split_size = query_split_size
        self.dtype = dtype
        self.device = device
        self.save_if_missing = save_if_missing
        self.document_embeddings = None

    def calculate_embeddings(self):
        # logger.info('Calculating embeddings for fact checks')
        document_embeddings = self.vectorizer_document.vectorize(
            self.knowledge_base.get_documents_texts(),
            save_if_missing=self.save_if_missing,
            normalize=True
        )

        document_embeddings = document_embeddings.transpose(
            0, 1)  # Rotate for matmul

        self.document_embeddings = document_embeddings.to(
            device=self.device, dtype=self.dtype)

    def set_knowledge_base(self, knowledge_base: Any):
        self.knowledge_base = knowledge_base
        self.calculate_embeddings()
        
    def retrieve_sims(self, query: str) -> Any:
        if self.document_embeddings is None:
            self.calculate_embeddings()
            
        if self.sliding_window:
            # logger.info('Splitting query into windows.')
            window = slice_text(
                query,
                self.sliding_window_type,
                self.sliding_window_size,
                self.sliding_window_stride
            )

            # logger.info('Calculating embeddings for the windows')
            query_embeddings = self.vectorizer_query.vectorize(
                list(window),
                save_if_missing=self.save_if_missing,
                normalize=True
            )

            query_lengths = [len(post) for post in list(window)]
            segment_array = torch.tensor([
                i
                for i, num_windows in enumerate(query_lengths)
                for _ in range(num_windows)
            ])

            delimiters = list(gen_sliding_window_delimiters(
                query_lengths, self.query_split_size))
        else:
            # logger.info('Calculating embeddings for query')
            query_embeddings = self.vectorizer_query.vectorize(
                [query],
                save_if_missing=self.save_if_missing,
                normalize=True
            )
            delimiters = [(0, 1)]

        results = []
        
        # logger.info('Calculating similarity for data splits')
        for start_id, end_id in delimiters:

            sims = torch.mm(
                query_embeddings[start_id:end_id].to(
                    device=self.device, dtype=self.dtype),
                self.document_embeddings
            )

            if self.sliding_window:
                segments = segment_array[start_id:end_id]
                segments -= int(segments[0])

                sims = scatter(
                    src=sims,
                    index=segments,
                    dim=0,
                    reduce=self.sliding_window_pooling,
                )
            results.extend(sims.tolist()[0])
            
        return results

    def retrieve(self, query: str) -> Any:
        if self.document_embeddings is None:
            self.calculate_embeddings()

        if self.sliding_window:
            # logger.info('Splitting query into windows.')
            window = slice_text(
                query,
                self.sliding_window_type,
                self.sliding_window_size,
                self.sliding_window_stride
            )

            # logger.info('Calculating embeddings for the windows')
            query_embeddings = self.vectorizer_query.vectorize(
                list(window),
                save_if_missing=self.save_if_missing,
                normalize=True
            )

            query_lengths = [len(post) for post in list(window)]
            segment_array = torch.tensor([
                i
                for i, num_windows in enumerate(query_lengths)
                for _ in range(num_windows)
            ])

            delimiters = list(gen_sliding_window_delimiters(
                query_lengths, self.query_split_size))
        else:
            # logger.info('Calculating embeddings for query')
            query_embeddings = self.vectorizer_query.vectorize(
                [query],
                save_if_missing=self.save_if_missing,
                normalize=True
            )
            delimiters = [(0, 1)]

        results = []

        # logger.info('Calculating similarity for data splits')
        for start_id, end_id in delimiters:

            sims = torch.mm(
                query_embeddings[start_id:end_id].to(
                    device=self.device, dtype=self.dtype),
                self.document_embeddings
            )

            if self.sliding_window:
                segments = segment_array[start_id:end_id]
                segments -= int(segments[0])

                sims = scatter(
                    src=sims,
                    index=segments,
                    dim=0,
                    reduce=self.sliding_window_pooling,
                )

            sorted_ids = torch.argsort(sims, descending=True, dim=1)
            if self.top_k is None:
                top_k = sorted_ids[0, :].tolist()
            else:
                top_k = sorted_ids[0, :self.top_k].tolist()
            top_k = self.knowledge_base.map_topK(top_k)

            results.append([
                self.knowledge_base.get_document(int(fc_id))
                for fc_id in top_k
            ])

        results = [item for sublist in results for item in sublist]
        return results, top_k
