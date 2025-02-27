from typing import Any, List
import src.datasets.cleaning as cleaning

class Dataset:
    """Dataset

    Abstract class for datasets. Subclasses should implement `load` function that load `id_to_documents`, `id_to_post`,
    and `fact_check_post_mapping` object attributes. The class also implemenets basic cleaning methods that might be
    reused.

    Attributes:
        clean_ocr: bool = True  Should cleaning of OCRs be performed
        remove_emojis: bool = False  Should emojis be removed from the texts?
        remove_urls: bool = True  Should URLs be removed from the texts?
        replace_whitespaces: bool = True  Should whitespaces be replaced by a single space whitespace?
        clean_twitter: bool = True  
        remove_elongation: bool = False  Should occurrence of a string of consecutive identical non-space 
    characters (at least three in a row) with just one instance of that character?

        After `load` is called, following attributes are accesible:
                fact_check_post_mapping: list[tuple[int, int]]  List of Factcheck-Post id pairs.
                id_to_documents: dict[int, str]  Factcheck id -> Factcheck text
                id_to_post: dict[int, str]  Post id -> Post text

    Methods:
        clean_text: Performs text cleaning based on initialization attributes.
        maybe_clean_ocr: Perform OCR-specific text cleaning, if `self.clean_ocr`
        load: Abstract method. To be implemented by the subclasses.

    """
    def __init__(
        self,
        clean_ocr: bool = True,
        dataset: str = None,  # Here to read and discard the `dataset` field from the argparser
        remove_emojis: bool = False,
        remove_urls: bool = True,
        replace_whitespaces: bool = True,
        clean_twitter: bool = True,
        remove_elongation: bool = False,
        **kwargs: Any
    ):
        self.clean_ocr = clean_ocr
        self.remove_emojis = remove_emojis
        self.remove_urls = remove_urls
        self.replace_whitespaces = replace_whitespaces
        self.clean_twitter = clean_twitter
        self.remove_elongation = remove_elongation
        
    def __len__(self):
        return len(self.fact_check_post_mapping)

    def __getitem__(self, idx):
        fc_id, p_id = self.fact_check_post_mapping[idx]
        return self.id_to_documents[fc_id], self.id_to_post[p_id]
    
    def get_documents(self) -> Any:
        return self.id_to_documents.items()
        
    def get_document(self, doc_id: int) -> Any:
        return self.id_to_documents[doc_id]
    
    def get_documents_texts(self) -> List[str]:
        return list(self.id_to_documents.values())
    
    def get_documents_ids(self) -> List[str]:
        return list(map(str, self.id_to_documents.keys()))
    
    def map_topK(self, topK: List[int]) -> List[int]:
        """
        Maps the topK to the new ids
        """
        if len(self.id_to_documents) == 0:
            return []

        ids_list = list(self.id_to_documents.keys())
        new_topK = [ids_list[idx] for idx in topK]

        return new_topK

    def clean_text(self, text):

        if self.remove_urls:
            text = cleaning.remove_urls(text)

        if self.remove_emojis:
            text = cleaning.remove_emojis(text)

        if self.replace_whitespaces:
            text = cleaning.replace_whitespaces(text)

        if self.clean_twitter:
            text = cleaning.clean_twitter_picture_links(text)
            text = cleaning.clean_twitter_links(text)

        if self.remove_elongation:
            text = cleaning.remove_elongation(text)

        return text.strip()
    
    def maybe_clean_ocr(self, ocr):
        if self.clean_ocr:
            return cleaning.clean_ocr(ocr)
        return ocr
    
    def __getattr__(self, name):
        if name in {'id_to_documents', 'id_to_post', 'fact_check_post_mapping'}:
            raise AttributeError(
                f"You have to `load` the dataset first before using '{name}'")
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def load(self):
        raise NotImplementedError