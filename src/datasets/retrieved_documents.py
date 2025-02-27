from typing import List
from src.datasets.dataset import Dataset

class RetrievedDocuments(Dataset):
    """
    A dataset that contains a list of documents and their corresponding ids.
    
    Args:
        name (str): The name of the dataset.
        documents (List[str]): The list of documents.
        ids (List[int]): The list of ids.
        
    Returns:
        RetrievedDocuments: An instance of the RetrievedDocuments class.
    """
    def __init__(self, name: str, documents: List[str], ids: List[int], **kwargs):
        super().__init__(**kwargs)
        self.id_to_documents = dict(zip(ids, documents))

    def load(self) -> 'RetrievedDocuments':
        return self
            