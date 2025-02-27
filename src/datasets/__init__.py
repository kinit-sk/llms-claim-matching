from src.datasets.dataset import Dataset
from src.datasets.multiclaim.multiclaim_dataset import MultiClaimDataset

def dataset_factory(name, **kwargs) -> Dataset:
    """
    Factory function to create a dataset object.
    
    Args:
        name (str): The name of the dataset.
        
    Returns:
        Dataset: An instance of the dataset.
    """
    dataset = {
        'multiclaim': MultiClaimDataset,
    }[name](**kwargs)
    
    return dataset