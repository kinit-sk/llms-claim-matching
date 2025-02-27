from nltk import sent_tokenize

def chunk_text(text: str, chunk_size: int, overlap:int = 0) -> list:
    """
    Split text into chunks of a specific size.
    
    Args:
        text: text to be split
        chunk_size: size of the chunk
        overlap: number of sentences that should overlap between two consecutive chunks
        
    Returns:
        list: list of documents
    """
    documents = []
    sentences = sent_tokenize(text)
    
    for i in range(0, len(sentences), chunk_size - overlap):
        documents.append(' '.join(sentences[i:i + chunk_size]))
        
    return documents
    