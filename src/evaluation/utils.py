import pandas as pd
import os


def find_fact_check_ids(dataset, documents, retrieved_documents):
    # remove the ids from documents that are out of index
    documents = [doc for doc in documents if int(
        doc) <= len(retrieved_documents) and int(doc) > 0]

    documents = [retrieved_documents[int(doc) - 1] for doc in documents]

    returned_ids = []
    for document in documents:
        for fact_check_id, text in dataset.id_to_documents.items():
            if text == document:
                returned_ids.append(fact_check_id)

    return returned_ids
