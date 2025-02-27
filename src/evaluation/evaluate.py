from collections import defaultdict
import logging
from typing import Generator

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import os

from src.evaluation.metrics import standard_metrics, irrelevant_removed_counts
from src.evaluation.utils import find_fact_check_ids


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predicted_ranks(predicted_ids: np.array, desired_ids: np.array, default_rank: int = None):
    """
    Return sorted ranks of the `desired_ids` in the `predicted_ids` array.

    If `default_rank` is set, the final array will be padded with the value for all the ids that were not present in the `predicted_ids` array.
    """

    predicted_ranks = dict()

    for desired in desired_ids:

        try:
            # +1 so that the first item has rank 1, not 0
            rank = np.where(predicted_ids == desired)[0][0] + 1
        except IndexError:
            rank = default_rank

        if rank is not None:
            predicted_ranks[desired] = rank

    return predicted_ranks


def process_results(gen: Generator, default_rank: int = None, csv_path: str = None):
    """
    Take the results generated from `gen` and process them. By default, only calculate metrics, but dumping the results into a csv file is also supported
    via `csv_path` attribute. For `default_rank` see `predicted_ranks` function.
    """

    ranks = list()
    rows = list()

    for predicted_ids, desired_ids, post_id in gen:
        post_ranks = predicted_ranks(predicted_ids, desired_ids, default_rank)
        ranks.append(post_ranks.values())

        if csv_path:
            rows.append((post_id, post_ranks, predicted_ids[:100]))

    logger.info(f'{sum(len(query) for query in ranks)} ranks produced.')

    if csv_path:
        pd.DataFrame(rows, columns=['post_id', 'desired_fact_check_ranks',
                     'predicted_fact_check_ids']).to_csv(csv_path, index=False)
        artifact = wandb.Artifact(wandb.run.name, type='dataset')
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)

    metrics = standard_metrics(ranks)
    wandb.log(metrics)

    return metrics


def evaluate_post_fact_check_pairs(gen, dataset):
    """
    Evaluate the performance of the model on the given data generator.
    """

    desired_fact_check_ids = defaultdict(lambda: list())
    for fact_check_id, post_id in dataset.fact_check_post_mapping:
        desired_fact_check_ids[post_id].append(fact_check_id)

    logging.info(f'{len(desired_fact_check_ids)} posts to evaluate.')

    for predicted_fact_check_ids, post_id in gen:
        yield predicted_fact_check_ids, desired_fact_check_ids[post_id], post_id


def increase_in_order(relevant_docs, previously_predicted_docs, predicted_docs, top_k=10):
    total_shifts = []
    increase_number = 0
    increased_possible = 0
    removed_number = 0
    total_to_top_k = 0
    increased_possible_top_k = 0
    
    for relevant, prev_predicted, predicted in zip(relevant_docs, previously_predicted_docs, predicted_docs):
        relevant = np.array(relevant)
        prev_predicted = np.array(prev_predicted)
        predicted = np.array(predicted)

        previous_ranks = predicted_ranks(prev_predicted, relevant, default_rank=100)
        current_ranks = predicted_ranks(predicted, relevant, default_rank=100)
        
        # identify the shift value
        shifts = []
        for doc in relevant:
            shifts.append(previous_ranks[doc] - current_ranks[doc])
            if previous_ranks[doc] != 1:
                increased_possible += 1
            if previous_ranks[doc] > top_k:
                increased_possible_top_k +=1
            if previous_ranks[doc] > current_ranks[doc]:
                increase_number += 1
                if previous_ranks[doc] > top_k and current_ranks[doc] <= top_k:
                    total_to_top_k += 1
            if previous_ranks[doc] != 100 and current_ranks[doc] == 100:
                removed_number += 1
                
        total_shifts.append(shifts)
        
    return increase_number, increased_possible, total_shifts, removed_number, total_to_top_k, increased_possible_top_k


def advanced_metrics(relevant_docs, previously_predicted_docs, predicted_docs):
    
    total_irrelevant_count_removed, total_relevant_count_removed, _, _, total_relevant_count, total_irrelevant_count = irrelevant_removed_counts(relevant_docs, previously_predicted_docs, predicted_docs)

    irrelevant_percentage = total_irrelevant_count_removed / total_irrelevant_count
    relevant_percentage = total_relevant_count_removed / total_relevant_count
    
    increase_number, increased_possible, _, _, total_to_top_k, increased_possible_top_k = increase_in_order(relevant_docs, previously_predicted_docs, predicted_docs)    
    
    return {
        'irrelevant_percentage': irrelevant_percentage,
        'relevant_percentage': relevant_percentage,
        'increased': increase_number,
        'increased_percentage': increase_number / increased_possible,
        'to_top_k': total_to_top_k,
        'top_k_percentage': total_to_top_k / increased_possible_top_k,   
    }


def evaluate_multiclaim(dataset, pipeline, csv_path=None, language=None):
    df = pd.DataFrame(columns=['post_id', 'fact_check_ids', 'generated_output', 'yes_prob', 'no_prob'])
    id_to_post = dataset.id_to_post

    for post_id, post in tqdm(id_to_post.items()):

        original_output, documents, yes_probs, no_probs = pipeline(query=post)
        if len(documents) == 0 or isinstance(documents[0], int):
            fact_check_ids = documents
        else:
            fact_check_ids = find_fact_check_ids(
                dataset, documents, pipeline.retrieved_documents)

        if csv_path is not None:
            df = pd.concat([df, pd.DataFrame(
                [[post_id, fact_check_ids, original_output, yes_probs, no_probs]],
                columns=['post_id', 'fact_check_ids', 'generated_output', 'yes_prob', 'no_prob'])])
            df.to_csv(csv_path, index=False)

        yield np.array(fact_check_ids), post_id