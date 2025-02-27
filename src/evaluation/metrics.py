"""
A library of metric-calculating functions. The default result format is Iterable[List[int]] -- an iterable of results for individual queries (posts).
Each query has a list of ranks assigned, representing the ranks of appropriate documents (fact-checks).
"""
import math
import random
import statistics
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import norm


def binary_ci(success: int, total: int, alpha: float = 0.95) -> Tuple[float, float, float]:
    """
    Using Agresti-Coull interval

    Return mean and confidence interval (lower and upper bound)
    """
    z = statistics.NormalDist().inv_cdf((1 + alpha) / 2)
    total = total + z**2
    loc = (success + (z**2) / 2) / total
    diameter = z * math.sqrt(loc * (1 - loc) / total)
    return loc, loc - diameter, loc + diameter


def bootstrap_ci(scores, alpha=0.95) -> Tuple[float, float, float]:
    """
    Bootstrapping based estimate.

    Return mean and confidence interval (lower and upper bound)
    """
    loc, scale = norm.fit(scores)
    bootstrap = [sum(random.choices(scores, k=len(scores))) /
                 len(scores) for _ in range(1000)]
    lower, upper = norm.interval(alpha, *norm.fit(bootstrap))

    return loc, lower, upper


def pair_success_at_k(ranks, k=10):
    """
    Pair S@K - How many fact-check-post pairs from all the pairs ended up in the top K.
    """
    values = [rank <= k for query in ranks for rank in query]
    return binary_ci(sum(values), len(values))


def post_success_at_k(ranks, k=10):
    """
    Post S@K - For how many posts at least one pair ended up in the top K.
    """
    values = [any(rank <= k for rank in query) for query in ranks]
    return binary_ci(sum(values), len(values))


def precision_at_k(ranks, k=10):
    """
    P@K - How many positive hits in the top K
    """
    values = [sum(rank <= k for rank in query) for query in ranks]
    return binary_ci(sum(values), len(values) * k)


def mrr(ranks):
    """
    Mean Reciprocal Rank: 1/r for r in ranks
    """
    values = [1 / min(query) for query in ranks]
    return bootstrap_ci(values)


def map_(ranks):
    """
    Mean Average Precision: As defined here page 7: https://datascience-intro.github.io/1MS041-2022/Files/AveragePrecision.pdf
    """
    values = [
        np.mean([
            (i + 1) / rank
            for i, rank in enumerate(sorted(query))
        ])
        for query in ranks
    ]
    return bootstrap_ci(values)


def map_k(ranks, k=5):
    values = []
    for query in ranks:
        ap_at_k = 0
        num_correct = 0
        for i, rank in enumerate(query):
            if rank <= k:
                num_correct += 1
                ap_at_k += num_correct / (i+1)
        values.append(ap_at_k)
    return bootstrap_ci(values)

def irrelevant_removed_counts(relevant_docs, previously_predicted_docs, predicted_docs):
    total_irrelevant_count_removed = 0
    total_relevant_count_removed = 0
    # count number of docs in relevant_docs list 
    total_relevant_count = sum([len(docs) for docs in relevant_docs])
    total_irrelevant_count = sum([len(docs) for docs in previously_predicted_docs]) - total_relevant_count
    
    irrelevant_count_removed_list = []
    relevant_count_removed_list = []
    
    for relevant, prev_predicted, predicted in zip(relevant_docs, previously_predicted_docs, predicted_docs):
        irrelevant_count_removed = 0
        relevant_count_removed = 0
        for doc in relevant:
            if doc not in predicted and doc in prev_predicted:
                relevant_count_removed += 1
                
        for doc in prev_predicted:
            if doc not in predicted and doc not in relevant:
                irrelevant_count_removed += 1
                
        total_irrelevant_count_removed += irrelevant_count_removed
        total_relevant_count_removed += relevant_count_removed
        irrelevant_count_removed_list.append(irrelevant_count_removed)
        relevant_count_removed_list.append(relevant_count_removed)
                
    return total_irrelevant_count_removed, total_relevant_count_removed, irrelevant_count_removed_list, relevant_count_removed_list, total_relevant_count, total_irrelevant_count  



def standard_metrics(ranks: Iterable[List[int]]) -> Dict[str, float]:
    """
    Calculate several metrics and their CIs based on the ranks provided

    Attributes:
        ranks - Iterable of results for individual queries. For each query a list of ranks is expected.
    """

    return {
        'pair_success_at_10': pair_success_at_k(ranks),
        'post_success_at_10': post_success_at_k(ranks),
        'precision_at_10': precision_at_k(ranks),
        'mrr': mrr(ranks),
        'map': map_(ranks),
        'map_5': map_k(ranks),
    }
