"""
Library of custom types used for datasets. Some basic functions over these types are also included.
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

Id2FactCheck = Dict[int, str]
Id2Post = Dict[int, str]
FactCheckPostMapping = List[Tuple[int, int]]

Language = str  # ISO 639-3 language code
LanguageDistribution = Dict[Language, float]  # Confidence for language identification. Should sum to [0, 1].

OriginalText = str
EnglishTranslation = str
TranslatedText = Tuple[OriginalText, EnglishTranslation, LanguageDistribution]

Instance = Tuple[Optional[datetime], str]  # When and where was a fact-check or post published


def is_in_distribution(language: Language, distribution: LanguageDistribution, threshold: float = 0.2) -> bool:
    """
    Check whether `language` is in a `distribution` with more than `treshold` x 100%
    """
    return next(
        (
            percentage >= threshold
            for distribution_language, percentage in distribution
            if distribution_language == language
        ),
        False
    )


def combine_distributions(texts: Iterable[TranslatedText]) -> LanguageDistribution:
    """
    Combine `LanguageDistribution`s from multiple `TranslatedText`s taking the length of the text into consideration.
    """
    total_length = sum(len(text[0]) for text in texts)
    distribution = defaultdict(lambda: 0)
    for original_text, _, text_distribution in texts:
        for language, percentage in text_distribution:
            distribution[language] += percentage * len(original_text) / total_length
    return list(distribution.items())
