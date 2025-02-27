import ast
import logging
import os
import random
from os.path import join as join_path
from typing import Iterable, Optional

import pandas as pd

from src.datasets.custom_types import Language, is_in_distribution, combine_distributions
from src.datasets.dataset import Dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiClaimDataset(Dataset):
    """MultiClaim dataset

    Class for multiclaim dataset that can load different variants. It requires data from `data/datasets/multiclaim`.

    Initialization Attributes:
        crosslingual: bool  If `True`, only crosslingual pairs (fact-check and post in different languages) are loaded.
        fact_check_fields: Iterable[str]  List of fields used to generate the final `str` representation for fact-checks. Supports `claim` and `title`.
        fact_check_language: Optional[Language]  If a `Language` is specified, only fact-checks with that language are selected.
        language: Optional[Language]  If a `Language` is specified, only fact-checks and posts with that language are selected.
        post_language: Optional[Language]  If a `Language` is specified, only posts with that language are selected.
        split: `train`, `test` or `dev`. `None` means all the samples.
        version: 'original' or 'english'. Language version of the dataset.

        Also check `Dataset` attributes.

        After `load` is called, following attributes are accesible:
            fact_check_post_mapping: list[tuple[int, int]]  List of Factcheck-Post id pairs.
            id_to_documents: dict[int, str]  Factcheck id -> Factcheck text
            id_to_post: dict[int, str]  Post id -> Post text


    Methods:
        load: Loads the data from the csv files. Populates `id_to_documents`, `id_to_post` and `fact_check_post_mapping` attributes.
    """

    our_dataset_path = join_path('.','datasets', 'multiclaim')
    csvs_loaded = False

    def __init__(
        self,
        crosslingual: bool = False,
        fact_check_fields: Iterable[str] = ('claim', ),
        fact_check_language: Optional[Language] = None,
        language: Optional[Language] = None,
        post_language: Optional[Language] = None,
        split: Optional[str] = None,
        version: str = 'original',
        **kwargs
    ):
        super().__init__(**kwargs)

        assert all(field in ('claim', 'title') for field in fact_check_fields)
        assert split in (None, 'dev', 'test', 'train')
        assert version in ('english', 'original')

        self.crosslingual = crosslingual
        self.fact_check_fields = fact_check_fields
        self.fact_check_language = fact_check_language
        self.language = language
        self.post_language = post_language
        self.split = split
        self.version = version

    @classmethod
    def maybe_load_csvs(cls):
        """
        Load the csvs and store them as class variables. When individual objects are initialized, they can reuse the same
        pre-loaded dataframes without costly text parsing.

        `OurDataset.csvs_loaded` is a flag indicating whether the csvs are already loaded.
        """

        if cls.csvs_loaded:
            return

        posts_path = join_path(cls.our_dataset_path, 'posts.csv')
        fact_checks_path = join_path(cls.our_dataset_path, 'fact_checks.csv')
        fact_check_post_mapping_path = join_path(
            cls.our_dataset_path, 'fact_check_post_mapping.csv')

        for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
            assert os.path.isfile(path)

        # We need to apply t = t.replace('\n', '\\n') for text fields before using `ast.literal_eval`.
        # `ast.literal_eval` has problems when there are new lines in the text, e.g.:
        # `ast.literal_eval('("\n")')` effectively tries to interpret the following code:

        # ```
        # ("
        # ")
        # ```

        # This raises a SyntaxError exception. By escaping new lines we are able to force it to interpret it properly. There might
        # be some other way to do this more systematically, but it is a workable fix for now.

        def parse_col(s): return ast.literal_eval(
            s.replace('\n', '\\n')) if s else s

        logger.info('Loading fact-checks.')
        cls.df_fact_checks = pd.read_csv(fact_checks_path).fillna(
            '').set_index('fact_check_id')
        for col in ['claim', 'instances', 'title']:
            cls.df_fact_checks[col] = cls.df_fact_checks[col].apply(parse_col)
        logger.info(f'{len(cls.df_fact_checks)} loaded.')

        logger.info('Loading posts.')
        cls.df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
        for col in ['instances', 'ocr', 'verdicts', 'text']:
            cls.df_posts[col] = cls.df_posts[col].apply(parse_col)
        logger.info(f'{len(cls.df_posts)} loaded.')

        logger.info('Loading fact-check-post mapping.')
        cls.df_fact_check_post_mapping = pd.read_csv(
            fact_check_post_mapping_path)
        logger.info(f'{len(cls.df_fact_check_post_mapping)} loaded.')

        cls.csvs_loaded = True

    def load(self):

        self.maybe_load_csvs()

        df_posts = self.df_posts.copy()
        df_fact_checks = self.df_fact_checks.copy()
        df_fact_check_post_mapping = self.df_fact_check_post_mapping.copy()

        if self.split:
            split_post_ids = set(self.split_post_ids(self.split))
            df_posts = df_posts[df_posts.index.isin(split_post_ids)]
            logger.info(f'Filtering by split: {len(df_posts)} posts remaining.')  # nopep8

        # Filter fact-checks by the language detected in claim
        if self.language or self.fact_check_language:
            if isinstance(self.fact_check_language, str) or isinstance(self.language, str):
                df_fact_checks = df_fact_checks[df_fact_checks['claim'].apply(
                    # claim[2] is the language distribution
                    lambda claim: is_in_distribution(
                        self.language or self.fact_check_language, claim[2])
                )]
                logger.info(
                    f'Filtering fact-checks by language: {len(df_fact_checks)} posts remaining.')
            elif isinstance(self.fact_check_language, list):
                print(self.fact_check_language)
                df_fact_checks = df_fact_checks[df_fact_checks['claim'].apply(
                    lambda claim: any(
                        is_in_distribution(language, claim[2])
                        for language in self.fact_check_language
                    )
                )]
                logger.info(
                    f'Filtering fact-checks by language: {len(df_fact_checks)} posts remaining.')

        # Filter posts by the language detected in the combined distribution.
        # There was a slight bug in the paper version of post language filtering and in effect we have slightly more posts per language
        # in the paper. The original version did not take into account that the sum of percentages in a distribution does not have to be equal to 1.
        if self.language or self.post_language:
            def post_language_filter(row):
                texts = [
                    text
                    for text in [row['text']] + row['ocr']
                    if text  # Filter empty texts
                ]
                distribution = combine_distributions(texts)
                return is_in_distribution(self.language or self.post_language, distribution)

            df_posts = df_posts[df_posts.apply(post_language_filter, axis=1)]
            logger.info(f'Filtering posts by language: {len(df_posts)} posts remaining.')  # nopep8

        # Create mapping variable
        post_ids = set(df_posts.index)
        fact_check_ids = set(df_fact_checks.index)
        fact_check_post_mapping = set(
            (fact_check_id, post_id)
            for fact_check_id, post_id in df_fact_check_post_mapping.itertuples(index=False, name=None)
            if fact_check_id in fact_check_ids and post_id in post_ids
        )
        logger.info(f'Mappings remaining: {len(fact_check_post_mapping)}.')

        # Leave only crosslingual samples
        if self.crosslingual:

            crosslingual_mapping = set()
            for fact_check_id, post_id in fact_check_post_mapping:

                # Here, we assume that all fact-check claims have exactly one language assigned (via Google Translate)
                # We will have to rework this part if we want to support multiple detected languages.
                fact_check_language = df_fact_checks.loc[fact_check_id,
                                                         'claim'][2][0][0]

                post_texts = [
                    text
                    for text in [df_posts.loc[post_id, 'text']] + df_posts.loc[post_id, 'ocr']
                    if text
                ]
                post_distribution = combine_distributions(post_texts)

                if not is_in_distribution(fact_check_language, post_distribution):
                    crosslingual_mapping.add((fact_check_id, post_id))

            fact_check_post_mapping = crosslingual_mapping
            logger.info(f'Crosslingual mappings remaining: {len(fact_check_post_mapping)}')  # nopep8

        # Filtering posts if any crosslingual or language filter were applied
        remaining_post_ids = set(
            post_id for _, post_id in fact_check_post_mapping)
        df_posts = df_posts[df_posts.index.isin(remaining_post_ids)]
        logger.info(f'Filtering posts.')
        logger.info(f'Posts remaining: {len(df_posts)}')

        # Create object attributes
        self.fact_check_post_mapping = list(fact_check_post_mapping)

        # Create final `str` representations
        self.id_to_documents = {
            # `self.version == 'english'` works like an index here. On False, it will return claim[0], on True, claim[1].
            fact_check_id: self.clean_text(claim[self.version == 'english'])
            for fact_check_id, claim in zip(df_fact_checks.index, df_fact_checks['claim'])
        }

        self.id_to_post = dict()
        for post_id, post_text, ocr in zip(df_posts.index, df_posts['text'], df_posts['ocr']):
            texts = list()
            if post_text:
                texts.append(post_text[self.version == 'english'])
            for ocr_text in ocr:
                texts.append(self.maybe_clean_ocr(
                    ocr_text[self.version == 'english']))
            self.id_to_post[post_id] = self.clean_text(' '.join(texts))

        return self

    @staticmethod
    def split_post_ids(split):
        rnd = random.Random(1)
        # This split only works for the particular dataset version with this number of posts.
        post_ids = list(range(28092))
        rnd.shuffle(post_ids)

        train_size = 0.8
        dev_size = 0.1

        train_end = int(len(post_ids) * train_size)
        dev_end = train_end + int(len(post_ids) * dev_size)

        return {
            'train': post_ids[:train_end],
            'dev':   post_ids[train_end:dev_end],
            'test':  post_ids[dev_end:],
        }[split]
