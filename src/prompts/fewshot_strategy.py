import pandas as pd
import numpy as np
from typing import Any
from src.retrievers import retriever_factory


def fewshot_factory(name: str, **kwargs: Any) -> Any:
    strategy = {
        'random': RandomStrategy,
        'cosine': CosineSimilarity,
    }[name]
    return strategy(**kwargs)


class RandomStrategy:
    def __init__(self, num_samples_per_class: int = 1, path: str = None, english: bool = False):
        self.num_samples_per_class = num_samples_per_class
        self.path = path
        self.english = english
        self.df = None
        
    def load(self):
        self.df = pd.read_csv(self.path)
        
    def sample(self, post: str, fact_check: str):
        if self.df is None:
            self.load()
        positive_examples = self.df[self.df['Rating'] == 'Yes'].sample(n=self.num_samples_per_class)
        negative_examples = self.df[self.df['Rating'] == 'No'].sample(n=self.num_samples_per_class)
        
        post_text = 'PostText_en' if self.english else 'PostText'
        fact_check_text = 'FactCheckText_en' if self.english else 'FactCheckText'
        
        positive = [
            (row[post_text], row[fact_check_text], row['Rating'])
            for _, row in positive_examples.iterrows()
        ]
        
        negative = [
            (row[post_text], row[fact_check_text], row['Rating'])
            for _, row in negative_examples.iterrows()
        ]
        
        l = []
        for p, n in zip(positive, negative):
            l.append(p)
            l.append(n)
        return l
    

class FewShotDataset:
    def __init__(self, path: str = None, english: bool = False, type: str = 'fact_check'):
        self.path = path
        self.english = english
        self.df = None
        self.type = type
        
    def load(self):
        self.df = pd.read_csv(self.path)
        
    def return_df(self):
        if self.df is None:
            self.load()
        return self.df
        
    def get_documents_texts(self):
        if self.df is None:
            self.load()
        
        if self.type == 'fact_check':
            return self.df['FactCheckText_en'].to_list() if self.english else self.df['FactCheckText'].to_list()
        else:
            return self.df['PostText_en'].to_list() if self.english else self.df['PostText'].to_list()


class CosineSimilarity:
    def __init__(self, num_samples_per_class: int = 1, path: str = None, english: bool = False):
        self.num_samples_per_class = num_samples_per_class
        self.path = path
        self.english = english
        self.df = None
        self.retriever = retriever_factory('embedding', model_name='intfloat/multilingual-e5-large', cache='./cache/multilingual-e5-large', top_k=None)
        
    def load(self):
        pass
    
    def sample(self, post: str, fact_check: str):
        fact_checks = FewShotDataset(path=self.path, english=self.english, type='fact_check')
        posts = FewShotDataset(path=self.path, english=self.english, type='post')
        self.df = fact_checks.return_df()
        post_text = 'PostText_en' if self.english else 'PostText'
        fact_check_text = 'FactCheckText_en' if self.english else 'FactCheckText'
        
        self.retriever.set_knowledge_base(posts)
        post_sims = self.retriever.retrieve_sims(post)
        
        self.retriever.set_knowledge_base(fact_checks)
        fact_check_sims = self.retriever.retrieve_sims(fact_check)

        post_sims = np.array(post_sims)
        fact_check_sims = np.array(fact_check_sims)
        final_sims = post_sims * fact_check_sims
        
        sorted_indexes = np.argsort(final_sims, axis=0)
        positives = []
        negatives = []
        
        for i in sorted_indexes.tolist():
            if self.df.iloc[i]['Rating'] == 'Yes' and len(positives) < self.num_samples_per_class:
                positives.append((self.df.iloc[i][post_text], self.df.iloc[i][fact_check_text], self.df.iloc[i]['Rating']))
            if self.df.iloc[i]['Rating'] == 'No' and len(negatives) < self.num_samples_per_class:
                negatives.append((self.df.iloc[i][post_text], self.df.iloc[i][fact_check_text], self.df.iloc[i]['Rating']))
                
            if len(positives) == self.num_samples_per_class and len(negatives) == self.num_samples_per_class:
                break
        
        l = []
        for p, n in zip(positives, negatives):
            l.append(p)
            l.append(n)
        return l
