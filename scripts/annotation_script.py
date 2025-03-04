import argparse
import logging
import pandas as pd
import os
import yaml
from tqdm import tqdm

import sys
sys.path.append("rag")

from src.datasets import dataset_factory
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['WANDB_MODE'] = 'disabled'

if __name__ == '__main__':
    languages = ['spa', 'eng', 'por', 'fra', 'msa', 'deu', 'ara', 'tha', 'hbs', 'kor', 'pol', 'slk', 'nld', 'ron', 'ell', 'ces', 'bul', 'hun', 'hin', 'mya']
    
    config_path = './configs/e5-config.yaml'
    
    for language in languages:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        temp_languages = languages.copy()
        temp_languages.remove(language)
        
        # Monolingual pairs
        config['steps'][0]['retriever']['knowledge_base']['language'] = language

        pipeline = Pipeline(rag_config=config)
        dataset = dataset_factory('multiclaim', language=language).load()
        
        os.makedirs(f'./results/annotations-monolingual/', exist_ok=True)
        csv_path = f'./results/annotations-monolingual/{language}.csv'
        
        df = pd.DataFrame(columns=['post_id', 'fact_check_ids', 'post_text', 'fact_check_claims'])
        id_to_post = dataset.id_to_post
        
        for post_id, post in tqdm(list(id_to_post.items())[:10]):
            original_output, documents, _, _ = pipeline(query=post)
            
            df = pd.concat([df, pd.DataFrame(
                [[post_id, documents, post, original_output]],
                columns=['post_id', 'fact_check_ids', 'post_text', 'fact_check_claims'])])
            
        df.to_csv(csv_path, index=False)
        
        # Cross-lingual pairs
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        temp_languages = languages.copy()
        temp_languages.remove(language)
            
        config['steps'][0]['retriever']['knowledge_base']['fact_check_language'] = temp_languages
        config['steps'][0]['retriever']['knowledge_base']['post_language'] = language
        
        pipeline = Pipeline(rag_config=config)
        
        os.makedirs(f'./results/annotations-crosslingual/', exist_ok=True)
        csv_path = f'./results/annotations-crosslingual/{language}.csv'
        
        df = pd.DataFrame(columns=['post_id', 'fact_check_ids', 'post_text', 'fact_check_claims'])
        
        for post_id, post in tqdm(id_to_post.items()):
            original_output, documents, _, _ = pipeline(query=post)
            
            df = pd.concat([df, pd.DataFrame(
                [[post_id, documents, post, original_output]],
                columns=['post_id', 'fact_check_ids', 'post_text', 'fact_check_claims'])])
            
        df.to_csv(csv_path, index=False)    
    