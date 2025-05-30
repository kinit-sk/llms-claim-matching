import argparse
import logging
import os
import pandas as pd
import yaml
from tqdm import tqdm

from src.prompts import prompt_factory
from src.models import model_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['WANDB_MODE'] = 'disabled'

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument(
        '--config', type=str, default='./configs/dummy-config.yaml', help='Model name')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--english', action='store_true', help='Use English model', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    if args.english:
        config['steps'][1]['prompt']['examples_path'] = './datasets/fewshot_examples-random-en.csv'
        config['steps'][1]['prompt']['english'] = True
    
    df_pairs = pd.read_csv('./datasets/annotation_pairs.csv')
    df_pairs = df_pairs
    
    rest = {
        k: v for k, v in config['steps'][1]['prompt'].items() if k not in ['type']
    }
    prompt = prompt_factory(config['steps'][1]['prompt']['type'], **rest)
    
    kwargs = {
        k: v for k, v in config['steps'][2]['llm'].items() if k not in ['model_name', 'model']
    }
    
    model = model_factory(config['steps'][2]['llm']['model'], name=config['steps'][2]['llm']['model_name'], **kwargs)
    
    os.makedirs(f'./results/', exist_ok=True)
    csv_path = f'./results/{args.output_dir}.csv' if args.output_dir is not None else None

    df_results = pd.DataFrame(columns=['Prompt', 'PostID', 'FactCheckID', 'PostText', 'FactCheckText', 'Prediction', 'GroundTruth', 'YesProb', 'NoProb'])

    for index, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        if args.english:
            query = row['post_text_en']
            factchecktext = row['fact_check_text_en']
        else:
            query = row['post_text']
            factchecktext = row['fact_check_text']

        post_id = row['post_id']
        factcheck_id = row['factcheck_id']

        created_prompt = prompt(query=query, documents=[factchecktext])
        
        model_output = model(prompt=created_prompt['prompt'])
        generated_text = model_output['output']
        yes_probs = model_output['yes_probs']
        no_probs = model_output['no_probs']
        df_results = pd.concat([df_results, pd.DataFrame([[created_prompt['prompt'][0], post_id, factcheck_id, query, factchecktext, generated_text, row['rating'], yes_probs, no_probs]], columns=['Prompt', 'PostID', 'FactCheckID', 'PostText', 'FactCheckText', 'Prediction', 'GroundTruth', 'YesProb', 'NoProb'])])
        
        df_results.to_csv(csv_path, index=False)
                    
    df_results.to_csv(csv_path, index=False)
