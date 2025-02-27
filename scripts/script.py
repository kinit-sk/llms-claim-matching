import argparse
import logging
import wandb
import os
import yaml

from src.datasets import dataset_factory
from src.evaluation.evaluate import evaluate_post_fact_check_pairs, evaluate_multiclaim, process_results
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['WANDB_MODE'] = 'disabled'

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument(
        '--config', type=str, default='./configs/dummy-config.yaml', help='Model name')
    parser.add_argument('--language', type=str, default=None, help='Language')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--wandb_name', type=str, default='test_name')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    language = args.language

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    config['steps'][0]['retriever']['knowledge_base']['language'] = language

    pipeline = Pipeline(rag_config=config)
    dataset = dataset_factory('multiclaim', language=language).load()
    
    os.makedirs(f'./results/{args.output_dir}', exist_ok=True)
    csv_path = f'./results/{args.output_dir}/{language}.csv' if args.output_dir is not None else None

    print(language)
    generator = evaluate_post_fact_check_pairs(
        evaluate_multiclaim(
            dataset, 
            pipeline, 
            language=language,
            csv_path=csv_path
        ),
        dataset
    )

    output_path = f'./results/{args.output_dir}/{language}-results.csv'
    wandb_name = f'{args.wandb_name}-{language}'

    with wandb.init(project='disai-llm', name=wandb_name, config=args) as wandb_run:
        for k, v in wandb_run.config.items():
            if isinstance(v, list):
                wandb_run.config.update(
                    {k: ' '.join(v)},
                    allow_val_change=True,
                )

        results = process_results(
            generator, default_rank=100, csv_path=output_path)

        print(results)
