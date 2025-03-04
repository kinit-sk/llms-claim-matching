import argparse
import random
from tqdm import tqdm
import pandas as pd
from src.prompts.fewshot_strategy import CosineSimilarity

def main():
    # Prepare data for selecting few-shot examples
    df_original = pd.read_csv('./datasets/annotation_pairs.csv')
    df_anotated = pd.read_csv('./annotated_pairs.csv')
    indexes_to_drop = []
    
    # Remove rows that are included in the AMC-16K dataset
    for index, row in df_anotated.iterrows():
        count = df_original[(df_original['post_id'] == row['DocumentID']) & (df_original['factcheck_id'] == row['FactCheckID'])].shape[0]
        if count != 0:
            indexes_to_drop.append(index)
    df_anotated_copy = df_anotated.copy()
    df_anotated_copy.drop(indexes_to_drop, inplace=True)
    df_anotated_copy.to_csv('./datasets/annotated_pairs_fewshot.csv', index=False)
    
    for english in [False, True]:
        few_shoter = CosineSimilarity(num_samples_per_class=10, path='./datasets/annotated_pairs_fewshot.csv', english=english)
        
        post_text = 'post_text_en' if english else 'post_text'
        fact_check_text = 'fact_check_text_en' if english else 'fact_check_text'
        df = pd.read_csv('./annotation_pairs.csv')
        
        output_path = './fewshot_examples-en.csv' if english else './fewshot_examples.csv'
        
        results_df = pd.DataFrame(columns=['post_id', 'factcheck_id', 'post_text', 'post_text_en', 'factcheck_text', 'factcheck_text_en', 'post_language', 'examples'])
        for index, row in tqdm(df.iterrows()):
            post = row[post_text]
            fact_check = row[fact_check_text]
            examples = few_shoter.sample(post=post, fact_check=fact_check)
            
            results_df = pd.concat([results_df, pd.DataFrame({
                'post_id': row['post_id'],
                'factcheck_id': row['factcheck_id'],
                'post_text': row['post_text'],
                'post_text_en': row['post_text_en'],
                'factcheck_text': row['fact_check_text'],
                'factcheck_text_en': row['fact_check_text_en'],
                'post_language': row['post_language'],
                'examples': [examples]
            })])

            if index % 100 == 0:
                results_df.to_csv(output_path, index=False)
            
        results_df.to_csv(output_path, index=False)
        
    df = pd.read_csv("./fewshot_examples.csv")
    df['examples'] = df['examples'].apply(eval)
    df['random_examples_ids'] = df['examples'].apply(lambda x: random.sample(range(10), 10))
    df['random_examples'] = df.apply(lambda x: [x['examples'][i] for i in x['random_examples_ids']], axis=1)
    df.to_csv('./fewshot_examples-random.csv', index=False)


    df_en = pd.read_csv("./fewshot_examples-en.csv")
    df_en['examples'] = df_en['examples'].apply(eval)
    df_en['random_examples_ids'] = df['random_examples_ids']
    df_en['random_examples'] = df_en.apply(lambda x: [x['examples'][i] for i in x['random_examples_ids']], axis=1)
    df_en.to_csv('./fewshot_examples-random-en.csv', index=False)

    
if __name__ == '__main__':
    main()
