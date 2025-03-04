import ast
import pandas as pd
from tqdm import tqdm
from src.datasets import dataset_factory

parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

df = pd.read_csv('./datasets/fewshot-random.csv')
df_factchecks = pd.read_csv('./datasets/multiclaim/fact_checks.csv').fillna('')
for col in ['claim', 'instances', 'title']:
    df_factchecks[col] = df_factchecks[col].apply(parse_col)
    
dataset = dataset_factory('multiclaim').load()
dataset_en = dataset_factory('multiclaim', version="english").load()

columns_to_add = [
    'post_text',
    'post_text_en',
    'factcheck_text',
    'factcheck_text_en',
    'random_examples',
]
for column in columns_to_add:
    df[column] = ''
df['random_examples_with_ids'] = df['random_examples_with_ids'].apply(eval)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    examples = row['random_examples_with_ids']
    post_text = dataset.id_to_post[row['post_id']]
    post_text_en = dataset_en.id_to_post[row['post_id']]
    
    fact_check_row = df_factchecks[df_factchecks['fact_check_id'] == row['factcheck_id']].iloc[0]
    claim = fact_check_row['claim']
    claim_og = claim[0]
    claim_en = claim[1]
    
    df.at[index, 'post_text'] = post_text
    df.at[index, 'post_text_en'] = post_text_en
    df.at[index, 'factcheck_text'] = claim_og
    df.at[index, 'factcheck_text_en'] = claim_en
    
    new_examples = []
    for example in examples:
        post_id, fc_id, rating = example
        post_text = dataset.id_to_post[post_id]
        
        fact_check_row = df_factchecks[df_factchecks['fact_check_id'] == fc_id].iloc[0]
        claim = fact_check_row['claim']
        claim_og = claim[0]
        new_examples.append((post_text,  claim_og, rating))
        
    df.at[index, 'random_examples'] = new_examples
    
df.to_csv('./datasets/fewshot_examples-random.csv',index=False)


df = pd.read_csv('./datasets/fewshot-random-en.csv')
for column in columns_to_add:
    df[column] = ''
df['random_examples_with_ids'] = df['random_examples_with_ids'].apply(eval)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    examples = row['random_examples_with_ids']
    post_text = dataset.id_to_post[row['post_id']]
    post_text_en = dataset_en.id_to_post[row['post_id']]
    
    fact_check_row = df_factchecks[df_factchecks['fact_check_id'] == row['factcheck_id']].iloc[0]
    claim = fact_check_row['claim']
    claim_og = claim[0]
    claim_en = claim[1]
    
    df.at[index, 'post_text'] = post_text
    df.at[index, 'post_text_en'] = post_text_en
    df.at[index, 'factcheck_text'] = claim_og
    df.at[index, 'factcheck_text_en'] = claim_en
    
    new_examples = []
    for example in examples:
        post_id, fc_id, rating = example
        post_text = dataset.id_to_post[post_id]
        
        fact_check_row = df_factchecks[df_factchecks['fact_check_id'] == fc_id].iloc[0]
        claim = fact_check_row['claim']
        claim_og = claim[0]
        new_examples.append((post_text,  claim_og, rating))
        
    df.at[index, 'random_examples'] = new_examples
    
df.to_csv('./datasets/fewshot_examples-random-en.csv',index=False)
