import ast
import pandas as pd
from tqdm import tqdm
from src.datasets import dataset_factory

parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s

dataset = dataset_factory('multiclaim').load()
dataset_en = dataset_factory('multiclaim', version="english").load()

df_posts = pd.read_csv('./datasets/multiclaim/posts.csv').fillna('')
for col in ['instances', 'ocr', 'verdicts', 'text']:
    df_posts[col] = df_posts[col].apply(parse_col)
    
df_factchecks = pd.read_csv('./datasets/multiclaim/fact_checks.csv').fillna('')
for col in ['claim', 'instances', 'title']:
    df_factchecks[col] = df_factchecks[col].apply(parse_col)

df = pd.read_csv('./datasets/amc-16k.csv')
columns_to_add = [
    'post_text',
    'post_text_en',
    'fact_check_text',
    'fact_check_text_en'
]
for col in columns_to_add:
    df[col] = ''
    
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    post_id = row['post_id']
    fact_check_id = row['factcheck_id']
    
    post = df_posts[df_posts['post_id'] == post_id].iloc[0]
    fact_check = df_factchecks[df_factchecks['fact_check_id'] == fact_check_id].iloc[0]
    
    fact_check_row = df_factchecks[df_factchecks['fact_check_id'] == fact_check_id].iloc[0]
    claim = fact_check_row['claim']
    claim_og = claim[0]
    claim_en = claim[1]
    
    df.at[index, 'post_text'] = dataset.id_to_post[post_id]
    df.at[index, 'post_text_en'] = dataset_en.id_to_post[post_id]
    df.at[index, 'fact_check_text'] = claim_og
    df.at[index, 'fact_check_text_en'] = claim_en
    
df.to_csv('./datasets/annotation_pairs.csv',index=False)