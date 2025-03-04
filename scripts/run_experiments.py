import os 

configs = [
    # C4AI Command R+ 104B
    '70b/e5-c4ai-vanilla.yaml',
    '70b/e5-c4ai-instruct.yaml',
    '70b/e5-c4ai-fewshot.yaml',
    '70b/e5-c4ai-cot.yaml',
    '70b/e5-c4ai-xlt.yaml',
    
    # Llama3.1 70B
    '70b/e5-llama3-vanilla.yaml',
    '70b/e5-llama3-instruct.yaml',
    '70b/e5-llama3-fewshot.yaml',
    '70b/e5-llama3-cot.yaml',
    'e5-llama3-xlt.yaml',
    
    # Mistral Large 123B
    '70b/e5-mistral-vanilla.yaml',
    '70b/e5-mistral-instruct.yaml',
    '70b/e5-mistral-fewshot.yaml',
    '70b/e5-mistral-cot.yaml',
    '70b/e5-mistral-xlt.yaml',
    
    # Qwen2.5 72B
    '70b/e5-qwen-vanilla.yaml',
    '70b/e5-qwen-instruct.yaml',
    '70b/e5-qwen-fewshot.yaml',
    '70b/e5-qwen-cot.yaml',
    '70b/e5-qwen-xlt.yaml',
    
    # Llama3.1 8B
    '10b/e5-llama3-vanilla.yaml',
    '10b/e5-llama3-instruct.yaml',
    '10b/e5-llama3-fewshot.yaml',
    '10b/e5-llama3-cot.yaml',
    '10b/e5-llama3-xlt.yaml',
    
    #Qwen2.5 8B
    '10b/e5-qwen-vanilla.yaml',
    '10b/e5-qwen-instruct.yaml',
    '10b/e5-qwen-fewshot.yaml',
    '10b/e5-qwen-cot.yaml',
    '10b/e5-qwen-xlt.yaml',
    
    # Mistral 7B
    '10b/e5-mistral-vanilla.yaml',
    '10b/e5-mistral-instruct.yaml',
    '10b/e5-mistral-fewshot.yaml',
    '10b/e5-mistral-cot.yaml',
    '10b/e5-mistral-xlt.yaml',
    
]

output_dirs = [
    'c4ai-104b-vanilla',
    'c4ai-104b-instruct',
    'c4ai-104b-fewshot',
    'c4ai-104b-cot',
    'c4ai-104b-xlt',

    'llama3_1-70b-vanilla',
    'llama3_1-70b-instruct',
    'llama3_1-70b-fewshot',
    'llama3_1-70b-cot',
    'llama3_1-70b-xlt',
    
    'mistral-123b-vanilla',
    'mistral-123b-instruct',
    'mistral-123b-fewshot',
    'mistral-123b-cot',
    'mistral-123b-xlt',
    
    'qwen2_5-72b-vanilla',
    'qwen2_5-72b-instruct',
    'qwen2_5-72b-fewshot',
    'qwen2_5-72b-cot',
    'qwen2_5-72b-xlt',
    
    'llama3_1-8b-vanilla',
    'llama3_1-8b-instruct',
    'llama3_1-8b-fewshot',
    'llama3_1-8b-cot',
    'llama3_1-8b-xlt',
    
    'qwen2_5-8b-vanilla',
    'qwen2_5-8b-instruct',
    'qwen2_5-8b-fewshot',
    'qwen2_5-8b-cot',
    'qwen2_5-8b-xlt',
    
    'mistral-7b-vanilla',
    'mistral-7b-instruct',
    'mistral-7b-fewshot',
    'mistral-7b-cot',
    'mistral-7b-xlt',
]

for output_dir, config in zip(output_dirs, configs):
    os.system(f"python -m scripts.annotated_experiments --config ./configs/{config} --output_dir {output_dir}")
    
for output_dir, config in zip(output_dirs, configs):
    output_dir = output_dir + '-en'
    os.system(f"python -m scripts.annotated_experiments --english --config ./configs/{config} --output_dir {output_dir}")
    
