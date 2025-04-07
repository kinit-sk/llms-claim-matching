# Large Language Models for Multilingual Previously Fact-Checked Claim Detection

This repository contains the source code for the paper "_Large Language Models for Multilingual Previously Fact-Checked Claim Detection_".

## Abstract

In our era of widespread false information, human fact-checkers often face the challenge of duplicating efforts when verifying claims that may have already been addressed in other countries or languages. As false information transcends linguistic boundaries, the ability to automatically detect previously fact-checked claims across languages has become an increasingly important task. This paper presents the first comprehensive evaluation of large language models (LLMs) for _multilingual_ previously fact-checked claim detection. We assess seven LLMs across 20 languages in both monolingual and cross-lingual settings. Our results show that while LLMs perform well for high-resource languages, they struggle with low-resource languages. Moreover, translating original texts into English proved to be beneficial for low-resource languages. These findings highlight the potential of LLMs for multilingual previously fact-checked claim detection and provide a foundation for further research on this promising application of LLMs.

## Reproducibility

### Installation

To set up the environment and replicate our experiments, install Python version 3.12.4 and the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Experiments

We provide several scripts to facilitate the evaluation of large language models for previously fact-checked claim detection.

#### Dataset Preprocessing

To prepare the dataset used in our experiments, run:

```bash
python -m scripts.preprocess_annotation
```

This script generates `./datasets/annotation_pairs.csv`, which includes metadata, text of social media posts, fact-checked claims, and their English translations.

#### Preparing Data for Few-Shot Experiments

For experiments involving few-shot prompting, prepare the few-shot demonstrations for each record by running:

```bash
python -m scripts.preprocess_fewshot
```

This creates multiple files, with the most important being:

- `./datasets/fewshot_examples-random.csv` -  Few-shot examples for experiments in the original language.
- `./datasets/fewshot_examples-random-en.csv` - Few-shot examples for experiments with English translations.


#### Running Experiments

To execute experiments using LLMs in both the original language and English translations, run:

```bash
python -m scripts.run_experiments
```

#### Evaluation

We provide Jupyter notebooks for evaluation.

##### Preprocessing Results for Evaluation

The CoT (Chain-of-Thought) and XLT (Cross-Lingual-Though Prompting) prompting techniques generate not only final labels but also reasoning. To extract the predicted labels, run the notebook `Process results.ipynb`. This will create a new column `prediction_label`, containing the final _Yes_ or _No_ labels.

##### Evaluating Model Performance

To evaluate results and generate visualizations used in the paper, execute the notebook `Evaluation.ipynb`.

## Annotation guidelines

Since our experiments involve human annotation, `Annotation guidelines.pdf` provides detailed instructions and examples on how to annotate the data.
