# AMC-16K (Annotated-MultiClaim-16K) dataset

## Dataset Description

The **AMC-16K** dataset consists of 8K monolingual and 8K cross-lingual pairs of social media posts and fact-checked claims, which were annotated by human annotators.

The dataset include 20 languages, especially *Arabic, Bulgarian, Czech, German, Greek, English, French, Serbo-Croatian, Hindi, Hungarian, Korean, Malay, Burmese, Dutch, Polish, Portuguese, Romanian, Slovak, Spanish* and *Thai*.

For monolingual settings, we selected 40 posts per langauge from the [**MultiClaim**](https://aclanthology.org/2023.emnlp-main.1027/) dataset and retrieved top 10 fact-checked claims in the same language using [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) embedding model.

For cross-lingual settings, we defined 20 language pairs, incorporating a variety of language combinations (e.g., Slovak posts with English fact-checks). For each post, we retrieved the top 100 fact-checked claims in languages different from the post's language. From these, we selected 400 post-claim pairs per language combination.

Language pairs considered in the dataset for the cross-lingual setting:

| Post Language | Fact-Check Language |
| --- | --- |
| Spanish | English |
| Hindi | English |
| English | Arabic |
| French | English |
| German | English |
| English | Porutguese |
| Spanish | Portuguese |
| German | French |
| Slovak | Czech |
| Slovak | English |
| Polish | Serbio-Croatian |
| Czech | English |
| Czech | Polish |
| Dutch | German |
| Malay | Arabic |
| Korean | English |
| Burmese | Malay |
| Arabic | French |
| Hungarian | Polish |
| Thai | Portuguese |

### Dataset statistics

Statistics of **AMC-16K** dataset. We provide the averaged word count (WC) with standard deviation for posts and fact-checked claims (FC claims). We also calculated the number of posts and fact-checked claims for each language.

| Code | Language       | Average WC<br>Posts | Average WC<br>FC claims | \# Posts | \# FC claims |
|------|----------------|:-------------------:|:-----------------------:|----------|--------------|
| ara  | Arabic         |   57.26 $\pm$ 92.63 |       30.82 $\pm$ 49.46 |       69 |          825 |
| bul  | Bulgarian      | 169.18 $\pm$ 238.08 |        11.95 $\pm$ 3.87 |       40 |          118 |
| ces  | Czech          | 151.54 $\pm$ 181.16 |        11.09 $\pm$ 8.87 |       56 |          201 |
| deu  | German         | 114.74 $\pm$ 146.74 |       19.90 $\pm$ 17.08 |       60 |          558 |
| ell  | Greek          | 120.62 $\pm$ 237.78 |        19.67 $\pm$ 8.47 |       40 |          271 |
| eng  | English        | 195.66 $\pm$ 266.51 |       23.92 $\pm$ 30.45 |      111 |         2651 |
| fra  | French         | 129.27 $\pm$ 152.93 |       18.52 $\pm$ 12.33 |       55 |          823 |
| hbs  | Serbo-Croatian | 130.70 $\pm$ 162.29 |       23.77 $\pm$ 25.12 |       40 |          405 |
| hin  | Hindi          |   46.95 $\pm$ 39.16 |       24.20 $\pm$ 14.18 |       43 |          326 |
| hun  | Hungarian      | 127.51 $\pm$ 155.68 |        10.68 $\pm$ 3.60 |       55 |          111 |
| kor  | Korean         |  95.19 $\pm$ 103.14 |         9.96 $\pm$ 7.19 |       48 |          172 |
| msa  | Malay          | 146.50 $\pm$ 196.29 |        13.42 $\pm$ 5.61 |       50 |          576 |
| mya  | Burmese        |   51.91 $\pm$ 52.08 |         7.78 $\pm$ 5.79 |       42 |           75 |
| nld  | Dutch          | 110.17 $\pm$ 113.22 |       21.24 $\pm$ 18.72 |       45 |          240 |
| pol  | Polish         | 139.75 $\pm$ 173.43 |       20.38 $\pm$ 15.63 |       71 |          808 |
| por  | Portuguese     | 105.28 $\pm$ 121.96 |       37.29 $\pm$ 61.69 |       40 |         1242 |
| ron  | Romanian       | 126.65 $\pm$ 140.73 |        13.78 $\pm$ 4.60 |       40 |          131 |
| slk  | Slovak         | 222.77 $\pm$ 562.15 |        13.61 $\pm$ 8.63 |       91 |          154 |
| spa  | Spanish        |  91.06 $\pm$ 142.76 |       20.55 $\pm$ 13.08 |       61 |          366 |
| tha  | Thai           |   82.50 $\pm$ 66.99 |         4.00 $\pm$ 2.92 |       55 |          137 |

List of analyzed languages and language pairs in our experiments along with the proportion of relevant pairs annotated by human annotators out of 400 pairs. Each language and language combination consists of 400 pairs.

| Languages | Relevant pairs<br>[%] | Language pairs<br>(post - fact-check) | Relevant pairs<br>[%] |
|:---------:|:---------------------:|:-------------------------------------:|:---------------------:|
|    ara    |                 20.00 |               spa - eng               |                 17.50 |
|    bul    |                 11.25 |               hin - eng               |                  5.25 |
|    ces    |                 16.50 |               eng - ara               |                  5.25 |
|    deu    |                 30.25 |               fra - eng               |                 12.00 |
|    ell    |                 26.75 |               deu - eng               |                 15.25 |
|    eng    |                 38.50 |               eng - por               |                  6.00 |
|    fra    |                 19.25 |               spa - por               |                  1.50 |
|    hbs    |                 19.50 |               deu - fra               |                 16.75 |
|    hin    |                 22.25 |               slk - ces               |                  7.50 |
|    hun    |                 13.75 |               slk - eng               |                 36.25 |
|    kor    |                 13.25 |               pol - hbs               |                 11.00 |
|    msa    |                 36.00 |               ces - eng               |                 22.50 |
|    mya    |                  9.50 |               ces - pol               |                  9.00 |
|    nld    |                 20.00 |               nld - deu               |                 12.25 |
|    pol    |                 20.25 |               msa - ara               |                  2.25 |
|    por    |                 31.75 |               kor - eng               |                 27.50 |
|    ron    |                 11.50 |               mya - msa               |                  0.50 |
|    slk    |                 14.25 |               ara - fra               |                  2.25 |
|    spa    |                 23.50 |               hun - pol               |                 13.75 |
|    tha    |                 12.25 |               tha - por               |                  7.75 |

## Dataset Structure

### Dataset Fields

The `amc-16k.csv` file contains the following fields:

- `post_id` - ID of the post from the [**MultiClaim**](https://zenodo.org/records/7737983) dataset
- `factcheck_id` - ID of the fact-checked claim from the [**MultiClaim**](https://zenodo.org/records/7737983) dataset
- `post_language` - Language of the post
- `factcheck_language` - Language of the fact-checked claim
- `rating` - Rating, whether the pair of the post and the fact-checked claim is relevant (*Yes*) or irrelevant (*No*)

### Data Instances

```json
{
    "post_id": 20480,
    "factcheck_id": 145253,
    "post_language": "slk",
    "factcheck_language": "slk",
    "rating": "Yes"
}
```

## Paper Citing

If you use the code or the dataset, please cite our paper, which is available on [arXiv](https://arxiv.org/abs/2503.02737):

```bibtex
@misc{vykopal2025largelanguagemodelsmultilingual,
      title={Large Language Models for Multilingual Previously Fact-Checked Claim Detection}, 
      author={Ivan Vykopal and Matúš Pikuliak and Simon Ostermann and Tatiana Anikina and Michal Gregor and Marián Šimko},
      year={2025},
      eprint={2503.02737},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.02737}, 
}
```