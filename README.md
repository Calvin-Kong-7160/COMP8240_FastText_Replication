# Reproducing *Bag of Tricks for Efficient Text Classification* (fastText)

COMP8240 (Applications of Data Science), Macquarie University, Session 2 2026 – Chun Hung (Calvin) Kong

This repository replicates the text-classification results of Joulin et al. (2017) using the official
[fastText](https://github.com/facebookresearch/fastText) code, applies the method to a new existing dataset
(BBC News), and constructs and evaluates a new dataset (BBC-Sentences) with human and LLM-as-judge annotation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Calvin-Kong-7160/COMP8240_FastText_Replication/blob/main/fasttext_replication.ipynb)

## How to run

Everything is in one notebook, `fasttext_replication.ipynb`, written for **Google Colab's free CPU runtime** (2 threads).
Click the badge above (or upload the notebook to Colab) and run the cells top to bottom.

| Section | What it does | Time on free Colab |
|---|---|---|
| 1 | Clones and builds the official fastText code (`make`) | ~1 min |
| 2 | Helper functions: download, preprocessing, training, testing | seconds |
| 3 | Replication on the paper's 8 datasets (unigram and bigram models) | AG/DBpedia/Yelp ~30 min; Yahoo, Amazon, Sogou ~2 h in total |
| 4 | BBC News articles (new existing dataset) | seconds |
| 5 | Builds BBC-Sentences (new constructed dataset) and trains on it | ~1 min |
| 6 | Exports the 100-sentence annotation sample | seconds |
| 6b | Click-through annotation tool (one annotator at a time) | ~15 min of human time |
| 7 | LLM-as-judge on the annotation sample (Colab's built-in model, no API key) | ~5 min |
| 8 | Cohen's kappa between annotators and the LLM; label-noise estimate | seconds |

All datasets are downloaded automatically; nothing needs to be placed by hand.
Python dependencies (`datasets`, `pandas`, `scikit-learn`, `ipywidgets`) are preinstalled or installed by the notebook;
`requirements.txt` lists them for running outside Colab.

## Settings (as in the paper)

10-dimensional embeddings, 5 epochs, `-wordNgrams 1` or `2`, 10M hash buckets for bigrams, `-minCount 1`.
The learning rate is chosen on a 5% validation split held out from the training data
({0.05, 0.1, 0.25, 0.5}; {0.1, 0.25} for the four largest datasets to save time).
Text is lower-cased and punctuation is separated from words.

## Data sources

| Dataset | Source |
|---|---|
| AG News, DBpedia, Yelp P/F, Yahoo! Answers, Amazon P | Hugging Face copies of Zhang et al. (2015): `fancyzhx/*`, `Yelp/yelp_review_full`, `community-datasets/yahoo_answers_topics` |
| Sogou News | `community-datasets/sogou_news` (Hugging Face) |
| Amazon Full | original Zhang et al. CSV release, fast.ai mirror (`s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz`) |
| BBC News | `SetFit/bbc-news` (Hugging Face): Greene & Cunningham (2006), 1,225 train / 1,000 test |

## Results

### Replication (test accuracy %, ours / paper)

| Dataset | Unigram | Bigram |
|---|---|---|
| AG News | 91.8 / 91.5 | 92.4 / 92.5 |
| Sogou | 93.7 / 93.9 | 96.7 / 96.8 |
| DBpedia | 98.3 / 98.1 | 98.5 / 98.6 |
| Yelp Polarity | 93.7 / 93.8 | 95.6 / 95.7 |
| Yelp Full | 60.5 / 60.4 | 64.0 / 63.9 |
| Yahoo! Answers | 72.0 / 72.0 | 72.5 / 72.3 |
| Amazon Full | 55.7 / 55.8 | 59.8 / 60.2 |
| Amazon Polarity | 91.1 / 91.2 | 94.5 / 94.6 |

All 16 results are within 0.4 points of the paper (mean absolute difference 0.14).
Training takes 4 s to 8.4 min per model on 2 CPU threads. fastText's multithreaded training is not deterministic,
so re-runs vary by about 0.1–0.3 points.

### New data

| Data | Paper settings (5 epochs) | Tuned on validation |
|---|---|---|
| BBC articles (unigram / bigram) | 37.0 / 22.4 | 95.7 / 95.1 (25 epochs) |
| BBC-Sentences (unigram / bigram) | 80.0 / 81.3 | 79.4 / 80.8 |

Two annotators (blind to the article label and to fastText's prediction) labelled 100 random test sentences:
Cohen's kappa 0.74 on the sentence's topic and 0.61 on whether it signals the article's topic.
34% of sentences do not signal their inherited label; fastText scores 86% on the sentences that do and 56% on those that don't.
Full numbers are in `results/`.

## Repository contents

```
fasttext_replication.ipynb   all code (sections 1–8, see table above)
requirements.txt             Python packages, for running outside Colab
results/
  replication.csv            replication results for the 8 datasets
  bbc_results.csv            BBC articles and BBC-Sentences, paper settings
  bbc_results_tuned.csv      the same with epochs and learning rate tuned on validation
  agreement.csv              Cohen's kappa and raw agreement: annotator A, annotator B, LLM judge
  llm_judge.csv              LLM-as-judge responses, raw output included
annotation/
  bbc_annotation_sheet.csv   the 100-sentence sheet annotators labelled (no predictions shown)
  bbc_annotation_key.csv     the same sentences with the inherited label and fastText's prediction
  annotator_A.csv, annotator_B.csv   the two human annotations
```

### BBC-Sentences

BBC-Sentences is built deterministically by the notebook (section 5): each BBC News article is split into sentences
and every sentence of five or more words inherits its article's label (section 5 prints the exact train/test sentence counts).
The full sentence files are not redistributed here because the article text is copyright of the BBC
(the original dataset is released for non-commercial research use); running section 5 regenerates them exactly.
The 100 annotated sentences in `annotation/` are included as a sample.

## References

- Joulin, Grave, Bojanowski, Mikolov. 2017. *Bag of Tricks for Efficient Text Classification.* EACL.
- Zhang, Zhao, LeCun. 2015. *Character-level Convolutional Networks for Text Classification.* NeurIPS.
- Greene, Cunningham. 2006. *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering.* ICML.
