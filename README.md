# Fake News Detection with Transformer Language Models

End-to-end NLP pipeline that fine-tunes **BERT**, **RoBERTa**, and **DeBERTa** to classify news articles as **Fake** or **Real** — built with a deliberate focus on data quality auditing and catching data leakage before it inflates the results.

---

## Overview

- **Task:** Binary text classification (Fake vs Real news)
- **Dataset:** 44,898 articles from two source files (Fake.csv, True.csv)
- **Models compared:** `bert-base-uncased`, `roberta-base`, `microsoft/deberta-base`
- **Approach:** Full audit → leakage detection & mitigation → cleaning → stratified split → fine-tuning → evaluation

## Key Finding: Data Leakage

Before any modeling, an audit of the raw data revealed two near-perfect label proxies that had to be excluded from the model input:

- **`subject` column:** 100% of subjects are exclusively Fake or exclusively Real — this single categorical column could "predict" the label almost perfectly on its own.
- **Wire-service dateline:** nearly every real article opens with `"CITY (Reuters) -"`; almost no fake article does, and *"reuters"* even shows up in the top-20 real-news vocabulary.

Both signals were removed from the model input (`subject` and `date` are excluded as features) so the models learn from article *content*, not source artifacts.

## Pipeline

1. Remove empty-text rows and exact duplicates
2. Build the model input: `content = title + text`
3. Deduplicate on content **before** splitting (prevents cross-split leakage)
4. Exclude `subject` and `date` as model features
5. Stratified 80 / 10 / 10 train / validation / test split (seed = 42)
6. Tokenize (max length 256) → fine-tune (3 epochs, lr = 2e-5, batch size 8) → evaluate

## Results (held-out test set)

| Metric    | BERT   | RoBERTa | DeBERTa |
|-----------|--------|---------|---------|
| Accuracy  | 0.9979 | 0.9995  | pending |
| Precision | 0.9967 | 1.0000  | pending |
| Recall    | 0.9995 | 0.9991  | pending |
| F1        | 0.9981 | 0.9995  | pending |
| ROC-AUC   | 0.9997 | 1.0000  | pending |
| Loss      | 0.0102 | 0.0040  | pending |

RoBERTa misclassified only 2 of 3,866 held-out articles. DeBERTa fine-tuning is in progress; results will be added once complete.

## Tech Stack

Python · Pandas · NumPy · Hugging Face Transformers · PyTorch · Scikit-learn · Matplotlib · Seaborn

## Repository Structure

```
├── notebooks/
│   └── fake_news_detection_bert_roberta_deberta.ipynb
├── data/                  # not included — see Dataset section below
├── README.md
└── requirements.txt
```

## Dataset

Sourced from the public "Fake and Real News" dataset (Fake.csv / True.csv). Raw CSVs are not included in this repository due to size — download them from the original source and place them in a local `data/` folder before running the notebook.

## How to Run

```bash
git clone https://github.com/mahmoudeimo95/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/fake_news_detection_bert_roberta_deberta.ipynb
```

## Limitations & Next Steps

- Dataset is time- and topic-bound (largely 2016–2017 US politics / world news)
- Leakage was mitigated, not eliminated by design — the underlying source-style gap in the text itself may still remain
- Next: complete DeBERTa fine-tuning, evaluate on an external fake-news dataset, and strip residual wire-service datelines from article text

## Author

**Mahmoud** — Data Analyst
[GitHub](https://github.com/mahmoudeimo95) · [Freelancer.com](https://www.freelancer.com/u/mahmoudm301)
