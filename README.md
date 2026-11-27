# Hinglish Hate Speech Detection

A research implementation for binary classification of religion-targeted communal content in Hindi-English code-mixed YouTube comments using XLM-RoBERTa and BiLSTM.

## Methodology

THAR benchmark: 11,549 labeled comments; stratified train/validation/test split of 9,239 / 1,155 / 1,155 (split seed 42). Class 1 denotes religion-targeted hostility; class 0 denotes other content, not necessarily all benign speech.

```mermaid
flowchart LR
 A[Comment] --> B[Normalization and tokenization] --> C[XLM-RoBERTa] --> D[BiLSTM] --> E[Dense head] --> F[Sigmoid and threshold]
```

The unfrozen model uses 256 LSTM hidden units per direction, sequence length 96, dropout 0.3, AdamW, batch size 8 and up to five epochs. Validation data selects the decision threshold; test data is used for evaluation.

## Reported results

| Metric | TF-IDF + Logistic Regression | Fine-tuned XLM-R + BiLSTM (4-seed mean) |
|---|---:|---:|
| Accuracy | 68.14% | 75.97% |
| Positive-class F1 | 71.21% | 76.50% |
| ROC-AUC | 75.84% | 82.80% |
| False positives | 278 | 184.5 |

Reported false-positive reduction: 33.6%. Recall changes from 83.49% to 82.94%. See [recorded evaluation outputs](results/) and [metrics CSV](results/metrics.csv). These are historical results from the supplied notebook/report, not newly reproduced. Frozen encoder experiments are described in the dissertation but are not exposed as a switch in this code snapshot.

## Run locally

Python 3.11 is a suggested environment; exact dependency versions from the experiments were not recorded here. A CUDA GPU is recommended for training.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Obtain the THAR dataset from its authorized source (see citation below), verify redistribution/use terms and place the compatible CSV at `data/raw/Thar_Dataset.csv`. Raw and processed data are intentionally excluded.

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --seed 123 --force_split
.\.venv\Scripts\python.exe run_pipeline.py --seed 2024
.\.venv\Scripts\python.exe run_pipeline.py --seed 7
.\.venv\Scripts\python.exe run_pipeline.py --seed 21
.\.venv\Scripts\python.exe tfidf_baseline.py
```

## Inference

Training writes a checkpoint in `checkpoints/thar_unfrozen_seed_7/best_thar_binary_model.pt` for seed 7. Alternatively, place an authorized matching checkpoint there. The original predictor was adjusted for seed-specific paths, sequence length 96 and recorded validation thresholds. Use only trusted checkpoint files. For newly trained weights, supply the threshold from that run's evaluation report (do not assume the historical threshold still applies).

```powershell
.\.venv\Scripts\python.exe predict.py "Aaj ka match bahut accha tha" --seed 7 --threshold 0.33
```

No trained weights are supplied. No live demo or new training/inference execution has been verified.

## Limitations

Results are limited to the THAR benchmark and tested split. Sarcasm and implicit hostility remain difficult; cross-platform generalization is unverified. This model is a research prototype, not an autonomous moderation decision system.

## Dataset reference

THAR: Targeted Hate Speech Against Religion: A High-Quality Hindi-English Code-Mixed Dataset with the Application of Deep Learning Models for Automatic Detection. DOI as listed in the supplied report: https://doi.org/10.1145/3653017 (bibliographic details should be verified before publication).

## Research Report

For detailed methodology, model architecture,
experimental results, and performance analysis,
refer to the technical research report.

[View Research Report](docs/Hinglish_Hate_Speech_Research_Report.pdf)
