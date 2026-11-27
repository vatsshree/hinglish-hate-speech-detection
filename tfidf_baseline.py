import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# -------------------------
# Paths
# -------------------------
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/tfidf_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load the existing splits
# -------------------------
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

TEXT_COL = "Comment"
LABEL_COL = "label"  # Change this if your split files use a different label column

X_train = train_df[TEXT_COL].fillna("").astype(str)
y_train = train_df[LABEL_COL].astype(int)

X_val = val_df[TEXT_COL].fillna("").astype(str)
y_val = val_df[LABEL_COL].astype(int)

X_test = test_df[TEXT_COL].fillna("").astype(str)
y_test = test_df[LABEL_COL].astype(int)

# -------------------------
# TF-IDF features
# -------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------
# Train Logistic Regression
# -------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_tfidf, y_train)

# -------------------------
# Select threshold on validation set
# -------------------------
val_probs = model.predict_proba(X_val_tfidf)[:, 1]

thresholds = np.arange(0.20, 0.81, 0.01)
best_threshold = 0.5
best_val_f1 = -1

for threshold in thresholds:
    val_preds = (val_probs >= threshold).astype(int)
    score = f1_score(y_val, val_preds, pos_label=1)

    if score > best_val_f1:
        best_val_f1 = score
        best_threshold = threshold

# -------------------------
# Evaluate once on test set
# -------------------------
test_probs = model.predict_proba(X_test_tfidf)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

results = {
    "model": "TF-IDF + Logistic Regression",
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_preds)),
    "communal_precision": float(precision_score(y_test, test_preds, pos_label=1, zero_division=0)),
    "communal_recall": float(recall_score(y_test, test_preds, pos_label=1, zero_division=0)),
    "communal_f1": float(f1_score(y_test, test_preds, pos_label=1, zero_division=0)),
    "macro_f1": float(f1_score(y_test, test_preds, average="macro")),
    "auc_roc": float(roc_auc_score(y_test, test_probs)),
    "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
}

# Save results
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=4)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(classification_report(
        y_test,
        test_preds,
        target_names=["non_communal", "communal"],
        digits=4,
    ))

# Save confusion matrix plot
cm = confusion_matrix(y_test, test_preds)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("TF-IDF + Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["Non-communal", "Communal"])
plt.yticks([0, 1], ["Non-communal", "Communal"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

print("\nTF-IDF Baseline Results")
print(json.dumps(results, indent=4))
print(f"\nSaved outputs to: {OUTPUT_DIR}")