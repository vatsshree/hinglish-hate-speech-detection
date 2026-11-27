
"""
Binary Evaluation Pipeline for THAR
0 = non_communal, 1 = communal
Threshold selected on validation data and applied unchanged to test data.
"""

import os
import sys
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.dataset import get_dataloader
from src.models.hybrid_model import HybridXLM_BiLSTM


def get_predictions(model, loader, device):
    y_true, y_probs = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_binary"].cpu().numpy()

            logits = model(input_ids, attention_mask)

            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_true.extend(labels.tolist())
            y_probs.extend(probs.tolist())

    return np.array(y_true).astype(int), np.array(y_probs)


def plot_confusion_matrix(cm, save_path):
    """Save a binary confusion matrix plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    classes = ["Non-communal", "Communal"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="THAR Binary Confusion Matrix",
    )

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_probs, auc, save_path):
    """Save the binary ROC curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="THAR Binary ROC Curve",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    val_csv=os.path.join(BASE_DIR, "data", "processed", "val.csv"),
    test_csv=os.path.join(BASE_DIR, "data", "processed", "test.csv"),
    checkpoint_path=os.path.join(
        BASE_DIR, "checkpoints", "thar", "best_thar_binary_model.pt"
    ),
    model_name="xlm-roberta-base",
    batch_size=8,
    max_len=64,
    output_dir=os.path.join(BASE_DIR, "outputs", "thar"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading THAR checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    val_loader = get_dataloader(
        val_csv,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_csv,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=False,
    )

    model = HybridXLM_BiLSTM(
        pretrained_model_name=model_name,
        lstm_hidden_size=256,
        dropout_rate=0.3,
        num_classes=1,
        freeze_encoder=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Select threshold using validation data only
    val_y, val_probs = get_predictions(model, val_loader, device)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        val_preds = (val_probs >= threshold).astype(int)
        score = f1_score(val_y, val_preds, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    print(f"\nValidation-selected threshold: {best_threshold:.2f}")
    print(f"Validation communal F1: {best_f1:.4f}")

    # Evaluate test data with the fixed validation threshold
    y_true, y_probs = get_predictions(model, test_loader, device)
    y_pred = (y_probs >= best_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = (
        roc_auc_score(y_true, y_probs)
        if len(np.unique(y_true)) == 2
        else None
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-communal", "Communal"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "task": "THAR binary community-related classification",
        "label_mapping": {
            "0": "non_communal",
            "1": "communal",
        },
        "threshold_selected_on_validation": best_threshold,
        "validation_communal_f1": float(best_f1),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_communal": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "test_recall_communal": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "test_f1_communal": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "test_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_auc_roc": float(auc) if auc is not None else None,
        "confusion_matrix_order": [
            "Non-communal (0)",
            "Communal (1)",
        ],
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "classification_report": report,
    }

    metrics_dir = os.path.join(output_dir, "metrics")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    report_path = os.path.join(
        metrics_dir, "thar_test_evaluation_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    plot_confusion_matrix(
        cm,
        os.path.join(plots_dir, "thar_binary_confusion_matrix.png"),
    )

    if auc is not None:
        plot_roc_curve(
            y_true,
            y_probs,
            auc,
            os.path.join(plots_dir, "thar_binary_roc_curve.png"),
        )

    print("\n=== THAR BINARY TEST RESULTS ===")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision_communal']:.4f}")
    print(f"Recall:    {metrics['test_recall_communal']:.4f}")
    print(f"F1:        {metrics['test_f1_communal']:.4f}")
    print(f"Macro F1:  {metrics['test_macro_f1']:.4f}")
    print(f"AUC-ROC:   {metrics['test_auc_roc']}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"\nReport saved: {report_path}")
    print(f"Plots saved: {plots_dir}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
