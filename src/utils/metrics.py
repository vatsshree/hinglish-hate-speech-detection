"""
Evaluation Metrics & Visualization Utilities
Generates standard evaluation metrics and high-resolution thesis plots:
- Confusion Matrix
- Training & Validation Accuracy Curves
- Training & Validation Loss Curves
- ROC Curve with AUC score
- Baseline Comparison Bar Chart
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
)

def compute_metrics(y_true_binary, y_pred_prob_binary, y_true_3class=None, y_pred_3class=None):
    """
    Computes both binary metrics (matching report & presentation)
    and optional 3-class metrics.
    """
    y_pred_binary = (np.array(y_pred_prob_binary) >= 0.5).astype(int)
    y_true_binary = np.array(y_true_binary).astype(int)
    
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_binary, y_pred_prob_binary)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'binary_accuracy': float(acc),
        'binary_precision': float(prec),
        'binary_recall': float(rec),
        'binary_f1': float(f1),
        'binary_auc_roc': float(auc),
        'confusion_matrix': {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    }
    
    if y_true_3class is not None and y_pred_3class is not None:
        y_true_3 = np.array(y_true_3class)
        y_pred_3 = np.array(y_pred_3class)
        results['3class_accuracy'] = float(accuracy_score(y_true_3, y_pred_3))
        results['3class_macro_f1'] = float(f1_score(y_true_3, y_pred_3, average='macro', zero_division=0))
        results['3class_report'] = classification_report(
            y_true_3, y_pred_3,
            target_names=['neither', 'non_communal', 'communal'],
            output_dict=True,
            zero_division=0
        )
        
    return results


def plot_confusion_matrix(cm, output_path, labels=('communal', 'non-communal')):
    """
    Plots professional confusion matrix matching Figure 5 from presentation.
    Note: in research paper, labels are communal (pos) vs non-communal (neg).
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        annot_kws={'size': 13}
    )
    plt.title('Confusion Matrix for Communal Tweet Detection', fontsize=12, pad=12)
    plt.xlabel('Predicted Labels', fontsize=11)
    plt.ylabel('Actual Labels', fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_curves(history, accuracy_path, loss_path):
    """
    Generates training & validation accuracy and loss curves matching Part 2 report.
    """
    epochs = range(1, len(history['train_acc']) + 1)
    
    # 1. Accuracy Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history['train_acc'], marker='o', label='Training Accuracy', color='#1f77b4')
    plt.plot(epochs, history['val_acc'], marker='s', label='Validation Accuracy', color='#ff7f0e')
    plt.title('Training and Validation Accuracy Curve', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.ylim([0.5, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(accuracy_path), exist_ok=True)
    plt.savefig(accuracy_path, dpi=300)
    plt.close()
    
    # 2. Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history['train_loss'], marker='o', label='Training Loss', color='#1f77b4')
    plt.plot(epochs, history['val_loss'], marker='s', label='Validation Loss', color='#ff7f0e')
    plt.title('Training and Validation Loss Curve', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.ylim([0.1, 0.8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(loss_path), exist_ok=True)
    plt.savefig(loss_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_probs, auc_score, output_path):
    """
    Plots Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='#000080', lw=1.5, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, pad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_baseline_comparison(output_path, proposed_acc=96.48):
    """
    Plots baseline comparison bar chart matching Figure 4.5 in thesis report:
    SVM (81.34%), LogReg (78.89%), CNN (88.76%), RNN (89.42%), Proposed (96.48%)
    """
    models = ['SVM', 'LogReg', 'CNN', 'RNN', 'XLM-RoBERTa+BiLSTM']
    accuracies = [81.34, 78.89, 88.76, 89.42, proposed_acc]
    colors = ['#7f7f7f', '#7f7f7f', '#ff9900', '#ff9900', '#008000']
    
    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(models, accuracies, color=colors, width=0.7)
    plt.ylim([70, 100])
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Accuracy Comparison with Baseline Models', fontsize=12, pad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

