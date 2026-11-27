
"""
Binary Training Engine for XLM-RoBERTa + BiLSTM
THAR: 0 = non_communal, 1 = communal
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.dataset import get_dataloader
from src.models.hybrid_model import HybridXLM_BiLSTM


def train_model(
    train_csv=os.path.join(BASE_DIR, "data", "processed", "train.csv"),
    val_csv=os.path.join(BASE_DIR, "data", "processed", "val.csv"),
    model_name="xlm-roberta-base",
    batch_size=8,
    max_len=64,
    epochs=10,
    encoder_lr=1e-5,
    head_lr=2e-5,
    weight_decay=0.01,
    patience=3,
    checkpoint_dir=os.path.join(BASE_DIR, "checkpoints", "thar"),
    output_dir=os.path.join(BASE_DIR, "outputs", "thar")
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = get_dataloader(
        train_csv, tokenizer,
        batch_size=batch_size, max_len=max_len, shuffle=True
    )
    val_loader = get_dataloader(
        val_csv, tokenizer,
        batch_size=batch_size, max_len=max_len, shuffle=False
    )

    print("Building binary XLM-RoBERTa + BiLSTM model...")
    model = HybridXLM_BiLSTM(
        pretrained_model_name=model_name,
        lstm_hidden_size=256,
        dropout_rate=0.3,
        num_classes=1,
        freeze_encoder=False
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
    [
        {
            "params": model.roberta.parameters(),
            "lr": encoder_lr
        },
        {
            "params": model.bilstm.parameters(),
            "lr": head_lr
        },
        {
            "params": model.dense.parameters(),
            "lr": head_lr
        },
        {
            "params": model.classifier_binary.parameters(),
            "lr": head_lr
        }
    ],
    weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0
    best_model_path = os.path.join(
        checkpoint_dir, "best_thar_binary_model.pt"
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} [Train]",
            leave=False
        )

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_binary"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            logits = logits.view(-1)
            loss = criterion(logits, labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            batch_size_actual = labels.size(0)
            train_loss += loss.item() * batch_size_actual

            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels.view(-1)).sum().item()
            train_total += batch_size_actual

            loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label_binary"].to(device)

                logits = model(input_ids, attention_mask)
                if isinstance(logits, (tuple, list)):
                    logits = logits[-1]

                logits = logits.view(-1)
                loss = criterion(logits, labels.view(-1))

                batch_size_actual = labels.size(0)
                val_loss += loss.item() * batch_size_actual

                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels.view(-1)).sum().item()
                val_total += batch_size_actual

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Acc: {epoch_val_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stopping_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": epoch_val_acc,
                "model_type": "binary",
                "label_mapping": {
                    "0": "non_communal",
                    "1": "communal"
                }
            }, best_model_path)

            print(f"Saved best checkpoint: {best_model_path}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    history_path = os.path.join(
        output_dir, "metrics", "training_history.json"
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Plot binary training curves without old baseline comparison
    try:
        from src.utils.metrics import plot_training_curves

        plot_training_curves(
            history,
            os.path.join(output_dir, "plots", "training_validation_accuracy.png"),
            os.path.join(output_dir, "plots", "training_validation_loss.png")
        )
    except Exception as exc:
        print(f"Could not generate training plots: {exc}")

    print("\nTHAR binary training complete.")
    print(f"Best checkpoint: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train_model()
