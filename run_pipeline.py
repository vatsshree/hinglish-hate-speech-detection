
"""
THAR Binary Classification Pipeline
1. Use existing THAR train/validation/test splits
2. Train unfrozen XLM-RoBERTa + BiLSTM
3. Evaluate on held-out test data
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing.data_split import prepare_splits
from src.training.train import train_model
from src.training.evaluate import evaluate_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_full_pipeline(
    batch_size=8,
    epochs=5,
    max_len=96,
    encoder_lr=1e-5,
    head_lr=2e-5,
    seed=42,
    force_split=False
):
    # Set seed before model and dataloader creation
    set_seed(seed)

    print("=" * 60)
    print(" THAR BINARY COMMUNITY-RELATED CONTENT DETECTION")
    print(" XLM-RoBERTa + BiLSTM (UNFROZEN)")
    print(f" Seed: {seed}")
    print("=" * 60)

    raw_csv = os.path.join(
        BASE_DIR, "data", "raw", "Thar_Dataset.csv"
    )
    processed_dir = os.path.join(
        BASE_DIR, "data", "processed"
    )

    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "val.csv")
    test_csv = os.path.join(processed_dir, "test.csv")

    # Separate folders for every seed
    checkpoint_dir = os.path.join(
        BASE_DIR, "checkpoints", f"thar_unfrozen_seed_{seed}"
    )
    output_dir = os.path.join(
        BASE_DIR, "outputs", f"thar_unfrozen_seed_{seed}"
    )

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"THAR dataset not found: {raw_csv}"
        )

    if force_split or not all(
        os.path.exists(path)
        for path in [train_csv, val_csv, test_csv]
    ):
        print("\n[Step 1/3] Creating THAR stratified 80/10/10 splits...")
        prepare_splits(raw_csv, processed_dir)
    else:
        print(
            "\nExisting split files found. "
            "Using the current train/validation/test split."
        )

    print(
        f"\n[Step 2/3] Training unfrozen model "
        f"(seed={seed}, epochs={epochs}, batch_size={batch_size}, "
        f"max_len={max_len}, encoder_lr={encoder_lr}, "
        f"head_lr={head_lr})..."
    )

    best_checkpoint = train_model(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=batch_size,
        max_len=max_len,
        epochs=epochs,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir
    )

    print("\n[Step 3/3] Evaluating on THAR test split...")

    evaluate_model(
        val_csv=val_csv,
        test_csv=test_csv,
        checkpoint_path=best_checkpoint,
        batch_size=batch_size,
        max_len=max_len,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print(" THAR PIPELINE COMPLETED")
    print(f" Seed:       {seed}")
    print(f" Checkpoint: {best_checkpoint}")
    print(f" Outputs:    {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run THAR binary classification pipeline"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--force_split",
        action="store_true",
        help="Regenerate THAR splits"
    )

    args = parser.parse_args()

    run_full_pipeline(
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_len=args.max_len,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        seed=args.seed,
        force_split=args.force_split
    )
