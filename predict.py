
"""
Live THAR Binary Predictor
0 = non_communal
1 = communal
"""

import argparse
import sys
import os
import torch
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessing.dataset import clean_tweet
from src.models.hybrid_model import HybridXLM_BiLSTM


def predict(text, seed=7, max_len=96, threshold=None):
    thresholds = {123: 0.63, 2024: 0.62, 7: 0.33, 21: 0.27}
    if threshold is None:
        threshold = thresholds.get(seed)
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("Provide a validation-selected threshold between 0 and 1.")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_name = "xlm-roberta-base"

    checkpoint_path = os.path.join(
        BASE_DIR,
        "checkpoints",
        f"thar_unfrozen_seed_{seed}",
        "best_thar_binary_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = HybridXLM_BiLSTM(
        pretrained_model_name=model_name,
        lstm_hidden_size=256,
        dropout_rate=0.3,
        num_classes=1,
        freeze_encoder=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cleaned = clean_tweet(text)

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        probability = torch.sigmoid(logits).item()

    prediction = (
        "COMMUNAL" if probability >= threshold
        else "NON-COMMUNAL"
    )

    print("\n" + "=" * 55)
    print("             THAR BINARY PREDICTION")
    print("=" * 55)
    print(f"Input Text:          {text}")
    print(f"Cleaned Text:        {cleaned}")
    print("-" * 55)
    print(f"Prediction:          {prediction}")
    print(f"Communal Probability: {probability * 100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    predict(args.text or input("Enter text to classify: "), args.seed, args.max_len, args.threshold)
