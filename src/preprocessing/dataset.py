
"""
Dataset and Preprocessing Pipeline
Binary classification using XLM-RoBERTa + BiLSTM.
THAR dataset: Comment -> communal/non_communal.
"""

import re
import html
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def clean_tweet(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()


class CommunalDataset(Dataset):
    """
    Binary classification:
    0 = Non-communal
    1 = Communal
    """

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = [clean_tweet(t) for t in texts]
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_binary": torch.tensor(
                self.labels[idx], dtype=torch.float
            )
        }


def get_dataloader(csv_path, tokenizer, batch_size=32,
                   max_len=128, shuffle=False):

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    dataset = CommunalDataset(
        texts=df["Comment"].tolist(),
        labels=df["label"].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
