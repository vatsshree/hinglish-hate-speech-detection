
"""
Binary Hybrid Architecture: XLM-RoBERTa + BiLSTM
Output: 0 = non_communal, 1 = communal
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class HybridXLM_BiLSTM(nn.Module):
    def __init__(
        self,
        pretrained_model_name="xlm-roberta-base",
        lstm_hidden_size=256,
        dropout_rate=0.3,
        num_classes=1,
        freeze_encoder=False
    ):
        super(HybridXLM_BiLSTM, self).__init__()

        # Multilingual Transformer Backbone
        self.roberta = AutoModel.from_pretrained(pretrained_model_name)
        encoder_dim = self.roberta.config.hidden_size

        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Dense projection
        self.dense = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Single binary logit
        self.classifier_binary = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        _, (hn, _) = self.bilstm(sequence_output)

        h_fwd = hn[0]
        h_bwd = hn[1]
        pooled_seq = torch.cat([h_fwd, h_bwd], dim=1)

        dropped = self.dropout(pooled_seq)
        features = self.dense(dropped)

        logits = self.classifier_binary(features).squeeze(-1)

        return logits
