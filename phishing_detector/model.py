"""Нейромережева архітектура: Char-CNN + BiLSTM + Attention.

Обґрунтування вибору:
* символьне представлення не потребує ручних ознак і стійке до нових доменів;
* згорткові шари з різними ядрами (3/5/7) виділяють локальні n-грами
  (напр. «paypa», «-login», «.com.», «xn--»);
* двонапрямна LSTM моделює послідовні залежності між частинами URL
  (домен → шлях → параметри);
* механізм уваги + max-pooling агрегують інформацію з усієї послідовності.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PhishingNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, conv_filters: int = 64,
                 kernel_sizes=(3, 5, 7), lstm_hidden: int = 64, dense_dim: int = 64,
                 dropout: float = 0.35):
        super().__init__()
        kernel_sizes = tuple(kernel_sizes)
        if any(k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes мають бути непарними (для збереження довжини послідовності)")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout / 2)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, conv_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        conv_out = conv_filters * len(kernel_sizes)
        self.conv_norm = nn.BatchNorm1d(conv_out)
        self.lstm = nn.LSTM(conv_out, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * lstm_hidden, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) індекси символів; повертає логіти (B,)."""
        mask = x != 0
        mask[:, 0] = True  # гарантуємо принаймні один «справжній» токен
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item())
        x, mask = x[:, :max_len], mask[:, :max_len]  # відсікаємо зайвий PAD у батчі

        emb = self.embed_dropout(self.embedding(x)).transpose(1, 2)  # (B, E, L)
        conv = torch.cat([F.relu(conv(emb)) for conv in self.convs], dim=1)  # (B, C, L)
        conv = self.conv_norm(conv).transpose(1, 2)  # (B, L, C)

        packed = pack_padded_sequence(conv, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)  # (B, L, 2H)

        scores = self.attn(out).squeeze(-1).masked_fill(~mask, -1e9)  # (B, L)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)  # (B, 2H)
        max_pool = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values  # (B, 2H)

        features = torch.cat([context, max_pool], dim=1)
        return self.classifier(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
