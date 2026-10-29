from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """单变量 LSTM + MLP 回归头。"""

    def __init__(self, configs) -> None:
        super().__init__()
        self.enc_in = configs.enc_in
        hidden = configs.d_model
        num_layers = configs.e_layers
        dropout = configs.dropout
        self.use_cudnn = bool(int(getattr(configs, "lstm_use_cudnn", 0)))

        if self.enc_in < 1:
            raise ValueError("enc_in must be >= 1")
        if hidden < 1 or num_layers < 1:
            raise ValueError("lstm_hidden and lstm_layers must be >= 1")

        # self.input_norm = nn.LayerNorm(self.enc_in)
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # feature_dim = hidden * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: [B, L, C]`` -> ``y_hat: [B, target_dim]``。"""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch, lookback_length, num_channels], "
                f"got shape={tuple(x.shape)}"
            )
        if x.shape[-1] != self.enc_in:
            raise ValueError(
                f"expected {self.enc_in} input channels, got {x.shape[-1]}"
            )

        # normalized = self.input_norm(x)
        if x.is_cuda and not self.use_cudnn:
            # 避开当前 torch/cuDNN 组合在 Windows 上的 RNN 反向崩溃（见文件头说明）
            with torch.backends.cudnn.flags(enabled=False):
                _, (h_n, _) = self.lstm(x)
        else:
            _, (h_n, _) = self.lstm(x)
        feature = h_n[-1]
        return self.head(feature)