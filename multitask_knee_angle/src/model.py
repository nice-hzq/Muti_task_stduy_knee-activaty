"""Multi-task model: shared CNN-BiLSTM encoder with classification + regression heads."""

import torch
import torch.nn as nn


class CNNLSTMMultiTask(nn.Module):
    """CNN + Bidirectional LSTM shared encoder with two task heads.

    Input:  ``[batch, channels, window]``
    Output: ``(cls_logits [batch, num_classes], knee_pred [batch, 1])``
    """

    def __init__(
        self,
        num_channels: int = 7,
        window_size: int = 128,
        num_classes: int = 8,
        conv_channels: list = None,
        conv_kernel: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = True,
        fc_hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64]

        # ── CNN encoder ───────────────────────────────────────────────
        cnn_layers = []
        in_ch = num_channels
        for out_ch in conv_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=conv_kernel,
                          padding=conv_kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_channels = in_ch

        # ── BiLSTM encoder ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        lstm_dir = 2 if lstm_bidirectional else 1
        self.lstm_out_dim = lstm_hidden_size * lstm_dir

        # ── Classification head ───────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes),
        )

        # ── Regression head ───────────────────────────────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: ``[batch_size, channels, window_size]``

        Returns:
            (cls_logits, knee_pred) tuple
        """
        # CNN: [B, C, W] -> [B, cnn_out, W]
        x = self.cnn(x)

        # LSTM expects [B, W, F]; permute
        x = x.permute(0, 2, 1)                           # [B, W, cnn_out]
        lstm_out, _ = self.lstm(x)                        # [B, W, lstm_out_dim]

        # Global average pooling over time
        pooled = lstm_out.mean(dim=1)                     # [B, lstm_out_dim]

        cls_logits = self.cls_head(pooled)                # [B, num_classes]
        knee_pred = self.reg_head(pooled)                 # [B, 1]

        return cls_logits, knee_pred
