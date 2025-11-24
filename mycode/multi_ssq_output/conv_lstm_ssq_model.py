# conv_lstm_ssq_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMSSQNet(nn.Module):
    """
    Conv-LSTM network for multi-output SSQ prediction.

    EEG:
      (B, L, C, F) → Conv1D over freq → GAP → LSTM over time → EEG feature
    KIN:
      (B, L, 16)   → LSTM over time → KIN feature

    Fusion:
      concat(EEG_feat, KIN_feat) → FC → 4-d output (Nausea, Oculomotor, Disorientation, Total)
    """

    def __init__(
        self,
        n_channels,
        n_freqs,
        kin_dim: int = 16,
        eeg_conv_channels: int = 32,
        eeg_lstm_hidden: int = 32,
        kin_lstm_hidden: int = 32,
        fc_hidden: int = 64,
        out_dim: int = 4,  # 4 SSQ subscales
    ):
        super().__init__()

        self.eeg_conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=eeg_conv_channels,
            kernel_size=3,
            padding=1
        )

        self.eeg_lstm = nn.LSTM(
            input_size=eeg_conv_channels,
            hidden_size=eeg_lstm_hidden,
            batch_first=True
        )

        self.kin_lstm = nn.LSTM(
            input_size=kin_dim,
            hidden_size=kin_lstm_hidden,
            batch_first=True
        )

        fusion_in = eeg_lstm_hidden + kin_lstm_hidden

        self.fc1 = nn.Linear(fusion_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.out = nn.Linear(fc_hidden // 2, out_dim)

    def forward(self, eeg_seq, kin_seq):
        """
        eeg_seq: (B, L, C, F)
        kin_seq: (B, L, 16)
        returns: (B, 4)  -> [Nausea, Oculomotor, Disorientation, Total]
        """
        B, L, C, Freq = eeg_seq.shape

        # EEG branch
        x_eeg = eeg_seq.view(B * L, C, Freq)     # (B*L, C, F)
        x_eeg = F.relu(self.eeg_conv(x_eeg))     # (B*L, convC, F)
        x_eeg = x_eeg.mean(dim=2)                # GAP over freq → (B*L, convC)
        x_eeg = x_eeg.view(B, L, -1)             # (B, L, convC)

        _, (h_eeg, _) = self.eeg_lstm(x_eeg)     # h_eeg: (1, B, H_eeg)
        eeg_feat = h_eeg[-1]                     # (B, H_eeg)

        # Kinematic branch
        _, (h_kin, _) = self.kin_lstm(kin_seq)   # h_kin: (1, B, H_kin)
        kin_feat = h_kin[-1]                     # (B, H_kin)

        # Fusion
        fusion = torch.cat([eeg_feat, kin_feat], dim=1)  # (B, H_eeg+H_kin)

        x = F.relu(self.fc1(fusion))
        x = F.relu(self.fc2(x))
        x = self.out(x)  # (B, 4)  -- NO SIGMOID: direct regression in SSQ units

        return x
