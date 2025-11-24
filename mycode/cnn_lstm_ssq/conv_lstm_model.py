import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMNet(nn.Module):
    """
    EEG:
        (B, L, C, F)
         -> Conv1D over freq
         -> GAP -> (B, L, convC)
         -> LSTM -> hidden EEG

    KIN:
        (B, L, 16)
         -> LSTM -> hidden KIN

    Fusion:
         [EEG_hidden || KIN_hidden]
         -> FC -> Sigmoid
    """

    def __init__(self,
                 n_channels,
                 n_freqs,
                 kin_dim=16,
                 eeg_conv_channels=32,
                 eeg_lstm_hidden=32,
                 kin_lstm_hidden=32,
                 fc_hidden=64):
        super().__init__()

        # EEG conv over freq
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
        self.out = nn.Linear(fc_hidden // 2, 1)

    def forward(self, eeg_seq, kin_seq):
        B, L, C, Freq = eeg_seq.shape

        # EEG branch
        eeg_seq = eeg_seq.view(B * L, C, Freq)
        eeg_seq = F.relu(self.eeg_conv(eeg_seq))
        eeg_seq = eeg_seq.mean(dim=2)            # GAP over freq → (B*L, convC)
        eeg_seq = eeg_seq.view(B, L, -1)

        _, (h_eeg, _) = self.eeg_lstm(eeg_seq)
        eeg_feat = h_eeg[-1]                     # (B, hidden_eeg)

        # Kinematic branch
        _, (h_kin, _) = self.kin_lstm(kin_seq)
        kin_feat = h_kin[-1]                     # (B, hidden_kin)

        # Fusion
        fusion = torch.cat([eeg_feat, kin_feat], dim=1)
        x = F.relu(self.fc1(fusion))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))

        return x.squeeze(-1)
