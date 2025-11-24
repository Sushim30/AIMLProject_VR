import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, n_channels, n_freqs, kin_dim=16, hidden_dim=512):
        super().__init__()

        self.n_channels = n_channels
        self.n_freqs = n_freqs

        input_dim = n_channels * n_freqs + kin_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.out = nn.Linear(hidden_dim // 4, 1)

    def forward(self, eeg, kin):
        # eeg: (B, C, F)
        # kin: (B, 16)
        B = eeg.size(0)
        eeg_flat = eeg.view(B, -1)
        x = torch.cat([eeg_flat, kin], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))

        return x.squeeze(-1)
