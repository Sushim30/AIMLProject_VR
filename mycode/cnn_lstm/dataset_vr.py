import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VRDataset(Dataset):
    """
    Single-window dataset (used by MLP baseline).
    Not used in Conv-LSTM version, but kept for compatibility.
    """

    def __init__(self, npz_paths, use_sequences=False, seq_len=10):
        self.use_sequences = use_sequences
        self.seq_len = seq_len
        self.samples = []

        for p in npz_paths:
            data = np.load(p)
            eeg = data["eeg_trpsd"]      # (W, C, F)
            kin = data["kinematic"]      # (W, 16)
            labels = data["labels"]      # (W,)
            data.close()

            W = len(labels)

            if use_sequences:
                if W < seq_len:
                    continue
                for t in range(W - seq_len + 1):
                    self.samples.append({
                        "eeg": eeg[t:t+seq_len],
                        "kin": kin[t:t+seq_len],
                        "label": labels[t+seq_len-1]
                    })
            else:
                for t in range(W):
                    self.samples.append({
                        "eeg": eeg[t],
                        "kin": kin[t],
                        "label": labels[t]
                    })

        print(f"[VRDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eeg = torch.tensor(s["eeg"], dtype=torch.float32)
        kin = torch.tensor(s["kin"], dtype=torch.float32)
        label = torch.tensor(s["label"], dtype=torch.float32)
        return eeg, kin, label


class VRSeqDataset(Dataset):
    """
    Sequence dataset for Conv-LSTM model.
    Each sample returns:
        eeg_seq : (L, C, F)
        kin_seq : (L, 16)
        label   : scalar
    """

    def __init__(self, npz_paths, seq_len=10):
        self.seq_len = seq_len
        self.samples = []

        for p in npz_paths:
            data = np.load(p)
            eeg = data["eeg_trpsd"]      # (W, C, F)
            kin = data["kinematic"]      # (W, 16)
            labels = data["labels"]      # (W,)
            data.close()

            W = len(labels)
            if W < seq_len:
                continue

            for t in range(W - seq_len + 1):
                self.samples.append({
                    "eeg": eeg[t:t+seq_len],
                    "kin": kin[t:t+seq_len],
                    "label": labels[t + seq_len - 1]
                })

        print(f"[VRSeqDataset] Loaded {len(self.samples)} sequence samples (seq_len={self.seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eeg = torch.tensor(s["eeg"], dtype=torch.float32)   # (L, C, F)
        kin = torch.tensor(s["kin"], dtype=torch.float32)   # (L, 16)
        label = torch.tensor(s["label"], dtype=torch.float32)
        return eeg, kin, label
