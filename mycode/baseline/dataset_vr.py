import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VRDataset(Dataset):
    """
    Loads a single processed npz file and returns:
        eeg_trpsd : (C, F)
        kinematic : (16,)
        label     : scalar
    """

    def __init__(self, npz_paths, seq_len=10, use_sequences=True):
        """
        npz_paths : list of paths to processed npz files
        seq_len   : number of windows in each sequence (LSTM input)
        use_sequences : if False → return single windows instead of sequences
        """
        self.use_sequences = use_sequences
        self.seq_len = seq_len

        self.samples = []  # list of dict {eeg, kin, label}

        for p in npz_paths:
            data = np.load(p)

            eeg = data['eeg_trpsd']       # (W, C, F)
            kin = data['kinematic']       # (W, 16)
            labels = data['labels']       # (W,)

            W = len(labels)

            if use_sequences:
                # sequence windows: (t, t+1, ..., t+seq_len-1)
                for t in range(W - seq_len):
                    self.samples.append({
                        'eeg': eeg[t:t+seq_len],
                        'kin': kin[t:t+seq_len],
                        'label': labels[t+seq_len-1]
                    })
            else:
                for t in range(W):
                    self.samples.append({
                        'eeg': eeg[t],
                        'kin': kin[t],
                        'label': labels[t]
                    })

        print(f"[VRDataset] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        eeg = torch.tensor(s['eeg'], dtype=torch.float32)
        kin = torch.tensor(s['kin'], dtype=torch.float32)
        label = torch.tensor(s['label'], dtype=torch.float32)


        return eeg, kin, label
