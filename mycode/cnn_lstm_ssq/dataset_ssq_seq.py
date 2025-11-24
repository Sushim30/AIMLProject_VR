import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VRSeqSSQDataset(Dataset):
    """
    Sequence dataset for predicting SSQ from EEG+kinematics.

    For each npz file (subject + condition):
        - find corresponding SSQ Total score
        - create sliding sequences over windows

    Each sample:
        eeg_seq : (L, C, F)
        kin_seq : (L, 16)
        label   : scalar SSQ Total (original scale)
    """

    def __init__(
        self,
        npz_paths: List[str],
        ssq_xlsx_path: str,
        seq_len: int = 10,
        label_type: str = "Total",
    ):
        """
        npz_paths     : list of processed npz files (one per subject-condition)
        ssq_xlsx_path : path to SSQ.Responses.xlsx
        seq_len       : sequence length in windows (e.g. 10 windows = 30 seconds)
        label_type    : which SSQ label to use: "Total", "Nausea", "Oculomotor", "Disorientation" or "all4"
        """
        self.seq_len = seq_len
        self.label_type = label_type
        self.samples = []

        # Load SSQ table
        ssq_df = pd.read_excel(ssq_xlsx_path)

        # Try to detect column names (adjust here if your headers differ)
        user_col_candidates = [c for c in ssq_df.columns if "UserId" in c]
        cond_col_candidates = [c for c in ssq_df.columns if "Condition" in c]

        if not user_col_candidates or not cond_col_candidates:
            raise ValueError("Could not find UserId / Condition columns in SSQ file")

        user_col = user_col_candidates[0]
        cond_col = cond_col_candidates[0]

        # Map (user_id_int, condition_str) → {Nausea, Oculomotor, Disorientation, Total}
        mapping = {}
        for _, row in ssq_df.iterrows():
            user_val = row[user_col]
            cond_val = str(row[cond_col]).strip()

            try:
                user_id_int = int(user_val)
            except Exception:
                continue

            entry = {
                "Nausea": row.get("Nausea", np.nan),
                "Oculomotor": row.get("Oculomotor", np.nan),
                "Disorientation": row.get("Disorentation", np.nan),  # typo in file
                "Total": row.get("Total", np.nan),
            }
            mapping[(user_id_int, cond_val)] = entry

        # Build samples
        for p in npz_paths:
            base = os.path.basename(p)        # e.g. "0001_FN.npz"
            subj_str, cond_ext = base.split("_")
            cond = cond_ext.split(".")[0]     # "FN"

            # Subject ids in SSQ may be 1,2,... while filenames are "0001"
            try:
                subj_id_int = int(subj_str)
            except Exception:
                print(f"[WARN] Could not parse subject id from {base}, skipping")
                continue

            if (subj_id_int, cond) not in mapping:
                print(f"[WARN] No SSQ row for subject {subj_id_int}, condition {cond}, skipping")
                continue

            ssq_entry = mapping[(subj_id_int, cond)]

            if self.label_type == "all4":
                label_val = np.array([
                    ssq_entry["Nausea"],
                    ssq_entry["Oculomotor"],
                    ssq_entry["Disorientation"],
                    ssq_entry["Total"],
                ], dtype=float)
            else:
                if self.label_type not in ssq_entry:
                    raise ValueError(f"Unknown label_type {self.label_type}")
                label_val = float(ssq_entry[self.label_type])

            if np.isnan(label_val).any() if isinstance(label_val, np.ndarray) else np.isnan(label_val):
                print(f"[WARN] NaN SSQ value for {subj_id_int}_{cond}, skipping")
                continue

            # Load npz data for this subject-condition
            data = np.load(p)
            eeg = data["eeg_trpsd"]      # (W, C, F)
            kin = data["kinematic"]      # (W, 16)
            data.close()

            W = eeg.shape[0]
            if W < seq_len:
                print(f"[WARN] Not enough windows ({W}) for {base}, need >= {seq_len}, skipping")
                continue

            # Sliding sequences; all share same SSQ label
            for t in range(W - seq_len + 1):
                self.samples.append({
                    "eeg": eeg[t:t+seq_len],    # (L, C, F)
                    "kin": kin[t:t+seq_len],    # (L, 16)
                    "label": label_val         # scalar or 4-dim vector
                })

        print(f"[VRSeqSSQDataset] Loaded {len(self.samples)} sequence samples (seq_len={self.seq_len}, label={self.label_type})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eeg = torch.tensor(s["eeg"], dtype=torch.float32)    # (L, C, F)
        kin = torch.tensor(s["kin"], dtype=torch.float32)    # (L, 16)

        label = s["label"]
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.float32)

        return eeg, kin, label
