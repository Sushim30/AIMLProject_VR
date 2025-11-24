# dataset_ssq_multi.py

import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VRSeqSSQMultiDataset(Dataset):
    """
    Sequence dataset for multi-output SSQ prediction.
    For each subject-condition npz:
      - find SSQ row in SSQ.Responses.xlsx
      - label is [Nausea, Oculomotor, Disorientation, Total] (4-dim vector)
      - build sliding sequences over windows

    Each sample:
        eeg_seq : (L, C, F)
        kin_seq : (L, 16)
        label   : (4,) tensor
    """

    def __init__(
        self,
        npz_paths: List[str],
        ssq_xlsx_path: str,
        seq_len: int = 10,
    ):
        self.seq_len = seq_len
        self.samples = []

        # Load SSQ table
        ssq_df = pd.read_excel(ssq_xlsx_path)

        # Detect Id and Condition columns (adapt if your header differs)
        user_col_candidates = [c for c in ssq_df.columns if "UserId" in c]
        cond_col_candidates = [c for c in ssq_df.columns if "Condition" in c]

        if not user_col_candidates or not cond_col_candidates:
            raise ValueError("Could not find UserId / Condition columns in SSQ file")

        user_col = user_col_candidates[0]
        cond_col = cond_col_candidates[0]

        # Build mapping: (subject_id_int, condition_str) -> 4-dim SSQ vector
        mapping = {}
        for _, row in ssq_df.iterrows():
            user_val = row[user_col]
            cond_val = str(row[cond_col]).strip()

            try:
                subj_id_int = int(user_val)
            except Exception:
                continue

            nausea = row.get("Nausea", np.nan)
            oculo = row.get("Oculomotor", np.nan)
            # note the typo "Disorentation" in original file
            disor = row.get("Disorentation", np.nan)
            total = row.get("Total", np.nan)

            label_vec = np.array([nausea, oculo, disor, total], dtype=float)
            if np.isnan(label_vec).any():
                # skip rows with NaNs in any subscale
                continue

            mapping[(subj_id_int, cond_val)] = label_vec

        # Build samples from each npz
        for p in npz_paths:
            base = os.path.basename(p)      # e.g. "0001_FN.npz"
            subj_str, cond_ext = base.split("_")
            cond = cond_ext.split(".")[0]   # "FN"

            # Convert "0001" -> 1 to match SSQ UserId
            try:
                subj_id_int = int(subj_str)
            except Exception:
                print(f"[WARN] Could not parse subject id from {base}, skipping")
                continue

            key = (subj_id_int, cond)
            if key not in mapping:
                print(f"[WARN] No SSQ entry for subject {subj_id_int}, condition {cond}, skipping {base}")
                continue

            label_vec = mapping[key]  # (4,)

            data = np.load(p)
            eeg = data["eeg_trpsd"]      # (W, C, F)
            kin = data["kinematic"]      # (W, 16)
            data.close()

            W = eeg.shape[0]
            if W < seq_len:
                print(f"[WARN] Not enough windows ({W}) for {base}, need >= {seq_len}, skipping")
                continue

            # Sliding sequences, all share same 4-dim SSQ label
            for t in range(W - seq_len + 1):
                self.samples.append({
                    "eeg": eeg[t:t+seq_len],   # (L, C, F)
                    "kin": kin[t:t+seq_len],   # (L, 16)
                    "label": label_vec         # (4,)
                })

        print(f"[VRSeqSSQMultiDataset] Loaded {len(self.samples)} sequence samples (seq_len={self.seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eeg = torch.tensor(s["eeg"], dtype=torch.float32)     # (L, C, F)
        kin = torch.tensor(s["kin"], dtype=torch.float32)     # (L, 16)
        label = torch.tensor(s["label"], dtype=torch.float32) # (4,)
        return eeg, kin, label
