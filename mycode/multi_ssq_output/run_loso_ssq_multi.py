# run_loso_ssq_multi.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_ssq_multi import VRSeqSSQMultiDataset
from conv_lstm_ssq_model import ConvLSTMSSQNet
from train_seq import train_one_epoch, validate  # same as before (vector-safe!)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_npz_files(base="processed"):
    return sorted([
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.endswith(".npz")
    ])


def run_loso_ssq_multi(
    ssq_xlsx_path: str,
    seq_len: int = 10,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
):
    all_files = get_npz_files()

    subject_ids = sorted(set(
        os.path.basename(f).split("_")[0] for f in all_files
    ))

    results = {}  # subject -> overall MAE

    for test_subj in subject_ids:
        print(f"\n========== [SSQ 4-Output] Testing on subject {test_subj} ==========")

        train_files = [f for f in all_files if not os.path.basename(f).startswith(test_subj)]
        test_files = [f for f in all_files if os.path.basename(f).startswith(test_subj)]

        if len(test_files) == 0:
            print(f"[SKIP] No files for subject {test_subj}")
            continue

        train_dataset = VRSeqSSQMultiDataset(train_files, ssq_xlsx_path, seq_len=seq_len)
        test_dataset = VRSeqSSQMultiDataset(test_files, ssq_xlsx_path, seq_len=seq_len)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"[SKIP] Not enough SSQ-labelled data for subject {test_subj}")
            continue

        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # EEG shape from one train file
        tmp = np.load(train_files[0])
        n_channels = tmp["eeg_trpsd"].shape[1]
        n_freqs = tmp["eeg_trpsd"].shape[2]
        tmp.close()

        print(f"[INFO] EEG shape: C={n_channels}, F={n_freqs}")

        model = ConvLSTMSSQNet(
            n_channels=n_channels,
            n_freqs=n_freqs
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.L1Loss()  # MAE over 4 outputs

        # ---- Training ----
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            tl = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            vl = validate(model, val_loader, loss_fn, DEVICE)

            train_losses.append(tl)
            val_losses.append(vl)

            print(f"Epoch {epoch+1:02d}: train={tl:.4f} | val={vl:.4f}")

        
        from visualize_loss import plot_loss_curves

        plot_loss_curves(train_losses, val_losses, test_subj)


        # ---- Test: get predictions and compute per-dim MAE ----
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for eeg, kin, label in test_loader:
                eeg, kin, label = eeg.to(DEVICE), kin.to(DEVICE), label.to(DEVICE)
                pred = model(eeg, kin)  # (B, 4)
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds, dim=0)   # (N, 4)
        all_labels = torch.cat(all_labels, dim=0) # (N, 4)
        print(all_labels)

        # Per-dimension MAE
        mae_vec = torch.mean(torch.abs(all_preds - all_labels), dim=0)  # (4,)
        overall_mae = torch.mean(mae_vec).item()

        results[test_subj] = overall_mae

        print(f"[RESULT] Subject {test_subj} MAE per subscale:")
        print(f"  Nausea       MAE: {mae_vec[0].item():.4f}")
        print(f"  Oculomotor   MAE: {mae_vec[1].item():.4f}")
        print(f"  Disorient.   MAE: {mae_vec[2].item():.4f}")
        print(f"  Total        MAE: {mae_vec[3].item():.4f}")
        print(f"  Overall Mean MAE: {overall_mae:.4f}")



    print("\n===== FINAL LOSO RESULTS (Conv-LSTM → 4 SSQ subscales) =====")
    for s, v in results.items():
        print(f"{s}: {v:.4f}")


if __name__ == "__main__":
    # Adjust this path to your actual SSQ.Responses.xlsx location
    # e.g. "../../SSQ.Responses.xlsx"
    ssq_path = "../SSQ.Responses.xlsx"
    run_loso_ssq_multi(ssq_xlsx_path=ssq_path)
