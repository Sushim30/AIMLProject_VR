import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_ssq_seq import VRSeqSSQDataset
from conv_lstm_model import ConvLSTMNet
from train_seq import train_one_epoch, validate
from visualize_subject import plot_time_series  # optional for seeing per-sequence fit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_npz_files(base="processed"):
    return sorted([
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.endswith(".npz")
    ])


def run_loso_ssq(
    ssq_xlsx_path: str,
    seq_len: int = 10,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    all_files = get_npz_files()

    # subjects like "0001", "0002", ...
    subject_ids = sorted(set(
        os.path.basename(f).split("_")[0] for f in all_files
    ))

    results = {}

    for test_subj in subject_ids:
        print(f"\n========== [SSQ] Testing on subject {test_subj} ==========")

        train_files = [f for f in all_files if not os.path.basename(f).startswith(test_subj)]
        test_files = [f for f in all_files if os.path.basename(f).startswith(test_subj)]

        if len(test_files) == 0:
            print(f"[SKIP] No files for subject {test_subj}")
            continue

        train_dataset = VRSeqSSQDataset(train_files, ssq_xlsx_path, seq_len=seq_len, label_type="Total")
        test_dataset = VRSeqSSQDataset(test_files, ssq_xlsx_path, seq_len=seq_len, label_type="Total")

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"[SKIP] Not enough SSQ-labelled data for subject {test_subj}")
            continue

        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Shape from any train file
        tmp = np.load(train_files[0])
        n_channels = tmp["eeg_trpsd"].shape[1]
        n_freqs = tmp["eeg_trpsd"].shape[2]
        tmp.close()

        print(f"[INFO] EEG shape: C={n_channels}, F={n_freqs}")

        model = ConvLSTMNet(
            n_channels=n_channels,
            n_freqs=n_freqs
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.L1Loss()   # MAE in SSQ units

        # ---- Training loop ----
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

        # ---- Test ----
        test_loss, preds, labels = validate(
            model, test_loader, loss_fn, DEVICE, return_preds=True
        )
        results[test_subj] = test_loss
        print(f"[RESULT] Subject {test_subj} Test MAE (SSQ Total) = {test_loss:.4f}")

        # Optional: visualize how well it fits per-sequence (normalized just for plotting)
        try:
            # normalize for nicer plot (0–1) just visually
            p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
            l_norm = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
            plot_time_series(p_norm.cpu(), l_norm.cpu(), f"{test_subj} (SSQ Total, normalized)")
        except Exception as e:
            print(f"[WARN] Could not plot for subject {test_subj}: {e}")

    print("\n===== FINAL LOSO RESULTS (Conv-LSTM → SSQ Total) =====")
    for s, v in results.items():
        print(f"{s}: {v:.4f}")


if __name__ == "__main__":
    # change this to your actual path:
    # e.g. "../datasets/starboy/SSQ.Responses.xlsx"
    ssq_path = "../SSQ.Responses.xlsx"
    run_loso_ssq(ssq_xlsx_path=ssq_path)
