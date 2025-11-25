import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_vr import VRSeqDataset
from conv_lstm_model import ConvLSTMNet
from train_seq import train_one_epoch, validate
from visualize_subject import plot_time_series

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_npz_files(base="processed"):
    return sorted([
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.endswith(".npz")
    ])


def run_loso_seq(seq_len=10, epochs=20, batch_size=32, lr=1e-4):
    all_files = get_npz_files()

    subject_ids = sorted(set(
        os.path.basename(f).split("_")[0]
        for f in all_files
    ))

    results = {}

    for test_subj in subject_ids:
        print(f"\n========== [SEQ] Testing on subject {test_subj} ==========")

        train_files = [f for f in all_files if not os.path.basename(f).startswith(test_subj)]
        test_files = [f for f in all_files if os.path.basename(f).startswith(test_subj)]

        if len(test_files) == 0:
            print(f"[SKIP] No files for subject {test_subj}")
            continue

        train_dataset = VRSeqDataset(train_files, seq_len=seq_len)
        test_dataset = VRSeqDataset(test_files, seq_len=seq_len)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"[SKIP] Not enough sequence data for subject {test_subj}")
            continue

        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Detect shape
        tmp = np.load(train_files[0])
        n_channels = tmp["eeg_trpsd"].shape[1]
        n_freqs = tmp["eeg_trpsd"].shape[2]
        tmp.close()

        print(f"[INFO] EEG shape for model: C={n_channels}, F={n_freqs}")

        model = ConvLSTMNet(
            n_channels=n_channels,
            n_freqs=n_freqs
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.L1Loss()

        # train
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            tl = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            vl = validate(model, val_loader, loss_fn, DEVICE)

            train_losses.append(tl)
            val_losses.append(vl)

            print(f"Epoch {epoch+1:02d}: train={tl:.4f} | val={vl:.4f}")

        from visualize_loss import plot_loss_curves

        #plot_loss_curves(train_losses, val_losses, test_subj)



        # test + visualize
        test_loss, preds, labels = validate(
            model, test_loader, loss_fn, DEVICE, return_preds=True
        )
        results[test_subj] = test_loss
        print(f"[RESULT] Subject {test_subj} Test MAE = {test_loss:.4f}")

        plot_time_series(preds.cpu(), labels.cpu(), test_subj)

    print("\n===== FINAL LOSO RESULTS (Conv-LSTM) =====")
    for s, v in results.items():
        print(f"{s}: {v:.4f}")


if __name__ == "__main__":
    run_loso_seq()
