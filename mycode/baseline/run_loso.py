import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset_vr import VRDataset
from baseline_mlp import BaselineMLP
from train_baseline import train_one_epoch, validate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_npz_files(base="processed"):
    """Return all .npz files sorted."""
    return sorted(
        [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".npz")]
    )


def run_loso():
    all_files = get_npz_files()

    # Extract list of unique subjects (e.g., "0001", "0002")
    subject_ids = sorted(
        list(set(os.path.basename(f).split("_")[0] for f in all_files))
    )

    results = {}

    for test_subj in subject_ids:

        print(f"\n========== Testing on subject {test_subj} ==========")

        # Select training and test files
        train_files = [f for f in all_files if not os.path.basename(f).startswith(test_subj)]
        test_files = [f for f in all_files if os.path.basename(f).startswith(test_subj)]

        if len(test_files) == 0:
            print(f"[SKIP] No files found for subject {test_subj}")
            continue

        # Load dataset to verify it contains samples
        test_dataset_check = VRDataset(test_files, use_sequences=False)
        if len(test_dataset_check) == 0:
            print(f"[SKIP] Subject {test_subj} has NO valid samples")
            continue

        # Reload properly (avoid double-loading)
        train_dataset = VRDataset(train_files, use_sequences=False)
        test_dataset = test_dataset_check

        # Train/validation split
        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # ----------------------------------------------------
        # 🔍 Extract dynamic EEG shape (C, F)
        # ----------------------------------------------------
        import numpy as np
        tmp = np.load(train_files[0])
        n_channels = tmp["eeg_trpsd"].shape[1]
        n_freqs = tmp["eeg_trpsd"].shape[2]
        tmp.close()

        print(f"[INFO] EEG shape for model: channels={n_channels}, freqs={n_freqs}")

        # ----------------------------------------------------
        # Build model dynamically
        # ----------------------------------------------------
        model = BaselineMLP(n_channels=n_channels, n_freqs=n_freqs).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.L1Loss()   # MAE loss (like the paper)

        # ----------------------------------------------------
        # Training Loop
        # ----------------------------------------------------
        for epoch in range(15):
            tl = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            vl = validate(model, val_loader, loss_fn, DEVICE)
            print(f"Epoch {epoch+1}: train={tl:.4f} | val={vl:.4f}")

        # ----------------------------------------------------
        # Testing
        # ----------------------------------------------------
        test_loss, preds, labels = validate(
            model,
            test_loader,
            loss_fn,
            DEVICE,
            return_preds=True
        )

        results[test_subj] = test_loss
        print(f"[RESULT] Subject {test_subj} Test MAE = {test_loss:.4f}")

        from visualize_subject import plot_time_series, scatter_plot, error_histogram

        #plot_time_series(preds, labels, test_subj)
        #scatter_plot(preds, labels, test_subj)
        #error_histogram(preds, labels, test_subj)


    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    print("\n===== FINAL LOSO RESULTS =====")
    for subj, loss in results.items():
        print(f"{subj}: {loss:.4f}")


# ENTRY POINT
if __name__ == "__main__":
    run_loso()
