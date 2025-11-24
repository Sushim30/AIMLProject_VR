"""
Complete preprocessing pipeline for VR Cybersickness dataset.

Creates:
    processed/<subject>_<condition>.npz

Contents of npz:
    eeg_trpsd : (W, C, F)
    kinematic : (W, 16)
    labels    : (W,)
    freqs     : (F,)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import signal
import mne


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass
class PreprocessConfig:
    data_dir: str                   # directory containing raw .xlsx files
    output_dir: str = "processed"   # directory to save .npz
    subject_id: str = "0001"
    condition: str = "FN"

    fs_target: float = 100.0
    eeg_band: Tuple[float, float] = (1.0, 40.0)
    line_freq: float = 50.0
    window_sec: float = 3.0
    psd_fmin: float = 0.5
    psd_fmax: float = 40.0


# ------------------------------------------------------------
# TIME + INTERPOLATION UTILITIES
# ------------------------------------------------------------

def _build_time(df: pd.DataFrame) -> np.ndarray:
    t = df["Millis"].to_numpy(float) / 1000.0
    return t - t[0]


def _resample(times: np.ndarray,
              data: np.ndarray,
              t_grid: np.ndarray) -> np.ndarray:
    uniq_idx = np.unique(times, return_index=True)[1]
    times2 = times[uniq_idx]
    data2 = data[uniq_idx]
    out = np.zeros((len(t_grid), data.shape[1]))
    for i in range(data.shape[1]):
        out[:, i] = np.interp(t_grid, times2, data2[:, i])
    return out


# ------------------------------------------------------------
# EEG PREPROCESSING
# ------------------------------------------------------------

def preprocess_eeg(eeg_df: pd.DataFrame, cfg: PreprocessConfig):
    # Try format 1: ch_1, ch_2, ...
    eeg_cols = [c for c in eeg_df.columns if c.startswith("ch_")]

    # If empty, try format 2 (international 10-20 labels)
    if len(eeg_cols) == 0:
        possible_eeg_cols = [
            "P3","C3","F3","Fz","F4","C4","P4","Cz","Pz",
            "A1","Fp1","Fp2","T3","T5","O1","O2",
            "X3","X2","F7","F8","X1","A2","T6","T4"
        ]

        eeg_cols = [c for c in eeg_df.columns if c in possible_eeg_cols]

    # If *still* empty, assume all non-time columns are EEG:
    if len(eeg_cols) == 0:
        eeg_cols = [c for c in eeg_df.columns if c not in ("Millis","Hardware","HardwareSeconds")]

    eeg_data = eeg_df[eeg_cols].to_numpy(float)
    t_raw = _build_time(eeg_df)

    t_grid = np.arange(t_raw[0], t_raw[-1], 1.0 / cfg.fs_target)
    eeg_resampled = _resample(t_raw, eeg_data, t_grid)

    nyq = cfg.fs_target / 2
    b_bp, a_bp = signal.butter(
        4,
        [cfg.eeg_band[0] / nyq, cfg.eeg_band[1] / nyq],
        btype="bandpass"
    )
    eeg_bp = signal.filtfilt(b_bp, a_bp, eeg_resampled, axis=0)

    w0 = cfg.line_freq / nyq
    b_notch, a_notch = signal.iirnotch(w0, 30)
    eeg_filtered = signal.filtfilt(b_notch, a_notch, eeg_bp, axis=0)

    return eeg_filtered, t_grid


# ------------------------------------------------------------
# MULTITAPER PSD + TR-PSD
# ------------------------------------------------------------

def compute_multitaper_psd(eeg: np.ndarray, cfg: PreprocessConfig):
    fs = cfg.fs_target
    win_len = int(cfg.window_sec * fs)
    W = eeg.shape[0] // win_len

    eeg = eeg[:W * win_len]
    eeg_w = eeg.reshape(W, win_len, eeg.shape[1])

    psds = []
    for w in range(W):
        seg = eeg_w[w].T
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            seg,
            sfreq=fs,
            fmin=cfg.psd_fmin,
            fmax=cfg.psd_fmax,
            adaptive=True,
            normalization="full",
            verbose=False
        )
        psds.append(psd)

    return np.stack(psds, axis=0), freqs


def compute_trpsd(psd_all: np.ndarray, n_baseline: int = 3):
    W = psd_all.shape[0]
    baseline = psd_all[:n_baseline].mean(axis=0)

    tr = np.zeros_like(psd_all)
    for t in range(W):
        wp = max(0, t - 1)
        wn = min(W - 1, t + 1)
        avg_local = (psd_all[wp] + psd_all[t] + psd_all[wn]) / 3
        tr[t] = avg_local - baseline
    return tr


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

def build_labels(cs_df: pd.DataFrame, t_grid: np.ndarray, cfg: PreprocessConfig):
    rating = cs_df["Rating"].to_numpy(float)
    t_raw = _build_time(cs_df)
    rating_grid = np.interp(t_grid, t_raw, rating)

    fs = cfg.fs_target
    win_len = int(cfg.window_sec * fs)
    W = len(t_grid) // win_len

    rating_grid = rating_grid[:W * win_len]
    labels = rating_grid.reshape(W, win_len).mean(axis=1)

    for i in range(1, len(labels)):
        if abs(labels[i] - labels[i - 1]) < 0.1:
            labels[i] = labels[i - 1]

    return labels


# ------------------------------------------------------------
# KINEMATIC FEATURES
# ------------------------------------------------------------

def compute_kinematic(tf_df: pd.DataFrame, t_grid: np.ndarray, cfg: PreprocessConfig):
    pos = tf_df[["HeadPosition_X", "HeadPosition_Y", "HeadPosition_Z"]].to_numpy(float)
    rot = tf_df[["HeadRotation_Yaw", "HeadRotation_Pitch", "HeadRotation_Roll"]].to_numpy(float)
    t_raw = _build_time(tf_df)

    pos_g = _resample(t_raw, pos, t_grid)
    rot_g = _resample(t_raw, rot, t_grid)

    fs = cfg.fs_target
    win_len = int(cfg.window_sec * fs)
    W = len(t_grid) // win_len

    pos_g = pos_g[:W * win_len].reshape(W, win_len, 3)
    rot_g = rot_g[:W * win_len].reshape(W, win_len, 3)

    feats = np.zeros((W, 16))

    lam_prev = np.zeros(3)
    alp_prev = np.zeros(3)
    s_prev = 0.0
    w_prev = 0.0

    for w in range(W):
        p0, p1 = pos_g[w, 0], pos_g[w, -1]
        r0, r1 = rot_g[w, 0], rot_g[w, -1]

        lam = (p1 - p0) / cfg.window_sec
        alp = (r1 - r0) / cfg.window_sec

        s = np.linalg.norm(lam)
        wv = np.linalg.norm(alp)

        d_lam = lam - lam_prev
        d_alp = alp - alp_prev
        d_s = s - s_prev
        d_w = wv - w_prev

        feats[w] = np.concatenate([lam, alp, [s, wv], d_lam, d_alp, [d_s, d_w]])

        lam_prev, alp_prev, s_prev, w_prev = lam, alp, s, wv

    return feats


# ------------------------------------------------------------
# MAIN PREPROCESS FUNCTION
# ------------------------------------------------------------

def preprocess_subject(cfg: PreprocessConfig) -> Dict[str, np.ndarray]:
    prefix = f"{cfg.subject_id}_{cfg.condition}"

    f_eeg = os.path.join(cfg.data_dir, f"{prefix}_EEG.xlsx")
    f_tf = os.path.join(cfg.data_dir, f"{prefix}_Transforms.xlsx")
    f_cs = os.path.join(cfg.data_dir, f"{prefix}_SubjectiveCs.xlsx")

    eeg_df = pd.read_excel(f_eeg)
    tf_df = pd.read_excel(f_tf)
    cs_df = pd.read_excel(f_cs)

    eeg, t_grid = preprocess_eeg(eeg_df, cfg)

    fs = cfg.fs_target
    win_len = int(cfg.window_sec * fs)
    N = (len(t_grid) // win_len) * win_len
    eeg = eeg[:N]
    t_grid = t_grid[:N]

    psd_all, freqs = compute_multitaper_psd(eeg, cfg)
    trpsd = compute_trpsd(psd_all)

    kinematic = compute_kinematic(tf_df, t_grid, cfg)
    labels = build_labels(cs_df, t_grid, cfg)

    return {
        "eeg_trpsd": trpsd,
        "kinematic": kinematic,
        "labels": labels,
        "freqs": freqs
    }


# ------------------------------------------------------------
# SAVE WRAPPER
# ------------------------------------------------------------

def run_and_save(cfg: PreprocessConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    data = preprocess_subject(cfg)

    out_file = os.path.join(
        cfg.output_dir, f"{cfg.subject_id}_{cfg.condition}.npz"
    )

    np.savez(
        out_file,
        eeg_trpsd=data["eeg_trpsd"],
        kinematic=data["kinematic"],
        labels=data["labels"],
        freqs=data["freqs"],
    )

    print(f"[✓] Saved processed dataset → {out_file}")


# ------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------

""" if __name__ == "__main__":
    for i in ["0001","0002","0003","0004","0005","0006","0007","1000","1001","1002","1003","1004","1100","1101","1102","1192"]:
        for j in ["FN","FR","SR","SN","NN","NH","NT","HN","HH","HT","BL"]:
            try:
                cfg = PreprocessConfig(
                    data_dir="../datasets/starboy/"+i,
                    subject_id=i, 
                    condition=j,
                )
                
                run_and_save(cfg)
            except:
                print("there was some dikkat") """

bad = ["0002","0003"]
conditions = ["FN","FR","SN","SR"]

for subj in bad:
    for cond in conditions:
        try:
            cfg = PreprocessConfig(
                data_dir=f"../datasets/starboy/{subj}",
                subject_id=subj,
                condition=cond,
            )
            run_and_save(cfg)
            print(f"[FIXED] {subj}_{cond}")
        except Exception as e:
            print(f"[ERROR] {subj}_{cond} → {e}")



