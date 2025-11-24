import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_excel("../datasets/starboy/0001/0001_FN_EEG.xlsx")

# EEG channels
eeg_channels = [col for col in df.columns if "ch_" in col]

# ---------------------------
# 2. Function to compute bandpower
# ---------------------------
def bandpower(data, fs=256, band=(1, 40)):
    f, Pxx = welch(data, fs=fs, nperseg=256)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])

def extract_features(row):
    feats = {}
    signal = row[eeg_channels].values.astype(float)

    # Example: compute alpha, theta power for each channel
    for i, ch in enumerate(eeg_channels):
        data = np.array([row[ch]])

        feats[ch+"_theta"] = bandpower(data, band=(4, 7))
        feats[ch+"_alpha"] = bandpower(data, band=(8, 12))
        feats[ch+"_beta"] = bandpower(data, band=(13, 30))

    # Frontal asymmetry (example using ch_1 and ch_2)
    feats["frontal_asym"] = np.log(feats["ch_1_alpha"]+1e-8) - np.log(feats["ch_2_alpha"]+1e-8)

    # Simple statistical features
    feats["mean"] = np.mean(signal)
    feats["std"] = np.std(signal)
    feats["rms"] = np.sqrt(np.mean(signal**2))

    return feats

# ---------------------------
# 3. Feature Extraction
# ---------------------------
feature_list = []
for _, row in df.iterrows():
    feature_list.append(extract_features(row))

features = pd.DataFrame(feature_list)

# ---------------------------
# 4. Load labels
# ---------------------------
# Suppose you have a column 'Cybersickness'
labels = df["Cybersickness"]

# ---------------------------
# 5. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# ---------------------------
# 6. ML Model
# ---------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200))
])

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
