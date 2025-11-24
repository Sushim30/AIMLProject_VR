""" import numpy as np

data = np.load("processed/0001_FN.npz")

print(data.files)          # shows stored arrays
print(data['eeg_trpsd'].shape)
print(data['kinematic'].shape)
print(data['labels'])
print(data['freqs'].shape) """

""" import pandas as pd

df = pd.read_excel("../datasets/starboy/0001/0001_FN_EEG.xlsx")   # your file path

print(df.shape)          # (num_rows, num_columns)
print(df.columns)        # list all column names """

import numpy as np
import os

for f in os.listdir("processed"):
    if f.endswith(".npz"):
        d = np.load(f"processed/{f}")
        print(d["eeg_trpsd"].shape)

""" import numpy as np
import os

base = "processed"

for f in os.listdir(base):
    if f.endswith(".npz"):
        path = os.path.join(base, f)
        try:
            d = np.load(path)
            eeg = d["eeg_trpsd"]
            d.close()     # IMPORTANT: release file lock

            # Check for empty EEG (shape: W, C, F)
            if eeg.shape[1] == 0:
                print("Deleting invalid file:", f)
                os.remove(path)

        except Exception as e:
            print(f"Error reading {f}: {e}")
 """


