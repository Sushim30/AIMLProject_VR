import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_time_series(preds, labels, subject_id):
    preds = preds.numpy()
    labels = labels.numpy()

    plt.figure(figsize=(14, 5))
    plt.plot(labels, label="Actual", color="black")
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.title(f"Cybersickness Prediction — Subject {subject_id}")
    plt.xlabel("Window index (3s per window)")
    plt.ylabel("Cybersickness level (0–1)")
    plt.legend()
    plt.grid(True)
    plt.show()

def scatter_plot(preds, labels, subject_id):
    preds = preds.numpy()
    labels = labels.numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(labels, preds, alpha=0.4)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Scatter: Prediction Accuracy — Subject {subject_id}")
    plt.grid(True)
    plt.plot([0,1],[0,1],"r--")  # perfect prediction line
    plt.show()

def error_histogram(preds, labels, subject_id):
    errors = (preds - labels).numpy()

    plt.figure(figsize=(8,5))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title(f"Error Distribution — Subject {subject_id}")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()
