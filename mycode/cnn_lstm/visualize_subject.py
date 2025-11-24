import matplotlib.pyplot as plt


def plot_time_series(preds, labels, subject_id):
    preds = preds.numpy()
    labels = labels.numpy()

    plt.figure(figsize=(14, 4))
    plt.plot(labels, color="black", label="Actual")
    plt.plot(preds, color="tab:blue", label="Predicted", alpha=0.7)
    plt.title(f"Cybersickness Prediction — Subject {subject_id}")
    plt.xlabel("Window index (3s per window)")
    plt.ylabel("Cybersickness level (0–1)")
    plt.legend()
    plt.grid(True)
    plt.show()
