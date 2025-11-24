import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, subject_id):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    
    plt.title(f"Training vs Validation Loss — Subject {subject_id}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
