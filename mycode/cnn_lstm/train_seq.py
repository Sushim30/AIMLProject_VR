import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for eeg, kin, label in loader:
        eeg, kin, label = eeg.to(device), kin.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(eeg, kin)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * eeg.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    preds_list, labels_list = [], []

    with torch.no_grad():
        for eeg, kin, label in loader:
            eeg, kin, label = eeg.to(device), kin.to(device), label.to(device)

            pred = model(eeg, kin)
            loss = loss_fn(pred, label)
            total_loss += loss.item() * eeg.size(0)

            if return_preds:
                preds_list.append(pred.cpu())
                labels_list.append(label.cpu())

    avg_loss = total_loss / len(loader.dataset)

    if return_preds:
        return avg_loss, torch.cat(preds_list), torch.cat(labels_list)

    return avg_loss
