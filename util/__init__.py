import os
from typing import List
import torch


def join(dirpath: str, specific: str) -> List[str]:
    return os.path.join(dirpath, specific)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def training(
    model,
    train_dataloader,
    num_epochs: int,
    loss_fn,
    learning_rate: float,
    device: str = "cpu",
    verbose: bool = True,
):
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_all_epochs = []

    print(f"Upcoming batches: {len(train_dataloader)}")
    print()

    for epoch in range(1, num_epochs + 1):
        loss_current_epoch = 0
        epoch_accs = []

        for b_idx, (images, labels) in enumerate(train_dataloader, 1):
            images = images.to(device)
            labels = labels.to(device)

            labels_predicted = model(images)

            loss = loss_fn(labels_predicted, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_current_epoch += loss.item()
            acc = accuracy(labels_predicted, labels)
            epoch_accs.append(acc)

            if verbose and not b_idx % 10:
                avg_acc = sum(epoch_accs) / b_idx * 100.0
                print(f"Batch {b_idx} processed", end="; ")
                print(f"avg accuracy: {avg_acc:.03f}%")

        loss_all_epochs.append(loss_current_epoch)

        if verbose:
            avg_acc = sum(epoch_accs) / b_idx * 100.0
            print("—" * 50)
            print(f"| Epoch [{epoch}/{num_epochs}]")
            print(f"| avg accuracy: {avg_acc:.03f}% ", end="; ")
            print("avg loss: {:.03f}".format(loss_current_epoch / b_idx))
            print(f"| recent accuracy: {acc * 100.0:.03f}% ", end="; ")
            print(f"total loss: {loss_current_epoch:.03f}")
            print("—" * 50, end="\n" * 2)

    return model, loss_all_epochs


def eval_cnn_classifier(model, eval_dataloader, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in eval_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            y_predicted = model(images)
            _, label_predicted = torch.max(y_predicted.data, 1)

            total += labels.size(0)
            correct += (label_predicted == labels).sum().item()

    return 100 * correct / total
