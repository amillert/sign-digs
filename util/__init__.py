import os
from typing import List
import torch


def join(dirpath: str, specific: str) -> List[str]:
    """
    Helper wrapper function around [[os.path.join]]
    """

    return os.path.join(dirpath, specific)


def accuracy(outputs, labels):
    """
    Helper function for calculating accuracy given predictions and reference data
    """

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
    """
    Helper function defining a training loop
    Returns the most recent model + list with all epochs cumulated losses
    """

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

            # perform forward pass through the model to obtain predictions
            labels_predicted = model(images)

            # compute the loss and perform backward pass to adjust weights
            loss = loss_fn(labels_predicted, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append the loss to the accumulated loss value of epoch
            loss_current_epoch += loss.item()
            # calculate the accuracy and store it
            acc = accuracy(labels_predicted, labels)
            epoch_accs.append(acc)

            # in verbose mode, each 10 mini-batches - display some additional information
            if verbose and not b_idx % 10:
                # compute the average accuracy, given the count of mini-batches passed in epoch
                avg_acc = sum(epoch_accs) / b_idx * 100.0
                print(f"Batch {b_idx} processed", end="; ")
                print(f"avg accuracy: {avg_acc:.03f}%")

        # store total accumulated loss per epoch
        loss_all_epochs.append(loss_current_epoch)

        # in a verbose mode, display info current epoch learning state
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
    """
    Helper function defining an evaluation step
    Returns the accuracy computed over the test set
    """

    model.to(device)
    model.eval()

    # no need to store, compute gradients in evaluation step
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
