#!/usr/bin/env python

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.cnn import SignDigitsCNN
import util
from util.preprocess import CustomImageDataset


if __name__ == "__main__":
    datapath = os.path.abspath("data")
    dataset = CustomImageDataset(datapath)
    data_len = len(dataset)
    test_size = int(0.05 * data_len)
    train_size = data_len - test_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    BATCH_SIZE = 16
    CPUs = os.cpu_count()
    train_dl = DataLoader(trainset, BATCH_SIZE, True, num_workers=CPUs)
    test_dl = DataLoader(testset, BATCH_SIZE, num_workers=CPUs)

    loss_fn = nn.CrossEntropyLoss()
    model = SignDigitsCNN()

    NUM_EPOCHS = 10
    LR = 2e-4

    # model, loss_total = util.training(model, train_dl, NUM_EPOCHS, loss_fn, LR)

    # torch.save(model.state_dict(), "model_cnn_classif.pt")

    # acc = util.eval_cnn_classifier(model, test_dl, "cpu")
    # print("Accuracy of the network on the test images: {:.03f}%".format(acc))

    model.load_state_dict(torch.load("model_cnn_classif.pt"))
    acc = util.eval_cnn_classifier(model, test_dl, "cpu")
    print("Accuracy of the network on the test images: {:.03f}%".format(acc))
