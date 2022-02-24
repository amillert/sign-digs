#!/usr/bin/env python

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.cnn import SignDigitsCNN
import util
from util.preprocess import CustomImageDataset
from util.argparser import get_cli_args


def main() -> None:
    args, subparser = get_cli_args()

    datapath = os.path.abspath(args.in_path)
    dataset = CustomImageDataset(datapath)
    data_len = len(dataset)
    test_size = int((1 - args.train_size) * data_len)
    train_size = data_len - test_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    BATCH_SIZE = args.batch_size
    CPUs = os.cpu_count()
    train_dl = DataLoader(trainset, BATCH_SIZE, True, num_workers=CPUs)
    test_dl = DataLoader(testset, BATCH_SIZE, num_workers=CPUs)

    loss_fn = nn.CrossEntropyLoss()
    model = SignDigitsCNN()

    if subparser == "train":
        NUM_EPOCHS = args.epochs
        LR = args.lr

        model, _ = util.training(model, train_dl, NUM_EPOCHS, loss_fn, LR)

        torch.save(model.state_dict(), args.out_model)

    elif subparser == "eval":
        model.load_state_dict(torch.load(args.in_model))

    acc = util.eval_cnn_classifier(model, test_dl, "cpu")
    print("Accuracy of the network on the test images: {:.03f}%".format(acc))


if __name__ == "__main__":
    main()
