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
    """
    The main function, defines a starting point of the script
    """

    # load script arguments
    args, subparser = get_cli_args()

    # load the dataset
    datapath = os.path.abspath(args.in_path)
    dataset = CustomImageDataset(datapath)

    # split the dataset into train/test subsets
    data_len = len(dataset)
    test_size = int((1 - args.train_size) * data_len)
    train_size = data_len - test_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # DataLoader object for iterating through the minibatches of data
    BATCH_SIZE = args.batch_size
    CPUs = os.cpu_count()
    train_dl = DataLoader(trainset, BATCH_SIZE, True, num_workers=CPUs)
    test_dl = DataLoader(testset, BATCH_SIZE, num_workers=CPUs)

    """
    Cross-entropy loss measures the performance of a classification model,
    whose output is in range [0, 1]; a common choice for classifiers
    """
    loss_fn = nn.CrossEntropyLoss()
    model = SignDigitsCNN()

    # perform training if stated in the arguments
    if subparser == "train":
        NUM_EPOCHS = args.epochs
        LR = args.lr

        model, _ = util.training(model, train_dl, NUM_EPOCHS, loss_fn, LR)

        torch.save(model.state_dict(), args.out_model)

    # otherwise (if so stated) - load the model
    elif subparser == "eval":
        model.load_state_dict(torch.load(args.in_model))

    # if neither mode provided - raise error
    else:
        raise Exception(
            "Please use either `train` mode to train the model or `eval` for evaluation if you want to test your model's performance"
        )

    # both in evaluation and after training mode - perform the evaluation step
    acc = util.eval_cnn_classifier(model, test_dl, "cpu")
    print("Accuracy of the network on the test images: {:.03f}%".format(acc))


if __name__ == "__main__":
    main()
