import argparse
import sys


def get_cli_args():
    args = sys.argv[1:]
    argument_parser = argparse.ArgumentParser(
        prog="sign digits detection",
        description="Argument parser of the Sign Digits CNN detection",
        allow_abbrev=True,
    )
    argument_parser.version = "0.1"

    subparsers = argument_parser.add_subparsers()

    train_subparser = subparsers.add_parser("train", help="Subparser for CNN training.")

    train_subparser.add_argument(
        "-i",
        "--in_path",
        action="store",
        type=str,
        help="Provide an input data path",
        required=True,
    )
    train_subparser.add_argument(
        "-o",
        "--out_model",
        action="store",
        type=str,
        help="Provide an output model path",
        required=True,
    )
    train_subparser.add_argument(
        "-s",
        "--train_size",
        action="store",
        type=float,
        help="Provide train data split as float",
        required=True,
    )
    train_subparser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        type=int,
        help="Provide the size of mini-batch",
        required=True,
    )
    train_subparser.add_argument(
        "-x",
        "--epochs",
        action="store",
        type=int,
        help="Provide the amount of epochs to train",
        required=True,
    )
    train_subparser.add_argument(
        "-l",
        "--lr",
        action="store",
        type=float,
        help="Provide learning rate for classification model",
        required=True,
    )

    eval_subparser = subparsers.add_parser("eval", help="Subparser for CNN evaluation.")

    eval_subparser.add_argument(
        "-i",
        "--in_path",
        action="store",
        type=str,
        help="Provide an input data path",
        required=True,
    )
    eval_subparser.add_argument(
        "-m",
        "--in_model",
        action="store",
        type=str,
        help="Provide an input model path to load",
        required=True,
    )
    eval_subparser.add_argument(
        "-s",
        "--train_size",
        action="store",
        type=float,
        help="Provide train data split as float",
        required=True,
    )
    eval_subparser.add_argument(
        "-b",
        "--batch_size",
        action="store",
        type=int,
        help="Provide the size of mini-batch",
        required=True,
    )
    eval_subparser.add_argument(
        "-x",
        "--epochs",
        action="store",
        type=int,
        help="Provide the amount of epochs to train",
        required=True,
    )
    eval_subparser.add_argument(
        "-l",
        "--lr",
        action="store",
        type=float,
        help="Provide learning rate for classification model",
        required=True,
    )

    return argument_parser.parse_args(args), args[0]
