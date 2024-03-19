"""
argument parser module
"""

import argparse

def parse_arguments():
    """
    Function parses arguments from command line

    :return: program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action="store_true", help="Disables CUDA training.")
    parser.add_argument("--train", action="store_true", help="train network")
    parser.add_argument("--NNmodel", type=int, action="store", default=0, nargs='?',
                        help="Set type of NN (default = 0 (ModifiedLeNet)).\n"
                             "1 = VGG"
                             "2 = depthModel")
    parser.add_argument("--dataset", type=int, action="store", default=0, nargs='?',
                        help="Set type of training set (default = 0 (MNIST)).\n"
                             "1 = Cifar-10\n"
                             "2 = Cifar-100"
                             "3 = cityScapes")
    parser.add_argument("--dropout", type=int, action="store", default=0, help="Apply dropout regularization")
    parser.add_argument("--adam", action="store_true",  help="Adam instead of SGD")
    parser.add_argument("--epochs", type=int, action="store", default=14, nargs='?',
                        help="Set number of training epochs (default = 14).")

    args = parser.parse_args()

    return args
