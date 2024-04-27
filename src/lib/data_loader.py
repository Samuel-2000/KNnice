import torch.utils.data
from . import paths
from torch import utils as utils
from torch.utils.data import Subset
from .cityscapes import CityscapesDataset
from .cityscapes import download_cityscapes


def data_load(args):
    """
    Function prepares and loads data
    :param args: dataset name enum arg
    :return: train loader, test loader
    """
    if args.dataset == 0:
        # Instantiate CityscapesDataset
        if not (paths.depth.exists()):
            download_cityscapes()

        cityscapes_dataset = CityscapesDataset(paths.depth, paths.image)

        # Create subsets
        train_set = Subset(cityscapes_dataset, range(len(cityscapes_dataset)))
        test_set = Subset(cityscapes_dataset, range(len(cityscapes_dataset)))

    else:
        print("error - zly argument dataset")
        exit(0)

    tr = range(len(train_set))
    te = range(len(test_set))

    train_set = torch.utils.data.Subset(train_set, tr)
    test_set = torch.utils.data.Subset(test_set, te)

    # get data loaders
    train_loader = utils.data.DataLoader(train_set, 64, shuffle=True)
    test_loader = utils.data.DataLoader(test_set, 1000, shuffle=False)

    return train_loader, test_loader
