import torch
import numpy as np
from tqdm import tqdm
from torch import optim as optim
from lib import data_loader, nets, paths


def train_net(args, device):
    """
    Prepares NN
    """
    train_loader, test_loader, val_loader = data_loader.data_load(args)
    model = get_net(device, train_loader, val_loader, test_loader, args)

    return model, test_loader


def get_net(device, train_loader, val_loader, test_loader, args):
    """
    Function prepares a neural network model for experiments

    :param device: device to use
    :param train_loader: training dataset loader
    :param test_loader: test dataset loader
    :param args: arguments
    :return: NN model
    """

    # Create instance of neural network
    model_classes = [nets.DepthModel] #nets.ModifiedLeNet
    try:
        model = model_classes[args.NNmodel]().to(device)
    except IndexError:
        print("Error: model not specified")
        exit(0)

    
    if paths.final_state.exists():
        print("using saved final state for training")
        model.load_state_dict(torch.load(paths.final_state, map_location=torch.device(device)))
        
    torch.save(model.state_dict(), paths.model_init_state) # return point if training goes bad.

    optimizer_class = optim.Adam if args.adam else optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=0.01)  # set optimizer

    train_loss_file = open(paths.train_loss_path, "a+")
    test_val_loss_list = []
    for epoch in tqdm(range(0, args.epochs), desc="Model training"):
        nets.train(model, train_loader, optimizer, device, train_loss_file)
        test_val_loss_list.append(nets.validate(model, val_loader, device))

    train_loss_file.close()
    torch.save(model.state_dict(), paths.final_state)  # save final parameters of model
    
    test_val_loss_list.append(nets.test(model, test_loader, device))
    np.savetxt(paths.test_validation_loss_path, test_val_loss_list, fmt='%1.5f')

    return model  # return neural network in final (trained) state

