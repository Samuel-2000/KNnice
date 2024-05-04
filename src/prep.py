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
    model = get_net(device, train_loader, test_loader, args)

    return train_loader, test_loader, model


def get_net(device, train_loader, test_loader, args):
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

    loss_list = []

    # Save initial state of network if not saved yet
    if not paths.model_init_state.exists():
        torch.save(model.state_dict(), paths.model_init_state)
        

    # Initialize neural network model
    model.load_state_dict(torch.load(paths.model_init_state, map_location=torch.device(device)))

    optimizer_class = optim.Adam if args.adam else optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=0.01)  # set optimizer

    train_loss_file = open(paths.train_loss_path, "a+")
    final_state = paths.final_state.exists()

    if final_state:
        model.load_state_dict(torch.load(paths.final_state, map_location=torch.device(device)))

    # Train model if not trained yet
#if not final_state:
    loss = nets.test(model, test_loader, device)
    loss_list.append(loss)

    for epoch in tqdm(range(1, args.epochs), desc="Model training"):
        nets.train(model, train_loader, optimizer, device, train_loss_file)
        loss = nets.test(model, test_loader, device)
        loss_list.append(loss)
        torch.save(model.state_dict(), str(paths.state) + f"{epoch}.pt")  # save final parameters of model

    torch.save(model.state_dict(), paths.final_state)  # save final parameters of model

    np.savetxt(paths.validation_loss_path, loss_list, fmt='%1.5f')
    

    train_loss_file.close()

    return model  # return neural network in final (trained) state

