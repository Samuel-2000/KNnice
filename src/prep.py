import torch
import numpy as np
from tqdm import tqdm
from torch import optim as optim
from lib import data_loader, nets, paths


def get_network(args, device):
    train_loader, test_loader, val_loader, first_test_case = data_loader.data_load(args)
    model = retrieve_trained_net(device, train_loader, val_loader, test_loader, args)

    return model, first_test_case


def retrieve_trained_net(device, train_loader, val_loader, test_loader, args):
    # Create instance of neural network
    model_classes = [nets.DepthModel]
    try:
        model = model_classes[args.NNmodel]().to(device)
    except IndexError:
        print("Error: model not specified")
        exit(0)

    
    if paths.final_state.exists():
        print("using saved final state for training")
        model.load_state_dict(torch.load(paths.final_state, map_location=torch.device(device)))
        torch.save(model.state_dict(), paths.model_init_state) # return point if training goes bad.
    elif paths.model_init_state.exists():
        model.load_state_dict(torch.load(paths.model_init_state, map_location=torch.device(device)))


    if args.train:
        optimizer_class = optim.Adam if args.adam else optim.SGD
        optimizer = optimizer_class(model.parameters(), lr=0.01)  # set optimizer

        train_loss_file = open(paths.train_loss_path, "a+")
        test_val_loss_list = []
        min_vall_loss = None
        for epoch in tqdm(range(0, args.epochs), desc="Model training"):
            nets.train(model, train_loader, optimizer, device, train_loss_file)
            val_loss = nets.validate(model, val_loader, device)
            test_val_loss_list.append(val_loss)
            torch.save(model.state_dict(), paths.final_state)  # save parameters of model
            if val_loss is None or val_loss < min_vall_loss:
                min_vall_loss = val_loss
            else:
                print(f"premature break because validation loss got higher (overtraining) after epoch: {epoch}")
                break

        train_loss_file.close()
        
        test_val_loss_list.append(nets.test(model, test_loader, device))
        np.savetxt(paths.test_validation_loss_path, test_val_loss_list, fmt='%1.5f')

    return model  # return neural network in final (trained) state

