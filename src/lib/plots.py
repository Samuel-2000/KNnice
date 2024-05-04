"""
Plot library for visualizing the neural network training progress

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from . import paths
import torch

color_trained = "dimgrey"

font = FontProperties()
font.set_size(20)


def plot_depth_activations(model, test_loader, device, save_path):
    inputs, targets = next(iter(test_loader))
    print(len(inputs))
    print(len(inputs[0]))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Forward pass
    outputs = model(inputs)
    
    # Convert tensors to numpy arrays
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    # Plot input image
    axes[0].imshow(inputs_np[0].transpose(1, 2, 0))  # Assuming channels-last format
    axes[0].set_title('Input Image')

    # Plot ground truth depth
    axes[1].imshow(targets_np[0].squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Depth')

    # Plot network's depth prediction
    axes[2].imshow(outputs_np[0].squeeze(), cmap='gray')
    axes[2].set_title('Network Depth Prediction')

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
