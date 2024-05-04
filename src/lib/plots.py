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
    image, target = next(iter(test_loader))[0]
    print(len(image))
    image = image.to(device)
    target = target.to(device)
    
    # Forward pass
    output = model(image)
    
    # Convert tensors to numpy arrays
    image_np = image.cpu().numpy()
    target_np = target.cpu().numpy()
    output_np = output.cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    # Plot image
    axes[0].imshow(image_np[0].transpose(1, 2, 0))  # Assuming channels-last format
    axes[0].set_title('image Image')

    # Plot ground truth depth
    axes[1].imshow(target_np[0].squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Depth')

    # Plot network's depth prediction
    axes[2].imshow(output_np[0].squeeze(), cmap='gray')
    axes[2].set_title('Network Depth Prediction')

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
