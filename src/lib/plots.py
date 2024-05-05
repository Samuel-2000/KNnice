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



def plot_depth_activations(model, first_test_case, device, save_path=None):
    image, target = first_test_case
    print("Image shape:", image.shape)
    
    image = image.to(device)
    
    # Forward pass
    output = model(image.unsqueeze(0))  # Add batch dimension

    # Convert tensors to numpy arrays
    image_np = image.cpu().numpy()
    target_np = target.cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 30))

    # Plot image
    if len(image_np.shape) == 2:  # Grayscale image
        axes[0].imshow(image_np, cmap='gray')
    else:  # Color image
        axes[0].imshow(image_np.transpose(1, 2, 0))  # Assuming channels-last format
    axes[0].set_title('Image')

    # Save the image subplot
    if save_path:
        image_save_path = os.path.splitext(save_path)[0] + '_image.png'
        plt.savefig(image_save_path)
        plt.close()
    
    # Plot ground truth depth
    axes[1].imshow(target_np.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Depth')

    # Save the ground truth depth subplot
    if save_path:
        target_save_path = os.path.splitext(save_path)[0] + '_ground_truth.png'
        plt.savefig(target_save_path)
        plt.close()

    # Plot network's depth prediction
    axes[2].imshow(output_np.squeeze(), cmap='gray')
    axes[2].set_title('Network Depth Prediction')

    # Save the network depth prediction subplot
    if save_path:
        output_save_path = os.path.splitext(save_path)[0] + '_network_prediction.png'
        plt.savefig(output_save_path)
        plt.close()

    # Show the figure
    plt.tight_layout()
    plt.show()

    # Save the original figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()