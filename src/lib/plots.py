"""
Plot library for visualizing the neural network training progress

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

color_trained = "dimgrey"

font = FontProperties()
font.set_size(20)



def plot_depth_activations(model, first_test_case, device, path):
    image, target = first_test_case

    image = image.to(device)

    # Forward pass,  Add batch dimension
    output = model(image.unsqueeze(0))

    # Convert tensors to numpy arrays
    image_np = image.cpu().numpy()
    target_np = target.cpu().numpy()
    output_np = output.detach().cpu().numpy()

    image_save_path =  Path(path, '_image.png')
    target_save_path = Path(path, '_ground_truth.png')
    output_save_path = Path(path, '_network_prediction.png')

    # Plot and save image subplot
    fig, axes = plt.subplots()
    axes.imshow(image_np.transpose(1, 2, 0))  # Assuming channels-last format
    axes.set_title('Image')
    plt.savefig(image_save_path)
    plt.close()

    # Plot and save ground truth depth subplot
    fig, axes = plt.subplots()
    axes.imshow(target_np.squeeze(), cmap='gray')
    axes.set_title('Ground Truth Depth')
    plt.savefig(target_save_path)
    plt.close()

    # Plot and save network's depth prediction subplot
    fig, axes = plt.subplots()
    axes.imshow(output_np.squeeze(), cmap='gray')
    axes.set_title('Network Depth Prediction')
    plt.savefig(output_save_path)
    plt.close()