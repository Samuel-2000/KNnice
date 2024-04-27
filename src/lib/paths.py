"""
creates folders, and simplifies working with file paths.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
from pathlib import Path
from . import arg_parse

args = arg_parse.parse_arguments()
model_arch = args.NNmodel
dataset_name = args.dataset
adam_true = args.adam
dropout = args.dropout

# general directories
proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dataset = Path(proj_dir, "datasets")

depth = Path(dataset, "depth/train")
image = Path(dataset, "image/train")

models_dir = Path(proj_dir, "models")

models_init_dir = Path(models_dir, "__models_initializations")
all_steps_dir = Path(models_dir, "_all_steps")


# current model directories
model_names = ["depthModel"] #"ModifiedLeNet", 
dataset_names = ["cityScapes"]

name = model_names[model_arch]
model_init_state = Path(models_init_dir, f"{name}_{dataset_names[dataset_name]}_init_state.pt")  # init_state
name += "_Adam" if adam_true else "_SGD"
name += "_dropout" if dropout == 1 else ""
name += dataset_names[dataset_name]

current_model_dir = Path(models_dir, name)

results = Path(current_model_dir, "results")
imgs = Path(current_model_dir, "images")

# model states
state = Path(current_model_dir, "state_")
final_state = Path(current_model_dir, "final_state.pt")

# directories for loss and accuracy during training
loss_acc = Path(results, "Training_results")

train_loss_path = Path(loss_acc, "train_loss_path.txt")
experiment_train_loss_path = Path(loss_acc, "step_size_experiment_train_loss_path.txt")
validation_loss_path = Path(loss_acc, "validation_loss.txt")
validation_acc_path = Path(loss_acc, "validation_accuracy.txt")

def init_dirs():
    """
    Function initializes directories
    """
    dirs = [results, imgs, loss_acc, models_init_dir]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
