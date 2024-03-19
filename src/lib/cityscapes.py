from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.models import resnet18

class CityscapesDataset(Dataset):
    def __init__(self, depth_dir: str, data_dir: str, resnet_weights: dict = resnet18(pretrained=False).state_dict()):
        self.depth_dir = depth_dir
        self.data_dir = data_dir
        self.resnet_weights = resnet_weights
        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
        ])

    def __len__(self):
        total_size = sum(len(files) for _, _, files in os.walk(self.data_dir))
        return total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        subfolder = None
        for folder in os.listdir(self.data_dir):
            dir_size = len(os.listdir(os.path.join(self.data_dir, folder)))
            if idx < dir_size:
                subfolder = folder
                break
            idx -= dir_size

        image_path = os.path.join(self.data_dir, subfolder, os.listdir(os.path.join(self.data_dir, subfolder))[idx])
        depth_path = os.path.join(self.depth_dir, subfolder, os.listdir(os.path.join(self.depth_dir, subfolder))[idx])

        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)

        image = self.preprocess(image)

        depth_transforms = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
        ])
        depth = depth_transforms(depth)

        return image.float(), depth.float()

