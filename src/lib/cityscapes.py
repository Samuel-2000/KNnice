from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.models import resnet18
from . import paths

import shutil
import requests
import zipfile

def download_cityscapes():
    # URL for the login action
    login_url = 'https://www.cityscapes-dataset.com/login/'

    # URLs for downloading the file after logging in
    urls = [
        'https://www.cityscapes-dataset.com/file-handling/?packageID=3',
        'https://www.cityscapes-dataset.com/file-handling/?packageID=7'
    ]

    # login credentials and any other required form data for POST request
    login_data = {
        'username': 'bahno',
        'password': 'X4W?y2H%h-t445t',
        'submit': 'Login'
    }

    # start a session so that cookies are saved between requests
    with requests.Session() as session:
        # send POST request to the login URL with the login data
        response = session.post(login_url, data=login_data)

        # check if login was successful
        if response.status_code == 200:
            print("Login successful")
        else:
            print("Login failed")
            exit()

        # send GET request to the download URLs
        for url in urls:
            print(f"Downloading file from: {url}")
            response = session.get(url, stream=True)

            # write the content to a file
            content_disposition = response.headers.get('content-disposition')
            filename = content_disposition.split('filename=')[1].replace("\"", "")

            with open(filename, 'wb') as download_file:
                if not os.path.isdir('dataset'):
                    os.mkdir('dataset')

                for chunk in response.iter_content(chunk_size=8192):
                    download_file.write(chunk)

                print(f"File downloaded successfully as {filename}")

                # unzip contents
                with zipfile.ZipFile("dataset/" + filename, 'r') as zip_ref:
                    # check what zip file we are extracting
                    if "disparity" in filename:
                        directory_to_extract_to = paths.depth
                    else:
                        directory_to_extract_to = paths.image

                    # extract the contents
                    zip_ref.extractall(directory_to_extract_to)
                    zip_ref.close()
                download_file.close()

            # remove old zip file, license and readme
            os.remove("dataset/" + filename)
            os.remove(directory_to_extract_to + "license.txt")
            os.remove(directory_to_extract_to + "README")

            # parent folder to be removed
            core_folder = os.path.join(directory_to_extract_to, filename.split("_")[0])

            # move the contents of the parent folder to its original place
            for item in os.listdir(core_folder):
                item_path = os.path.join(core_folder, item)

                # check if the item is a folder
                if os.path.isdir(item_path):
                    # construct the destination path for the current item
                    destination_item_path = os.path.join(directory_to_extract_to, item)
                    # move the subfolder to the destination
                    shutil.move(item_path, destination_item_path)

            # remove the parent folder
            os.rmdir(core_folder)

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

