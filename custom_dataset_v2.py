import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
from PIL import Image
import pathlib

class StanfordCars(Dataset):
    def __init__(
            self,
            root_dir,
            transforms=None,
            train=True
        ):

        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train

        if self.train:
            self.annotation = loadmat(pathlib.Path(self.root_dir) / "cars_train_annos.mat", squeeze_me=True)["annotations"]
            self.subdir = "cars_train"
        else:
            self.annotation = loadmat(pathlib.Path(self.root_dir) / "cars_test_annos.mat", squeeze_me=True)["annotations"]
            self.subdir = "cars_test"
    
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.subdir, self.annotation[index]["fname"])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.annotation[index]["class"] - 1)

        if self.transforms:
            image = self.transforms(image)

        return image, label

