from torch.utils.data import Dataset, DataLoader
from typing import Callable
from scipy.io import loadmat
from PIL import Image
import os

class CarsDataset(Dataset):
    def __init__(self, imgs_dir: str, annos_path: str, transform: Callable 
= None):        
        self.imgs_paths = self.__load_imgs_paths(imgs_dir)
        self.labels = self.__load_labels(annos_path)
        self.transform = transform
    
    def __load_imgs_paths(self, dir):
        return [os.path.join(dir, filename) for filename in 
os.listdir(dir) if os.path.isfile(os.path.join(dir, filename))]
    
    def __load_labels(self, path):
        mat = loadmat(path)
        labels = {}

        for arr in mat['annotations'][0]:
            filename, label = str(arr[5][0]), int(arr[4][0,0])-1
            labels[filename] = label

        return labels
    
    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, index):
        img_path = self.imgs_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[os.path.basename(img_path)]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
