import numpy as np
from PIL import Image
from PIL import ImageFile   #ImageFile contains support for PIL to open and save images
ImageFile.LOAD_TRUNCATED_IMAGES=True  #If image is truncated(or corrupt) then also load it..
import torch

class Cloader:
    def __init__(self,image_path,targets,resize=None,transforms=None):
        self.image_path=image_path
        self.targets=targets
        self.resize=resize
        self.transforms=transforms
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_path[idx])
        targets = self.targets[idx]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

