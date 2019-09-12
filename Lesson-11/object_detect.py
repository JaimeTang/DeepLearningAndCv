import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


class PennFudanDataset(object):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)


    def __len__(self):
        return len(self.imgs)

def main():
    root = 'PennFudanPed'
    masks = sorted(os.listdir(os.path.join(root, 'PedMasks')))
    mask_path = os.path.join(root, 'PedMasks', masks[1])
    mask = Image.open(mask_path)
    plt.imshow(mask)
    plt.show()

    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    masks = mask == obj_ids[:, None, None]
    print(masks)


if __name__=='__main__':
    main()