import numpy as np
import cv2 
import random

from skimage.io import imread
from skimage.util import random_noise

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        
        return len(self.img_paths)


    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path,allow_pickle=True)
        npmask = np.load(mask_path,allow_pickle=True)

        npimage = npimage.transpose((2, 0, 1))
        nplabel = np.empty((npimage.shape[1],npimage.shape[2], 1))
        nplabel[:, :, 0] = npmask
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage.astype("float32"),nplabel
