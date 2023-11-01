import os
import torch
from torch.utils.data import DataLoader, Dataset
import random
import copy

import numpy as np
import cv2
import glob

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs import image_size, data_dir, visualize, augment



# augment the input
alb_transform = A.Compose([
    A.Resize(image_size, image_size), 
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
if augment:
    alb_transform = A.Compose([
        A.Resize(image_size, image_size), 
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ToTensorV2()
    ])

# define main dataset class 
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.img_files = glob.glob(os.path.join(data_dir,'img','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(data_dir,'mask',os.path.basename(img_path)))
        self.transform = transform

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            image = cv2.imread(img_path).astype('float32')
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype('int64')
            
            # By default OpenCV uses BGR color space for color images,
            # so we need to convert the image to RGB color space.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a binary mask using boolean indexing
            # the mask has multiple classes for instance segmentation task
            # for this semantic segmentation problem we need to convert all
            # non-zero classes to 1 to obtain a binary mask
            mask = (mask > 0).astype(np.int64)

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            return image, mask.long()

    def __len__(self):
        return len(self.img_files)


# Define a dataset object
dataset = SegmentationDataset(data_dir=data_dir, transform=alb_transform)


# Draw a sample from dataset:
def visualize_augmentations(dataset, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=2, ncols=samples, figsize=(18, 7))
    for i in range(samples):
        idx = random.randint(0, len(dataset))
        image, mask = dataset[idx]
        ax[0, i].imshow(image.astype('int'))
        ax[1, i].imshow(mask, interpolation="nearest")
        ax[0, i].set_title(f"Augmented image {idx}")
        ax[1, i].set_title(f"Augmented mask {idx}")
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()
        idx += 1
    plt.tight_layout()
    plt.show()

if visualize:
    visualize_augmentations(dataset, samples=5)