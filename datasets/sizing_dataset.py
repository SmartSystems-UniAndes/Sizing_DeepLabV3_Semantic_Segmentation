import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class SizingDataset(Dataset):
    def __init__(self, root, image_folder, mask_folder, transform=False, resize=None):
        image_files = self.get_images_files(root, image_folder)
        mask_files = self.get_images_files(root, mask_folder)

        self.data_df = pd.DataFrame({"image_file": image_files, "mask_file": mask_files})
        self.transform = transform
        self.resize = resize
        self.class_names = np.array(['background', 'pv_panel'])
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.original_size = None

    @staticmethod
    def get_images_files(root, image_folder):
        image_folder_path = os.path.join(root, image_folder)
        image_files = sorted(os.listdir(image_folder_path))
        image_files = [os.path.join(image_folder_path, image) for image in image_files]

        return image_files

    def __len__(self):
        return len(self.data_df)

    def transforms(self, image, mask):
        if self.resize:
            image = image.resize(self.resize)
            mask = mask.resize(self.resize)

        image = np.array(image, dtype=np.int8)
        image = image[:, :, ::-1]
        image = image.astype(np.float64)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        mask = np.array(mask, dtype=np.int32)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask[mask > 0] = 1
        mask = torch.from_numpy(mask).long()

        return image, mask

    def un_transforms(self, image, mask):
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        image += self.mean_bgr
        image = image.astype(np.uint8)
        image = image[:, :, ::-1]
        image = Image.fromarray(image)

        if self.resize:
            image = image.resize(self.original_size)

        mask = Image.fromarray((mask * 255).astype(np.uint8))
        if self.resize:
            mask = mask.resize(self.original_size)

        return image, mask

    def __getitem__(self, index):
        image_file = self.data_df.iloc[index]["image_file"]
        image = Image.open(image_file)
        self.original_size = image.size

        mask_file = self.data_df.iloc[index]["mask_file"]
        mask = Image.open(mask_file)

        if self.transform:
            image, mask = self.transforms(image, mask)

        sample = {"image": image, "mask": mask, "tag": os.path.basename(os.path.normpath(image_file))}

        return sample
