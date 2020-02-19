import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.data_preprocess import crop_resize_data, encode_label


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        label = label.astype(np.long)
        return {'image': torch.from_numpy(image.copy()),
                'label': torch.from_numpy(label.copy())}


class DataSet(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        :param csv_file: a csv_file include image path and label path
        :param transform: data augment operation, numpy to torch tensor operation
        """
        super(DataSet, self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(), 'utils', 'data_list', csv_file),
                                header=None, names=["image", "gray_label", "color_label"])
        self.images = self.data["image"].values[1:]
        self.labels = self.data["gray_label"].values[1:]
        self.transfrom = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        label = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        image, label = crop_resize_data(image, label)
        label = encode_label(label)
        sample = (image, label)
        if self.transfrom:
            sample = self.transfrom(sample)
        return sample
