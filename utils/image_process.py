import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from utils.process_label import encode_label, decode_color_label, decode_label


class LaneDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(), "data_list", csv_file), header=None, names=["image", "label"])
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]
        self.transfrom = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        label = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
        train_img, train_mask = crop_resize_data(image, label)
        train_mask = encode_label(train_mask)
        sample = [train_img.copy(), train_mask.copy()]
        if self.transfrom:
            sample = self.transfrom(sample)
        return sample


class PixelAug(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0, 1) < 0.15:
            seq = iaa.Sequential([iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                                             iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2)),
                                             iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return image, mask


# class DeformAug(object):
#     def __call__(self, sample):
#         image, mask = sample
#         seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
#         seg_to = seq.to_deterministic()
#         image = seg_to.augment_image(image)
#         mask = seg_to.augment_image(mask)
#         return image, mask


class ScaleAug(object):
    def __call__(self, sample):
        image, mask = sample
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()
        if np.random.uniform(0, 1) < 0.15:
            scale = random.uniform(0.8, 1.2)
            aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)))
            aug_mask = cv2.resize(aug_mask, (int(scale * w), int(scale * h)))
            if scale < 1.0:
                new_h, new_w, _ = aug_image.shape
                pad_h, pad_w = int((h - new_h) / 2), int((w - new_w) / 2)
                pad = [[pad_h, h - new_h - pad_h], [pad_w, w - new_w - pad_w], [0, 0]]
                aug_image = np.pad(aug_image, pad, mode='constant')
                aug_mask = np.pad(aug_mask, pad[:2], mode='constant')
            if scale > 1:
                new_h, new_w, _ = aug_image.shape
                crop_h, crop_w = int((new_h - h) / 2), int((new_w - w) / 2)
                aug_image = aug_image[crop_h:crop_h + h, crop_w:crop_w + w, :]
                aug_mask = aug_mask[crop_h:crop_h + h, crop_w:crop_w + w]
        return aug_image, aug_mask


class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0
        h, w, _ = image.shape
        center_x_min, center_x_max = mask_size_half, w + offset - mask_size_half
        center_y_min, center_y_max = mask_size_half, h + offset - mask_size_half
        center_x, center_y = np.random.randint(center_x_min, center_x_max), np.random.randint(center_y_min, center_y_max)
        x_min, y_min = center_x - mask_size_half, center_y - mask_size_half
        x_max, y_max = x_min + self.mask_size, y_min + self.mask_size
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
        if np.random.uniform(0, 1) < self.p:
            image[y_min:y_max, x_min:x_max] = (0, 0, 0)
        return image, mask


class ToTensor(object):
    def __call__(self, sample):
        image, mask = sample
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        mask = mask.astype(np.long)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}


