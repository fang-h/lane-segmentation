import numpy as np
from imgaug import augmenters as iaa


class PixelAug(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, label):
        if np.random.uniform(0, 1) < p:
            seq = iaa.Sequential([iaa.OneOf([iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                                             iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2)),
                                             iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return image, label


class ScaleAug(object):
    def __init__(self, p, scale_range):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image, label):
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_label = label.copy()
        if np.random.uniform(0, 1) < self.p:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)))
            aug_label = cv2.resize(aug_label, (int(scale * w), int(scale * h)))
            if scale < 1.0:
                new_h, new_w, _ = aug_image.shape
                pad_h, pad_w = int((h - new_h) / 2), int((w - new_w) / 2)
                pad = [[pad_h, h - new_h - pad_h], [pad_w, w - new_w - pad_w], [0, 0]]
                aug_image = np.pad(aug_image, pad, mode='constant')
                aug_label = np.pad(aug_label, pad[:2], mode='constant')
            if scale > 1:
                new_h, new_w, _ = aug_image.shape
                crop_h, crop_w = int((new_h - h) / 2), int((new_w - w) / 2)
                aug_image = aug_image[crop_h:crop_h + h, crop_w:crop_w + w, :]
                aug_label = aug_label[crop_h:crop_h + h, crop_w:crop_w + w]
        return aug_image, aug_label


class CutOut(object):
    def __init__(self, p, cut_size):
        self.p = p
        self.cut_size = cut_size

    def __call__(self, image, label):
        cut_size_half = self.cut_size // 2
        offset = 1 if self.cut_size % 2 == 0 else 0
        h, w, _ = image.shape
        center_x_min, center_x_max = cut_size_half, w + offset - cut_size_half
        center_y_min, center_y_max = cut_size_half, h + offset - cut_size_half
        center_x, center_y = np.random.randint(center_x_min, center_x_max), np.random.randint(center_y_min, center_y_max)
        x_min, y_min = center_x - cut_size_half, center_y - cut_size_half
        x_max, y_max = x_min + self.cut_size, y_min + self.cut_size
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
        if np.random.uniform(0, 1) < self.p:
            image[y_min:y_max, x_min:x_max] = (0, 0, 0)
        return image, label
