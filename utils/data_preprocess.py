"""the original image size(label size) is (1710, 3384, 3), after observe the image, and find the top of the image
   merely have no lane lines, for save train time and GPU resource, a function to get proper h-value to crop the top
   of the image(mask)， and resize it"""


import os
import cv2
import numpy as np
import pandas as pd


def get_h_value(csv_file, ratio):
    """
    :param csv_file: path of all data
    :param ratio:  sum of save lane line / (sum of crop lane lines + sum of save lane line)
    :return:
    """
    label = pd.read_csv(os.path.join(csv_file), header=None, names=["label"])
    label = label["label"].values[1:]
    croped = np.zeros((600, len(label)))
    saved = np.zeros((600, len(label)))
    for i in range(len(label)):
        mask = cv2.imread(label[i], cv2.IMREAD_GRAYSCALE)
        # changed mask to binary
        mask[np.where(mask > 0)] = 1
        # suppose h_value in the range(200, 800)
        var1 = 0
        const_top = np.sum(mask[:200, :])
        const_bottom = np.sum(mask[800:, :])
        const_middle = np.sum(mask[200:800, :])
        for h_value in range(200, 800):
            current = np.sum(mask[h_value, :])
            var1 += current
            var2 = const_middle - var1
            croped[h_value - 200, i] = var1 + const_top
            saved[h_value - 200, i] = var2 + const_bottom
    croped = np.sum(croped, axis=1)
    saved = np.sum(saved, axis=1)
    R = saved / (croped + saved)
    np.save('R.npy', R)  # save it for analysis
    for i in range(len(R) - 1, 0, -1):
        if R[i] >= ratio:
            return i


def crop_resize_data(image, label, h_value=723, size_to_resize=(1536, 448)):

    """"int the train and val dataset, the original size is (1710, 3384, 3), we select h_value = 723 which save
        99.99% lane lines, after crop, image_size is(987, 3384), resize to (768, 224), (1536, 448)"""
    image = image[h_value:, :, :]
    label = label[h_value:, :]
    image = cv2.resize(image, size_to_resize, interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, size_to_resize, interpolation=cv2.INTER_NEAREST)  # not generate new label value
    return image, label


def encode_label(label):
    """see more details in http://apolloscape.auto/lane_segmentation.html"""
    encode_label = np.zeros((label.shape[0], label.shape[1]))
    id_train = {0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202,
                    231, 230, 228, 229, 233, 212, 223],
                1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210], 5: [214],
                6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}
    for i in range(8):
        for item in id_train[i]:
            encode_label[label == item] = i
    return encode_label
