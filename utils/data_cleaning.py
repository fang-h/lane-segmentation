import os
import numpy as np
import cv2
import pandas as pd

Road02_path_file = 'Road02.csv'  # not discard
Road03_path_file = 'Road03.csv'  # discard 1000 images(28215 pixel sum)
Road04_path_file = 'Road04.csv'  # disacrd 500 images(24076 pixel sum)


def discard_wrong_label(csv_file):
    image_list = []
    gray_label_list = []
    color_label_list = []
    data = pd.read_csv(csv_file, header=None, names=["image", "gray_label", "color_label"])
    image = data["image"].values[1:]
    gray_label = data["gray_label"].values[1:]
    color_label = data["color_label"].values[1:]
    for i in range(len(image)):
        gray_label_image = cv2.imread(gray_label[i], cv2.IMREAD_GRAYSCALE)
        gray_label_image[np.where(gray_label_image == 255)] = 0
        gray_label_image[np.where(gray_label_image > 0)] = 1
        lane_pixel_sum_per_image = np.sum(gray_label_image)
        if lane_pixel_sum_per_image > 28215: 
            image_list.append(image[i])
            gray_label_list.append((gray_label[i]))
            color_label_list.append(color_label[i])
    all = pd.DataFrame({'image': image_list, 'gray_label': gray_label_list, 'color_label': color_label_list})
    all.to_csv(os.path.join(os.getcwd(), 'data_list', 'Road03_clean' + '.csv'), index=False)


if __name__ == '__main__':
    discard_wrong_label(Road03_path_file)
    











