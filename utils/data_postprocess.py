import numpy as np


def decode_color_label(labels):
    """see more details in http://apolloscape.auto/lane_segmentation.html"""
    if labels.dim() != 2:
        labels = np.reshape(labels, (labels.shape[1], labels.shape[2]))
    decode_label = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')

    decode_label[0][labels == 0] = 0
    decode_label[1][labels == 0] = 0
    decode_label[2][labels == 0] = 0
    decode_label[0][labels == 1] = 70
    decode_label[1][labels == 1] = 130
    decode_label[2][labels == 1] = 180
    decode_label[0][labels == 2] = 0
    decode_label[1][labels == 2] = 0
    decode_label[2][labels == 2] = 142
    decode_label[0][labels == 3] = 220
    decode_label[1][labels == 3] = 220
    decode_label[2][labels == 3] = 0
    decode_label[0][labels == 4] = 128
    decode_label[1][labels == 4] = 64
    decode_label[2][labels == 4] = 128
    decode_label[0][labels == 5] = 190
    decode_label[1][labels == 5] = 153
    decode_label[2][labels == 5] = 153
    decode_label[0][labels == 6] = 128
    decode_label[1][labels == 6] = 128
    decode_label[2][labels == 6] = 0
    decode_label[0][labels == 7] = 255
    decode_label[1][labels == 7] = 128
    decode_label[2][labels == 7] = 0
    decode_label = np.transpose(decode_label, [1, 2, 0])
    return decode_label