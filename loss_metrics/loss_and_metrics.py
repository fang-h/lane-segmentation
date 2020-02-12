“”“use CrossEntropy Loss as the cost loss for class every pixel,
   and use miou as the metrics to evaluate the performance of the model”“”



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MySoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, num_class):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.num_class = num_class

    def forward(self, input, target):
		# convert to [N, C]
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, self.num_class)
		# convert to [N]
        target = target.view(-1)
        return nn.CrossEntropyLoss(reduction='mean')(input, target)
		
		
def compute_iou(pred, gt, result):
    pred = pred.numpy()
    gt = gt.numpy()
    for i in range(8):
        single_gt = gt == i
        single_pred = pred == i
        temp_tp = np.sum(single_gt * single_pred)  # compute the intersection
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp  # compute the union
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result


