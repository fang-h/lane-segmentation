"""Xception_DeeplabV3p, HRNetV2"""

import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from model.DeeplabV3p import DeeplabV3p
from model.HRNetV2 import SemanticHighResolutionNet
from model.HRNetV2_config import HRNetV2Config
from loss_metrics.loss_and_metrics import compute_iou
from utils.data_load import DataSet, ToTensor
from utils.data_postprocess import decode_color_label
from config import Config

GPU = [0, 1]


def ensemble(val_csv, save_image, ensemble_method=None):
    model_name = ['HRNetV2', 'Xception_DeeplabV3p']
    pth_file = ['HRNetV2.pth', 'Xception_DeeplabV3p.pth']
    ensemble_method = ensemble_method
    cfg = Config()
    save_path = cfg.ENSEMBLE_SAVE_PATH
    os.makedirs(save_path, exist_ok=True)
    if save_image is False:
        val_file = open(os.path.join(save_path, ensemble_method + ".csv"), 'w')

    # data prepare
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    val_dataset = DataSet(val_csv, transform=transforms.Compose([ToTensor()]))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False,
                                **kwargs)

    data_process = tqdm(val_dataloader)
    result = {"intersection": {i: 0 for i in range(8)}, "union": {i: 0 for i in range(8)}}
    model_list = []
    for i, name in enumerate(model_name):
        if name == 'Xception_DeeplabV3p':
            model = DeeplabV3p(cfg.NUM_CLASSES, backbone='Xception')
        elif name == 'HRNetV2':
            model = SemanticHighResolutionNet(HRNetV2Config().config)
        pth_file_i = pth_file[i]
        if torch.cuda.is_available():
            model = model.cuda(device=GPU[0])
            model = torch.nn.DataParallel(model, device_ids=GPU)
        model_dict = torch.load(os.path.join(save_path, pth_file_i))
        model.load_state_dict(model_dict)
        model_list.append(model)
    image_name = 1
    for batch_data in data_process:
        image, label = batch_data['image'], batch_data['label']
        if torch.cuda.is_available():
            image, label = image.cuda(device=GPU[0]), label.cuda(device=GPU[0])
        pred_prob = []
        for model in model_list:
            model.eval()
            out_i = model(image)
            pred_prob.append(F.softmax(out_i, dim=1))
        ensemble = torch.cat([x for x in pred_prob], dim=0)
        if ensemble_method == 'mean':
            ensemble = torch.mean(ensemble, dim=0, keepdim=True)
            ensemble = torch.argmax(ensemble, axis=1)
        if ensemble_method == 'max':
            ensemble = torch.max(ensemble, axis=0, keepdims=True)[0]
            ensemble = torch.argmax(ensemble, axis=1)

        if save_image is True:
            mask = decode_color_label(ensemble.cpu())
            cv2.imwrite(os.path.join(cfg.IMAGE_SAVE_PATH, str(image_name) + '.png'), mask)
            image_name += 1
        if save_image is False:
            result = compute_iou(ensemble.cpu(), label.cpu(), result)
        data_process.set_description_str("val")
    if save_image is False:
        for i in range(8):
            result_string = "{}: {:.6f} \n".format(i, result["intersection"][i] / result["union"][i])
            val_file.write(result_string)
        val_file.flush()


if __name__ == '__main__':
    ensemble()



