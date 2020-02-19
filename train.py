import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


from model.DeeplabV3p import DeeplabV3p
from model.HRNetV2 import SemanticHighResolutionNet
from model.HRNetV2_config import HRNetV2Config
from utils.data_load import DataSet, ToTensor
from utils.data_augment import HorizonFlip, PixelAug, ScaleAug, CutOut
from loss_metrics.loss_and_metrics import Loss, compute_iou
from config import Config


GPU = [1]


def adjust_lr_for_low_resolution(optimizer, epoch):
    """"""
    if epoch < 2:
        lr = 1e-4
    elif epoch < 5:
        lr = 3e-4
    elif epoch < 10:
        lr = 2e-4
    elif epoch < 15:
        lr = 1e-4
    elif epoch < 20:
        lr = 5e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_for_high_resolution(optimizer, epoch):
    """"""
    if epoch < 1:
        lr = 1e-4
    elif epoch < 3:
        lr = 5e-4
    elif epoch < 6:
        lr = 4e-4
    elif epoch < 9:
        lr = 3e-4
    elif epoch < 12:
        lr = 2e-4
    elif epoch < 15:
        lr = 1e-4
    elif epoch < 18:
        lr = 9e-5
    elif epoch < 21:
        lr = 8e-5
    elif epoch < 24:
        lr = 7e-5
    elif epoch < 27:
        lr = 5e-5
    elif epoch < 30:
        lr = 1e-5
    elif epoch < 33:
        lr = 5e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(model, data_loader, optimizer, config, train_file, epoch):
    """

    :param model: DeepLabV3+ or HRNetV2
    :param data_loader: a instance of torch.utils.data.DataLoader, (image, label)
    :param optimizer: adam or ...
    :param config: some settings
    :param train_file: record the train logs such as loss
    :param epoch:
    :return:
    """
    model.train()
    total_loss = 0
    data_process = tqdm(data_loader)  # tqdm is a tool to display information when train or test
    for batch_data in data_process:
        image, label = batch_data['image'], batch_data['label']
        if torch.cuda.is_available():
            image, label = image.cuda(device=GPU[0]), label.cuda(device=GPU[0])
        optimizer.zero_grad()
        out = model(image)
        loss = Loss(config.NUM_CLASSES)(out, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        data_process.set_description_str("epoch:{}".format(epoch))
        data_process.set_postfix_str("loss:{:.6f}".format(loss.item()))
    train_file.write("Epoch:{}, loss is {:.6f} \n".format(epoch, total_loss / len(data_loader)))
    train_file.flush()


def val_epoch(model, data_loader, config, val_file, epoch):
    model.eval()
    total_loss = 0
    result = {"intersection": {i: 0 for i in range(8)}, "union": {i: 0 for i in range(8)}}
    data_process = tqdm(data_loader)
    for batch_data in data_process:
        image, label = batch_data['image'], batch_data['label']
        if torch.cuda.is_available():
            image, label = image.cuda(device=GPU[0]), label.cuda(device=GPU[0])
        out = model(image)
        loss = Loss(config.NUM_CLASSES)(out, label)
        total_loss += loss.item()
        prediction = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(prediction.cpu(), label.cpu(), result)
        data_process.set_description_str("epoch: {}".format(epoch))
        data_process.set_postfix_str("loss:{:.6f}".format(loss.item()))
    val_file.write("Epoch:{}".format(epoch))
    for i in range(8):
        result_string = "{}: {:.6f} \n".format(i, result["intersection"][i] / result["union"][i])
        val_file.write(result_string)
    val_file.write("Epoch:{}, loss is {:.6f} \n".format(epoch, total_loss / len(data_loader)))
    val_file.flush()


def train(model_name, model_path, train_csv_file, val_csv_file, start_epoch):
    """
    :param model_name: Xception_DeeplabV3p or MobileNetV2_DeeplabV3p or HRNetV2
    :param model_path: trained model path, if not none, load it
    :param train_csv_file: image path and label path of train
    :param val_csv_file: image path and label path of validation
    :param start_epoch:
    :return:
    """
    cfg = Config()
    if model_name == "Xception_DeeplabV3p":
        save_path = cfg.Xception_SAVE_PATH
        model = DeeplabV3p(cfg.NUM_CLASSES, backbone='Xception')
    if model_name == 'MobileNetV2_DeeplabV3p':
        save_path = cfg.MobileNetV2_SAVE_PATH
        model = DeeplabV3p(cfg.NUM_CLASSES, backbone='MobileNetV2')
    if model_name == 'HRNetV2':
        save_path = cfg.HRNetV2_SAVE_PATH
        model = SemanticHighResolutionNet(HRNetV2Config().config)
    os.makedirs(save_path, exist_ok=True)

    if torch.cuda.is_available():
        model = model.cuda(device=GPU[0])
        model = torch.nn.DataParallel(model, device_ids=GPU)

    if model_path is not None:
        model_dict = torch.load(os.path.join(save_path, model_path))
        model.load_state_dict(model_dict)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = DataSet(train_csv_file,
                            transform=transforms.Compose([HorizonFlip(0.7), PixelAug(0.2), ScaleAug(0.3, [0.7, 1.3]),
                                                          CutOut(0.2, 64), ToTensor()]))

    train_dataloader = DataLoader(train_dataset, batch_size=2*len(GPU), shuffle=True, drop_last=True, **kwargs)

    val_dataset = DataSet(val_csv_file, transform=transforms.Compose([ToTensor()]))
    val_dataloader = DataLoader(val_dataset, batch_size=1*len(GPU), shuffle=False, drop_last=False, **kwargs)

    train_file = open(os.path.join(save_path, 'train_high_logs.csv'), 'w')
    val_file = open(os.path.join(save_path, 'val_high_logs.csv'), 'w')

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(start_epoch, cfg.EPOCHS):
        adjust_lr_for_high_resolution(optimizer, epoch)
        train_epoch(model, train_dataloader, optimizer, cfg, train_file, epoch)
        if epoch % 3 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_high_{}.pth'.format(epoch)))
            # torch.save(model, os.path.join(save_path, 'model_x{}.pth'.format(epoch)))
        val_epoch(model, val_dataloader, cfg, val_file, epoch)


if __name__ == '__main__':
    train()











