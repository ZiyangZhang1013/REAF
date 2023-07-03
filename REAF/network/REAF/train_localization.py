import os

import numpy as np
import torch
from segmentation_models_pytorch.utils.metrics import IoU
from torch.utils.data import DataLoader

from dataset.dataset_processing import DataProcessing_image_lesionmap
from model.model import UnetPlusPlus
from utils.utils import Logger, data_augmentadion, dice_coef


class Config(object):
    cross_validation_index = ["0"]

    def __init__(self, cross_val_index=cross_validation_index[0], time="", model_name="UNet++"):
        self.model_name = model_name
        self.gpu_id = "0"
        self.image_path = '/root/autodl-tmp/paper/REAF_BUSI/img'
        self.mask_path = "/root/autodl-tmp/paper/REAF_BUSI/mask"
        self.train_mapping_path = '/root/autodl-tmp/paper/REAF_BUSI/train.txt'
        self.test_mapping_path = '/root/autodl-tmp/paper/REAF_BUSI/test.txt'
        self.model_save_path = "./results/model_state_dict/localization/%s_%s%s.pkl" % (
            self.model_name, cross_val_index, time)
        self.log_path = "./results/logs/localization/%s_%s%s.log" % (self.model_name,
                                                                     cross_val_index, time)

        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
        if not os.path.exists(os.path.dirname(self.model_save_path)):
            os.makedirs(os.path.dirname(self.model_save_path))

        self.network_input_size = (224, 224)
        self.batch_size = 32
        self.ecpoch = 200
        self.learning_rate = 1e-4

        self.num_workers = 12


def train(time=1, model_name="UNet++"):

    config = Config(time=time, model_name=model_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    model = torch.nn.Sequential()
    if config.model_name == 'UNet++':
        config.batch_size = 32
        model = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    model = model.cuda()

    dset_train = DataProcessing_image_lesionmap(config.image_path,
                                                config.mask_path,
                                                config.train_mapping_path,
                                                transform=data_augmentadion(config))

    dset_test = DataProcessing_image_lesionmap(config.image_path,
                                               config.mask_path,
                                               config.test_mapping_path,
                                               transform=data_augmentadion(config, train=False))

    train_loader = DataLoader(dset_train,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers,
                              drop_last=False)

    test_loader = DataLoader(dset_test,
                             batch_size=config.batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=False)

    criterion_localization = torch.nn.BCEWithLogitsLoss()
    iou_metric = IoU(threshold=0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    max_result = 0

    log = Logger()
    log.open(config.log_path, mode="a")

    best_msg = "\nBest until now:"
    for epoch in range(config.ecpoch):
        msg = ""

        model.train()

        lesionmap_pre = []
        lesionmap_gt = []
        lesionmap_losses = []
        for step, (img, lesionmap, severity) in enumerate(train_loader):
            img = img.cuda()
            lesionmap = lesionmap.cuda()

            # train
            pre_localization = model(img)

            loss_localization = criterion_localization(pre_localization, lesionmap)

            optimizer.zero_grad()  # clear gradients for this training step
            loss_localization.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lesionmap_pre += pre_localization.detach().cpu().tolist()
            lesionmap_gt += lesionmap.detach().cpu().tolist()
            lesionmap_losses.append(loss_localization.item())

        loss_localizationmap_avg = np.mean(lesionmap_losses)

        iou = iou_metric(torch.Tensor(lesionmap_pre), torch.Tensor(lesionmap_gt))
        dice = dice_coef(torch.Tensor(lesionmap_pre), torch.Tensor(lesionmap_gt))

        msg += "\ntrain:%s\n" % epoch
        msg += "    loss_localizationmap:%.4f" % loss_localizationmap_avg
        msg += "    iou:%.4f" % iou
        msg += "    dice:%.4f" % dice

        # visualize = []
        model.eval()

        with torch.no_grad():
            lesionmap_pre = []
            lesionmap_gt = []
            lesionmap_losses = []
            for step, (img, lesionmap, severity) in enumerate(test_loader):
                img = img.cuda()
                lesionmap = lesionmap.cuda()
                severity = severity.cuda()

                # train
                pre_localization = model(img)
                loss_localization = criterion_localization(pre_localization, lesionmap)

                lesionmap_pre += pre_localization.detach().cpu().tolist()
                lesionmap_gt += lesionmap.detach().cpu().tolist()
                lesionmap_losses.append(loss_localization.item())

                # visualize.append(lesionmap)
                # visualize.append(pre_localization)

            loss_localizationmap_avg = np.mean(lesionmap_losses)
            iou = iou_metric(torch.Tensor(lesionmap_pre), torch.Tensor(lesionmap_gt))
            dice = dice_coef(torch.Tensor(lesionmap_pre), torch.Tensor(lesionmap_gt))

            msg += "\ntest:%s\n" % epoch
            msg += "    loss_localizationmap:%.4f" % loss_localizationmap_avg
            msg += "    iou:%.4f" % iou
            msg += "    dice:%.4f" % dice

        result = (dice + iou) / 2
        if max_result < result:
            best_msg = "\nBest until now:" + msg
            max_result = result
            torch.save(model.state_dict(), config.model_save_path)

        log.write(msg)
        log.write(best_msg)


train(time=0)
