import os

import albumentations
import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class REAF_Custom_Dataset(Dataset):

    def __init__(self, img_path, mask_path, transform, csv_path):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        img_name = patience_info['name']
        img_file = os.path.join(self.img_path, img_name)
        mask_file = os.path.join(self.mask_path, img_name)
        label = patience_info['label']
        img = cv2.imread(img_file, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            T = self.transform(image=img, mask=mask)
            img, mask = T["image"], T["mask"]
        img = transforms.ToTensor()(Image.fromarray(img).convert("RGB"))
        mask = transforms.ToTensor()(Image.fromarray(mask).convert("L"))
        return {'imgs': img, 'mask': mask, 'labels': label, 'names': img_name}

    def __len__(self):
        return len(self.info)


def get_REAF_dataset(img_path, mask_path, csv_path, img_size, mode='train'):
    p = 0.5
    train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=p),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=p),
        albumentations.Resize(height=img_size, width=img_size, p=1),
    ],
                                             p=1)
    test_transform = albumentations.Compose([
        albumentations.Resize(height=img_size, width=img_size, p=1),
    ])

    if mode == 'train':
        transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = REAF_Custom_Dataset(img_path, mask_path, transform, csv_path)

    return dataset


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix
