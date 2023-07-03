import os

import cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


class DataProcessing_image_lesionmap(Dataset):

    def __init__(self, data_path, lesionmap_path, img_filename, transform):
        self.img_path = data_path
        self.lesionmap_path = lesionmap_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        for line in fp.readlines():
            filename, label = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
        fp.close()

    def __getitem__(self, index):
        name = self.img_filename[index]

        img = cv2.imread(os.path.join(self.img_path, name), cv2.COLOR_BGR2RGB)
        lesionmap = cv2.imread(os.path.join(self.lesionmap_path, name), cv2.COLOR_BGR2GRAY)

        T = self.transform(image=img, mask=lesionmap)
        img, lesionmap = T["image"], T["mask"]

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        lesionmap = ToTensor()(Image.fromarray(lesionmap).convert("L"))

        name = self.img_filename[index]
        severity = self.labels[index]

        return img, lesionmap, severity

    def __len__(self):
        return len(self.img_filename)


class DataProcessing_image_lesionmap_predict(Dataset):

    def __init__(self, data_path, lesionmap_path, img_filename, transform):
        self.img_path = data_path
        self.lesionmap_path = lesionmap_path
        self.transform = transform
        # reading img file from file
        fp = open(img_filename, 'r')

        self.img_filename = []
        self.labels = []
        self.lesion_numers = []
        for line in fp.readlines():
            filename, label = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
        fp.close()

    def __getitem__(self, index):
        name = self.img_filename[index]

        img = cv2.imread(os.path.join(self.img_path, name), cv2.COLOR_BGR2RGB)
        lesionmap = cv2.imread(os.path.join(self.lesionmap_path, name), cv2.COLOR_BGR2GRAY)

        T = self.transform(image=img, mask=lesionmap)
        img, lesionmap = T["image"], T["mask"]

        img = ToTensor()(Image.fromarray(img).convert("RGB"))
        lesionmap = ToTensor()(Image.fromarray(lesionmap).convert("L"))

        name = self.img_filename[index]
        severity = self.labels[index]

        return img, lesionmap, severity, name

    def __len__(self):
        return len(self.img_filename)
