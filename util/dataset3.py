import cv
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np
import random
from PIL import Image

class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'train/image1/*.jpg'))
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, intex):
        # 根据index读取图像
        image1_path = self.imgs_path[intex]
        image2_path = image1_path.replace('image1', 'image2')
        # image3_path = image1_path.replace('image1', 'image3')

        # 根据image_path生成label_path
        label_path = image1_path.replace('image1', 'label')

        # labelgray_path = image1_path.replace('image1', 'label_gray')
        # 读取训练图像和标签
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        # image3 = cv2.imread(image3_path)

        label = cv2.imread(label_path)
        # 将图像转为单通道图片
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image2= cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(labelgray_path, label)

        # 随机进行数据增强，为2时不处理
        """"""
        flipCote = random.choice([-1, 0, 1])
        # if flipCote != 2:
        #     image1 = self.augment(image1, flipCote)
        #     image2 = self.augment(image2, flipCote)
        #     image3 = self.augment(image3, flipCote)
        #     label = self.augment(label, flipCote)


        label = label.reshape(label.shape[0], label.shape[1], 1)
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        # image3 = self.transform(image3)

        label = self.transform(label)
        # image = torch.cat([image1, image2], dim=0)


        return image1, image2, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    # isbi_dataset = ISBI_Loader(data_path="./data/shuguang",
    #                            transform=Transforms.ToTensor())
    # print("数据个数：", len(isbi_dataset))
    imgs_path = glob.glob(os.path.join("../data/Landsat", 'train/image2/*.bmp'))
    len1 = len(imgs_path)
    print(len1)
    # train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
    #                                            batch_size=4,
    #                                            shuffle=False)
    # i = 0
    # for image1, image2, label in train_loader:
    #     print(i)
    #     i += 1