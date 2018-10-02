import os
import cv2
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from scipy import misc
from imgaug import augmenters as iaa
import imgaug as ia
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from unet import *


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Scale((256, 256)),
            iaa.Fliplr(0.5),
            # iaa.PiecewiseAffine(scale=(0.0001, 0.0002)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Affine(shear=(-20, 20))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


transforms_via_imgaug = ImgAugTransform()

# _, label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)

class ConvertToBinary():
    def __init__(self, threshold=0, value=255, type = cv2.THRESH_BINARY):
        self.threshold = threshold
        self.value = value
        self.type = type

    def __call__(self, nparray):
        _, nparray[:, :, 1:] = cv2.threshold(nparray[:, :, 1:], self.threshold, self.value, self.type)
        return nparray

convert_to_binary = ConvertToBinary()



data_transform = {
    'train': transforms.Compose([
        transforms_via_imgaug,
        convert_to_binary,
        transforms.ToTensor()
    ]),

    'val': transforms.Compose([
        transforms.ToTensor()
    ])
}



class NeuronDataset(Dataset):
    def __init__(self, data_dir, phase, transform=None):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform


    def __len__(self):
        if self.phase == 'train':
            return 20
        if self.phase == 'val':
            return 10

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, '{}'.format(self.phase),
                                       'image', '{}.png'.format(index + 1)))

        label = Image.open(os.path.join(self.data_dir, '{}'.format(self.phase),
                                       'label', '{}.png'.format(index + 1)))

        # label.show()
        #
        # TF.resize(label, (256, 256), interpolation=2).show()

        image = np.array(TF.resize(image, (256, 256), interpolation=2))
        label = np.array(TF.resize(label, (256, 256), interpolation=2))


        image_white_black = np.stack([image, label, label], axis=-1)
        # print(image_white_black[:, :, 1:].shape)

        if self.transform:
            image_white_black = self.transform[self.phase](image_white_black)
        # else:
        #     image_white_black = transforms.ToTensor()

        image = image_white_black[0].view(-1, *image_white_black[0].size())

        # _, label = cv2.threshold(label, 5, 255, cv2.THRESH_BINARY)
        # label = np.array(label)

        label = image_white_black[1:]
        label[1] = 255 - label[1]

        return image, label


# neuron_dataset = NeuronDataset(DATA_DIR, 'train', data_transform)
#
#
# plt.figure()
# for i in range(1):
#     image_sample, label_sample = neuron_dataset[i]
#     print()
#
#     plt.subplot(1, 3, 1)
#     image = image_sample[0, :, :].numpy()
#     plt.imshow(image, cmap='gray')
#
#     plt.subplot(1, 3, 2)
#     white = label_sample[0, :, :].numpy()
#     plt.imshow(white, cmap='gray')
#
#     plt.subplot(1, 3, 3)
#     black = label_sample[1, :, :].numpy()
#     plt.imshow(black, cmap='gray')
#
#     plt.show()

# neuron_dataset = {phase: NeuronDataset(DATA_DIR, phase, data_transform)}

DATA_DIR = './data'
















