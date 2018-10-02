import os
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


# for i in range(21, 41):
#     PATH = '../DRIVE/training/images/{}_training.tif'.format(i)
#     image = Image.open(PATH).convert('L')
#     image.save('./data/train/image/{}.png'.format(i - 20))
#
#
#

# for i in range(21, 41):
#     PATH = '../DRIVE/training/1st_manual/{}_manual1.gif'.format(i)
#     image = Image.open(PATH)
#     image.save('./data/train/label/{}.png'.format(i - 20))

for i in range(9):
    PATH = '../DRIVE/test/1st_manual/0{}_manual1.gif'.format(i + 1)
    image = Image.open(PATH).convert('L')
    image.save('./data/val/label/{}.png'.format(i + 1))












