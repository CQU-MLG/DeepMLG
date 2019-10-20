# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from tensorflow.keras import utils

def get_training_set(file_path):
    train_imgs = []
    train_gt = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            img = np.array(ndimage.imread(eachline[0]))
            train_imgs.append(img)
            gt = int(eachline[1])
            train_gt.append(gt)
            line = f.readline()

    train_imgs = np.reshape(train_imgs, newshape=[len(train_imgs), len(train_imgs[0]), len(train_imgs[0][0]), len(train_imgs[0][0][0])])
    # one-hot
    train_gt = utils.to_categorical(train_gt, 2)
    print(np.array(train_imgs).shape)
           
    return train_imgs, train_gt