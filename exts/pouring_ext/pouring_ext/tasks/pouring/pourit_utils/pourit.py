'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : pourit.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/8/22 09:11

# @Desciption: 
'''

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import cv2



def ZeroPaddingResizeCV(img, size=(512, 512), interpolation=None):
    isize = img.shape
    n, ih, iw, ic = isize[0], isize[1], isize[2], isize[3]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)

    #cv2.resize: (H,W,1)->(H,W);(H,W,3)->(H,W,3)
    img_resized = np.zeros((n, new_w, new_h, ic))
    for i in range(isize[0]):
        img_resized[i] = cv2.resize(img[i], (new_w, new_h), interpolation)

    if len(img.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=2)

    new_img = np.zeros((n, h, w, ic), np.uint8)
    new_img[:, (h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img_resized

    return new_img

def ZeroPaddingResizeCVSingleChannel(img, size=(512, 512), interpolation=None):
    isize = img.shape
    n, ih, iw, ic = isize[0], isize[1], isize[2], isize[3]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)

    #cv2.resize: (H,W,1)->(H,W);(H,W,3)->(H,W,3)
    img_resized = np.zeros((n, new_w, new_h))
    for i in range(isize[0]):
        img_resized[i] = cv2.resize(img[i], (new_w, new_h), interpolation)

    if len(img.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=2)

    new_img = np.zeros((n, h, w, ic), np.float32)
    new_img[:, (h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = np.expand_dims(img_resized, 3)

    return new_img


