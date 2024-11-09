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
from torch.utils.data import Dataset
import os
import imageio
from . import transforms
import cv2



def ZeroPaddingResizeCV(img, size=(512, 512), interpolation=None):
    isize = img.shape
    ih, iw, ic = isize[0], isize[1], isize[2]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)

    #cv2.resize: (H,W,1)->(H,W);(H,W,3)->(H,W,3)
    img = cv2.resize(img, (new_w, new_h), interpolation)

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    new_img = np.zeros((h, w, ic), np.uint8)
    new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img

    return new_img

