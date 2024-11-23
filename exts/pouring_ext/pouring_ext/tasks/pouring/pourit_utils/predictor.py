'''
# -*- coding: utf-8 -*-
# @Project   : afa
# @File      : demo.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/2/25 10:16

# @Desciption: 
'''

# import _init_path
import argparse
import os

import numpy as np
from copy import deepcopy
import torch
from collections import OrderedDict
from omegaconf import OmegaConf

from .net_cam2d import CamNet
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from . import transforms
from .pourit import ZeroPaddingResizeCV, ZeroPaddingResizeCVSingleChannel
from .imutils import denormalize_img
from .camutils import (cam_valid, multi_scale_cam)

import cv2
import matplotlib.pyplot as plt
import time
import signal

import rospy
from std_msgs.msg import Float64MultiArray



class LiquidPredictor():
    def __init__(self, cfg, args ):
        self.cfg = cfg
        self.args = args
        self.initialization_camnet(self.cfg, self.args)

        self.T_obj2cam = None
        self.T_cam2base = None


    def initialization_camnet(self, cfg, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.camNet = CamNet(backbone=cfg.backbone.config,
                           stride=cfg.backbone.stride,
                           num_classes=cfg.dataset.num_classes,
                           embedding_dim=256,
                           pretrained=True,
                           pooling=args.pooling, )

        trained_state_dict = torch.load(args.model_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v

        self.camNet.load_state_dict(state_dict=new_state_dict, strict=True)
        self.camNet.eval()
        self.camNet.to(self.device)


    @torch.no_grad()
    def inference(self, input_image, input_size):
        
        # Change shape if input is just a single image
        if len(input_image.shape)<4:
            input_image = np.expand_dims(input_image,0)

        img = ZeroPaddingResizeCV(input_image, size=(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size))
        img = transforms.normalize_img(img)
        img = np.transpose(img, (0, 3, 1, 2))
        img_tensor = torch.tensor(img)
        img_tensor_cuda = img_tensor.to(self.device)
        img_denorm_tensor = denormalize_img(img_tensor)

        torch.cuda.synchronize()
        start_time = time.time()

        cls_pred, cam = multi_scale_cam(self.camNet.half(), inputs=img_tensor_cuda.half(), scales=[1.])
        cls_pred = (torch.sum(cls_pred)>0).type(torch.int16) #(origin, flip_origin)
        valid_cam = cam_valid(cam, cls_pred)  
        # valid_cam = torch.transpose(valid_cam, 2, 3) # Transpose
        valid_cam = valid_cam.type(torch.float32)
        valid_cam = ZeroPaddingResizeCVSingleChannel(valid_cam.permute([0,2,3,1]).cpu().numpy(), size=input_size)
        valid_cam = torch.tensor(valid_cam).type(torch.float32).permute([0,3,1,2])

        return valid_cam







