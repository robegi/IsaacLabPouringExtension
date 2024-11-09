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
from .pourit import ZeroPaddingResizeCV
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
    def inference(self, input_image):

        img = ZeroPaddingResizeCV(input_image, size=(self.cfg.dataset.crop_size, self.cfg.dataset.crop_size))
        img = transforms.normalize_img(img)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img).unsqueeze(0)
        img_tensor_cuda = img_tensor.to(self.device)
        img_denorm_tensor = denormalize_img(img_tensor)

        torch.cuda.synchronize()
        start_time = time.time()

        cls_pred, cam = multi_scale_cam(self.camNet.half(), inputs=img_tensor_cuda.half(), scales=[1.])
        cls_pred = (torch.sum(cls_pred)>0).type(torch.int16) #(origin, flip_origin)
        valid_cam = cam_valid(cam, cls_pred)

        torch.cuda.synchronize()
        end_time = time.time()
        print('Elapsed time = {:.0f} Hz \r'.format(1./(end_time - start_time)), end='', flush=True)

        valid_cam = valid_cam.cpu().float()
        valid_cam = valid_cam.max(dim=1)[0]
        cam_heatmap = plt.get_cmap('plasma')(valid_cam.numpy())[:,:,:,0:3]*255
        cam_heatmap = cam_heatmap[..., ::-1]
        cam_heatmap = np.ascontiguousarray(cam_heatmap)
        cam_heatmap_tensor = torch.from_numpy(cam_heatmap) #RGB to BGR
        cam_cmap_tensor = cam_heatmap_tensor.permute([0, 3, 1, 2]) #(1,3,512,512)
        cam_img = cam_cmap_tensor*0.5 + img_denorm_tensor[:, [2,1,0] ,:, :]*0.5

        cam_img_show = np.transpose(cam_img.squeeze().numpy(), (1,2,0)).astype(np.uint8)
        cam_show = np.transpose(cam_cmap_tensor.squeeze().numpy(), (1,2,0)).astype(np.uint8)

        return cam_img_show







