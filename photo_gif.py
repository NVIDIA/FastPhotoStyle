"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import division
from PIL import Image
from torch import nn
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter


class GIFSmoothing(nn.Module):
    def forward(self, *input):
        pass
        
    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)

    def process_opencv(self, initImg, contentImg):
        '''
        :param initImg: intermediate output. Either image path or PIL Image
        :param contentImg: content image output. Either path or PIL Image
        :return: stylized output image. PIL Image
        '''
        if type(initImg) == str:
            init_img = cv2.imread(initImg)
            init_img = init_img[2:-2,2:-2,:]
        else:
            init_img = np.array(initImg)[:, :, ::-1].copy()

        if type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
        else:
            cont_img = np.array(contentImg)[:, :, ::-1].copy()

        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img

