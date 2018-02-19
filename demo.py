"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import argparse
import time
import numpy as np
import cv2
from PIL import Image
from photo_wct import PhotoWCT
from photo_smooth import Propagator
from smooth_filter import smooth_filter


parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--vgg1', default='./models/vgg_normalised_conv1_1_mask.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='./models/vgg_normalised_conv2_1_mask.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='./models/vgg_normalised_conv3_1_mask.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='./models/vgg_normalised_conv4_1_mask.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='./models/vgg_normalised_conv5_1_mask.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='./models/feature_invertor_conv5_1_mask.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='./models/feature_invertor_conv4_1_mask.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='./models/feature_invertor_conv3_1_mask.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='./models/feature_invertor_conv2_1_mask.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='./models/feature_invertor_conv1_1_mask.t7', help='Path to the decoder1')
args = parser.parse_args()

# Load model
p_wct = PhotoWCT(args)
p_pro = Propagator()
p_wct.cuda(0)

content_image_path = "./images/content1.png"
content_seg_path = []
style_image_path = "./images/style1.png"
style_seg_path = []
output_image_path = "results/example1.png"

# Load image
cont_img = Image.open(content_image_path).convert('RGB')
styl_img = Image.open(style_image_path).convert('RGB')
try:
  cont_seg = Image.open(content_seg_path)
  styl_seg = Image.open(style_seg_path)
except:
  cont_seg = []
  styl_seg = []

cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)
cont_img = Variable(cont_img.cuda(0), volatile=True)
styl_img = Variable(styl_img.cuda(0), volatile=True)

cont_seg = np.asarray(cont_seg)
styl_seg = np.asarray(styl_seg)

start_style_time = time.time()
stylized_img = p_wct.transform(cont_img, styl_img, cont_seg, styl_seg)
end_style_time = time.time()
print('Elapsed time in stylization: %f' % (end_style_time - start_style_time))
utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1)

start_propagation_time = time.time()
out_img = p_pro.process(output_image_path, content_image_path)
end_propagation_time = time.time()
print('Elapsed time in propagation: %f' % (end_propagation_time - start_propagation_time))
cv2.imwrite(output_image_path, out_img)

start_postprocessing_time = time.time()
out_img = smooth_filter(output_image_path, content_image_path, f_radius=15, f_edge=1e-1)
end_postprocessing_time = time.time()
print('Elapsed time in post processing: %f' % (end_postprocessing_time - start_postprocessing_time))

out_img.save(output_image_path)
