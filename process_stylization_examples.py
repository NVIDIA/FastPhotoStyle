"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import argparse

from photo_wct import PhotoWCT
import process_stylization

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

folder = 'examples'
cont_img_folder = os.path.join(folder, 'content_img')
cont_seg_folder = os.path.join(folder, 'content_seg')
styl_img_folder = os.path.join(folder, 'style_img')
styl_seg_folder = os.path.join(folder, 'style_seg')
outp_img_folder = os.path.join(folder, 'results')
cont_img_list = [f for f in os.listdir(cont_img_folder) if os.path.isfile(os.path.join(cont_img_folder, f))]
cont_img_list.sort()

# Load model
p_wct = PhotoWCT(args)
p_wct.cuda(0)

for f in cont_img_list:
    print("Process " + f)
    
    content_image_path = os.path.join(cont_img_folder, f)
    content_seg_path = os.path.join(cont_seg_folder, f).replace(".png", ".pgm")
    style_image_path = os.path.join(styl_img_folder, f)
    style_seg_path = os.path.join(styl_seg_folder, f).replace(".png", ".pgm")
    output_image_path = os.path.join(outp_img_folder, f)
    
    process_stylization.stylization(
        p_wct=p_wct,
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_seg_path=content_seg_path,
        style_seg_path=style_seg_path,
        output_image_path=output_image_path,
    )
