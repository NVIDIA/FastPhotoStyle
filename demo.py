"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse

import process_stylization
from photo_wct import PhotoWCT

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
parser.add_argument('--content_image_path', default='./images/content1.png')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path', default='./images/style1.png')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--output_image_path', default='./results/example1.png')
args = parser.parse_args()

# Load model
p_wct = PhotoWCT(args)
p_wct.cuda(0)

process_stylization.stylization(
    p_wct=p_wct,
    content_image_path=args.content_image_path,
    style_image_path=args.style_image_path,
    content_seg_path=args.content_seg_path,
    style_seg_path=args.style_seg_path,
    output_image_path=args.output_image_path,
)
