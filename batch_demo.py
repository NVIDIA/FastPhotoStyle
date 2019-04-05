"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

This file takes in directories of images of both style and content. It assumes that there is only one style image for
each content image. If this is not the case it will read in the top style image and use that. It should
"""

from __future__ import print_function
import argparse
import torch
import process_stylization
from photo_wct import PhotoWCT
from dataset.get_datasets import get_datasets
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_image_dir', default='./images/content1.png')
parser.add_argument('--content_seg_dir', default=[])
parser.add_argument('--style_image_dir', default='./images/style1.png')
parser.add_argument('--style_seg_dir', default=[])
parser.add_argument('--output_image_dir', default='./results/example1.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
args = parser.parse_args()


datasets = get_datasets(csv_image_cols=['t0_pan', 't0_ps', 't-1_ps'])
print(len(datasets['test']))
for i in range(len(datasets['test'])):
    pan_t0 = datasets['test'][i]['t0_pan']
    ps_t0 = datasets['test'][i]['t0_ps']
    ps_t1 = datasets['test'][i]['t-1_ps']
    print('SHAPES: pan_t0:{}, ps_t0: {}, ps_t-1: {}'.format(pan_t0.shape, ps_t0.shape, ps_t1.shape))

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.001)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(0)

process_stylization.stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=args.content_image_path,
    style_image_path=args.style_image_path,
    content_seg_path=args.content_seg_path,
    style_seg_path=args.style_seg_path,
    output_image_path=args.output_image_path,
    cuda=args.cuda,
    save_intermediate=args.save_intermediate,
    no_post=args.no_post
)
