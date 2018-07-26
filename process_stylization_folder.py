"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import os
import torch
from photo_wct import PhotoWCT
import process_stylization

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA.')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--folder', type=str, default='examples')
parser.add_argument('--beta', type=float, default=0.9999)
parser.add_argument('--cont_img_ext', type=str, default='.png')
parser.add_argument('--cont_seg_ext', type=str, default='.pgm')
parser.add_argument('--styl_img_ext', type=str, default='.png')
parser.add_argument('--styl_seg_ext', type=str, default='.pgm')
args = parser.parse_args()

folder = args.folder
cont_img_folder = os.path.join(folder, 'content_img')
cont_seg_folder = os.path.join(folder, 'content_seg')
styl_img_folder = os.path.join(folder, 'style_img')
styl_seg_folder = os.path.join(folder, 'style_seg')
outp_img_folder = os.path.join(folder, 'results')
cont_img_list = [f for f in os.listdir(cont_img_folder) if os.path.isfile(os.path.join(cont_img_folder, f))]
cont_img_list.sort()

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))
# Load Propagator
if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.01)
else:
    from photo_smooth import Propagator
    p_pro = Propagator(args.beta)

for f in cont_img_list:
    content_image_path = os.path.join(cont_img_folder, f)
    content_seg_path = os.path.join(cont_seg_folder, f).replace(args.cont_img_ext, args.cont_seg_ext)
    style_image_path = os.path.join(styl_img_folder, f)
    style_seg_path = os.path.join(styl_seg_folder, f).replace(args.styl_img_ext, args.styl_seg_ext)
    output_image_path = os.path.join(outp_img_folder, f)

    print("Content image: " + content_image_path )
    if os.path.isfile(content_seg_path):
        print("Content mask: " + content_seg_path )

    print("Style image: " + style_image_path )
    if os.path.isfile(style_seg_path):
        print("Style mask: " + style_seg_path )

    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_seg_path=content_seg_path,
        style_seg_path=style_seg_path,
        output_image_path=output_image_path,
        cuda=args.cuda,
        save_intermediate=args.save_intermediate,
        no_post=args.no_post
    )
