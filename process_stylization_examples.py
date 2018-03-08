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
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth',
                    help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA.')
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
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

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
        cuda=args.cuda,
    )
