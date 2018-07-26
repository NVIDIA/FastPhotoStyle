"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from smooth_filter import smooth_filter


class ReMapping:
    def __init__(self):
        self.remapping = []

    def process(self, seg):
        new_seg = seg.copy()
        for k, v in self.remapping.items():
            new_seg[seg == k] = v
        return new_seg


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def memory_limit_image_resize(cont_img):
    # prevent too small or too big images
    MINSIZE=256
    MAXSIZE=960
    orig_width = cont_img.width
    orig_height = cont_img.height
    if max(cont_img.width,cont_img.height) < MINSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((int(cont_img.width*1.0/cont_img.height*MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            cont_img.thumbnail((MINSIZE, int(cont_img.height*1.0/cont_img.width*MINSIZE)), Image.BICUBIC)
    if min(cont_img.width,cont_img.height) > MAXSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((MAXSIZE, int(cont_img.height*1.0/cont_img.width*MAXSIZE)), Image.BICUBIC)
        else:
            cont_img.thumbnail(((int(cont_img.width*1.0/cont_img.height*MAXSIZE), MAXSIZE)), Image.BICUBIC)
    print("Resize image: (%d,%d)->(%d,%d)" % (orig_width, orig_height, cont_img.width, cont_img.height))
    return cont_img.width, cont_img.height


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cuda, save_intermediate, no_post, cont_seg_remapping=None, styl_seg_remapping=None):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)

        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

        if cuda:
            cont_img = cont_img.cuda(0)
            styl_img = styl_img.cuda(0)
            stylization_module.cuda(0)

        # cont_img = Variable(cont_img, volatile=True)
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)
        if cont_seg_remapping is not None:
            cont_seg = cont_seg_remapping.process(cont_seg)
        if styl_seg_remapping is not None:
            styl_seg = styl_seg_remapping.process(styl_seg)

        if save_intermediate:
            with Timer("Elapsed time in stylization: %f"):
                stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
            if ch != new_ch or cw != new_cw:
                print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
            utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1, padding=0)

            with Timer("Elapsed time in propagation: %f"):
                out_img = smoothing_module.process(output_image_path, content_image_path)
            out_img.save(output_image_path)

            if not cuda:
                print("NotImplemented: The CPU version of smooth filter has not been implemented currently.")
                return

            if no_post is False:
                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter(output_image_path, content_image_path, f_radius=15, f_edge=1e-1)
            out_img.save(output_image_path)
        else:
            with Timer("Elapsed time in stylization: %f"):
                stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
            if ch != new_ch or cw != new_cw:
                print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
            grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out_img = Image.fromarray(ndarr)

            with Timer("Elapsed time in propagation: %f"):
                out_img = smoothing_module.process(out_img, cont_pilimg)

            if no_post is False:
                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
            out_img.save(output_image_path)

