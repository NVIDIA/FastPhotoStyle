"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
from smooth_filter import smooth_filter
from process_stylization import Timer, memory_limit_image_resize
from scipy.io import loadmat
colors = loadmat('segmentation/data/color150.mat')['colors']


def overlay(img, pred_color, blend_factor=0.4):
    import cv2
    edges = cv2.Canny(pred_color, 20, 40)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=1)
    out = (1-blend_factor)*img + blend_factor * pred_color
    edge_pixels = (edges==255)
    new_color = [0,0,255]
    for i in range(0,3):
        timg = out[:,:,i]
        timg[edge_pixels]=new_color[i]
        out[:,:,i] = timg
    return out


def visualize_result(label_map):
    label_map = label_map.astype('int')
    label_map_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label in np.unique(label_map):
        label_map_rgb += (label_map == label)[:, :, np.newaxis] * \
            np.tile(colors[label],(label_map.shape[0], label_map.shape[1], 1))
    return label_map_rgb


class SegReMapping:
    def __init__(self, mapping_name, min_ratio=0.02):
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio

    def cross_remapping(self, cont_seg, styl_seg):
        cont_label_info = []
        new_cont_label_info = []
        for label in np.unique(cont_seg):
            cont_label_info.append(label)
            new_cont_label_info.append(label)

        style_label_info = []
        new_style_label_info = []
        for label in np.unique(styl_seg):
            style_label_info.append(label)
            new_style_label_info.append(label)

        cont_set_diff = set(cont_label_info) - set(style_label_info)
        # Find the labels that are not covered by the style
        # Assign them to the best matched region in the style region
        for s in cont_set_diff:
            cont_label_index = cont_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in style_label_info:
                    new_cont_label_info[cont_label_index] = new_label
                    break
        new_cont_seg = cont_seg.copy()
        for i,current_label in enumerate(cont_label_info):
            new_cont_seg[(cont_seg == current_label)] = new_cont_label_info[i]

        cont_label_info = []
        for label in np.unique(new_cont_seg):
            cont_label_info.append(label)
        styl_set_diff = set(style_label_info) - set(cont_label_info)
        valid_styl_set = set(style_label_info) - set(styl_set_diff)
        for s in styl_set_diff:
            style_label_index = style_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in valid_styl_set:
                    new_style_label_info[style_label_index] = new_label
                    break
        new_styl_seg = styl_seg.copy()
        for i,current_label in enumerate(style_label_info):
            # print("%d -> %d" %(current_label,new_style_label_info[i]))
            new_styl_seg[(styl_seg == current_label)] = new_style_label_info[i]

        return new_cont_seg, new_styl_seg

    def self_remapping(self, seg):
        init_ratio = self.min_ratio
        # Assign label with small portions to label with large portion
        new_seg = seg.copy()
        [h,w] = new_seg.shape
        n_pixels = h*w
        # First scan through what are the available labels and their sizes
        label_info = []
        ratio_info = []
        new_label_info = []
        for label in np.unique(seg):
            ratio = np.sum(np.float32((seg == label))[:])/n_pixels
            label_info.append(label)
            new_label_info.append(label)
            ratio_info.append(ratio)
        for i,current_label in enumerate(label_info):
            if ratio_info[i] < init_ratio:
                for j in range(self.label_mapping.shape[0]):
                    new_label = self.label_mapping[j,current_label]
                    if new_label in label_info:
                        index = label_info.index(new_label)
                        if index >= 0:
                            if ratio_info[index] >= init_ratio:
                                new_label_info[i] = new_label
                                break
        for i,current_label in enumerate(label_info):
            new_seg[(seg == current_label)] = new_label_info[i]
        return new_seg


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path,
                style_seg_path, output_image_path,
                cuda, save_intermediate, no_post, label_remapping, output_visualization=False):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        styl_pilimg = styl_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw, new_ch), Image.NEAREST)
            styl_seg.resize((new_sw, new_sh), Image.NEAREST)

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

        cont_seg = label_remapping.self_remapping(cont_seg)
        styl_seg = label_remapping.self_remapping(styl_seg)
        cont_seg, styl_seg = label_remapping.cross_remapping(cont_seg, styl_seg)

        if output_visualization:
            import cv2
            cont_seg_vis = visualize_result(cont_seg)
            styl_seg_vis = visualize_result(styl_seg)
            cont_seg_vis = overlay(cv2.imread(content_image_path), cont_seg_vis)
            styl_seg_vis = overlay(cv2.imread(style_image_path), styl_seg_vis)
            cv2.imwrite(content_seg_path + '.visualization.jpg', cont_seg_vis)
            cv2.imwrite(style_seg_path + '.visualization.jpg', styl_seg_vis)

        if save_intermediate:
            with Timer("Elapsed time in stylization: %f"):
                stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
            if ch != new_ch or cw != new_cw:
                print("De-resize image: (%d,%d)->(%d,%d)" % (new_cw, new_ch, cw, ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch, cw), mode='bilinear')
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
                print("De-resize image: (%d,%d)->(%d,%d)" % (new_cw, new_ch, cw, ch))
                stylized_img = nn.functional.upsample(stylized_img, size=(ch, cw), mode='bilinear')
            grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            out_img = Image.fromarray(ndarr)

            with Timer("Elapsed time in propagation: %f"):
                out_img = smoothing_module.process(out_img, cont_pilimg)

            if no_post is False:
                with Timer("Elapsed time in post processing: %f"):
                    out_img = smooth_filter(out_img, cont_pilimg, f_radius=15, f_edge=1e-1)
            out_img.save(output_image_path)
    return

