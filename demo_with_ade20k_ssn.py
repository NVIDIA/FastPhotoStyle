"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import os
import torch
import process_stylization_ade20k_ssn
from torch import nn
from photo_wct import PhotoWCT
from segmentation.dataset import round2nearest_multiple
from segmentation.models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
from scipy.misc import imread, imresize
import cv2
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model_path', help='folder to model path', default='baseline-resnet50_dilated8-ppm_bilinear_deepsup')
parser.add_argument('--suffix', default='_epoch_20.pth', help="which snapshot to load")
parser.add_argument('--arch_encoder', default='resnet50_dilated8', help="architecture of net_encoder")
parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup', help="architecture of net_decoder")
parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')
parser.add_argument('--num_val', default=-1, type=int, help='number of images to evalutate')
parser.add_argument('--num_class', default=150, type=int, help='number of classes')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize. current only supports 1')
parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')
parser.add_argument('--imgMaxSize', default=1000, type=int, help='maximum input image size of long edge')
parser.add_argument('--padding_constant', default=8, type=int, help='maxmimum downsampling rate of the network')
parser.add_argument('--segm_downsampling_rate', default=8, type=int, help='downsampling rate of the segmentation label')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')

parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth', help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
parser.add_argument('--content_image_path', default="./images/content3.png")
parser.add_argument('--content_seg_path', default='./results/content3_seg.pgm')
parser.add_argument('--style_image_path', default='./images/style3.png')
parser.add_argument('--style_seg_path', default='./results/style3_seg.pgm')
parser.add_argument('--output_image_path', default='./results/example3.png')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=False)
parser.add_argument('--output_visualization', action='store_true', default=False)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
parser.add_argument('--label_mapping', type=str, default='ade20k_semantic_rel.npy')
args = parser.parse_args()

segReMapping = process_stylization_ade20k_ssn.SegReMapping(args.label_mapping)

# Absolute paths of segmentation model weights
SEG_NET_PATH = 'segmentation'
args.weights_encoder = os.path.join(SEG_NET_PATH,args.model_path, 'encoder' + args.suffix)
args.weights_decoder = os.path.join(SEG_NET_PATH,args.model_path, 'decoder' + args.suffix)
args.arch_encoder = 'resnet50_dilated8'
args.arch_decoder = 'ppm_bilinear_deepsup'
args.fc_dim = 2048

# Load semantic segmentation network module
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])

# Load FastPhotoStyle model
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


def segment_this_img(f):
    img = imread(f, mode='RGB')
    img = img[:, :, ::-1]  # BGR to RGB!!!
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in args.imgSize:
        scale = this_short_size / float(min(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, args.padding_constant)
        target_width = round2nearest_multiple(target_width, args.padding_constant)
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input = dict()
    input['img_ori'] = img.copy()
    input['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (img.shape[0],img.shape[1])
    with torch.no_grad():
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            feed_dict = dict()
            feed_dict['img_data'] = timg.cuda()
            feed_dict = async_copy_to(feed_dict, args.gpu_id)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp.cpu() / len(args.imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))
    return preds


cont_seg = segment_this_img(args.content_image_path)
cv2.imwrite(args.content_seg_path, cont_seg)
style_seg = segment_this_img(args.style_image_path)
cv2.imwrite(args.style_seg_path, style_seg)
process_stylization_ade20k_ssn.stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=args.content_image_path,
    style_image_path=args.style_image_path,
    content_seg_path=args.content_seg_path,
    style_seg_path=args.style_seg_path,
    output_image_path=args.output_image_path,
    cuda=True,
    save_intermediate=args.save_intermediate,
    no_post=args.no_post,
    label_remapping=segReMapping,
    output_visualization=args.output_visualization
)
