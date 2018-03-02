import os

import torch
import torch.nn as nn
from torch.utils.serialization import load_lua

from models import VGGEncoder, VGGDecoder
from photo_wct import PhotoWCT


def weight_assign(lua, pth, maps):
    for k, v in maps.items():
        getattr(pth, k).weight = nn.Parameter(lua.get(v).weight.float())
        getattr(pth, k).bias = nn.Parameter(lua.get(v).bias.float())


def photo_wct_loader(p_wct):
    p_wct.e1.load_state_dict(torch.load('pth_models/vgg_normalised_conv1.pth'))
    p_wct.d1.load_state_dict(torch.load('pth_models/feature_invertor_conv1.pth'))
    p_wct.e2.load_state_dict(torch.load('pth_models/vgg_normalised_conv2.pth'))
    p_wct.d2.load_state_dict(torch.load('pth_models/feature_invertor_conv2.pth'))
    p_wct.e3.load_state_dict(torch.load('pth_models/vgg_normalised_conv3.pth'))
    p_wct.d3.load_state_dict(torch.load('pth_models/feature_invertor_conv3.pth'))
    p_wct.e4.load_state_dict(torch.load('pth_models/vgg_normalised_conv4.pth'))
    p_wct.d4.load_state_dict(torch.load('pth_models/feature_invertor_conv4.pth'))


if __name__ == '__main__':
    if not os.path.exists('pth_models'):
        os.mkdir('pth_models')
    
    ## VGGEncoder1
    vgg1 = load_lua('models/vgg_normalised_conv1_1_mask.t7')
    e1 = VGGEncoder(1)
    weight_assign(vgg1, e1, {
        'conv0': 0,
        'conv1_1': 2,
    })
    torch.save(e1.state_dict(), 'pth_models/vgg_normalised_conv1.pth')
    
    ## VGGDecoder1
    inv1 = load_lua('models/feature_invertor_conv1_1_mask.t7')
    d1 = VGGDecoder(1)
    weight_assign(inv1, d1, {
        'conv1_1': 1,
    })
    torch.save(d1.state_dict(), 'pth_models/feature_invertor_conv1.pth')
    
    ## VGGEncoder2
    vgg2 = load_lua('models/vgg_normalised_conv2_1_mask.t7')
    e2 = VGGEncoder(2)
    weight_assign(vgg2, e2, {
        'conv0': 0,
        'conv1_1': 2,
        'conv1_2': 5,
        'conv2_1': 9,
    })
    torch.save(e2.state_dict(), 'pth_models/vgg_normalised_conv2.pth')
    
    ## VGGDecoder2
    inv2 = load_lua('models/feature_invertor_conv2_1_mask.t7')
    d2 = VGGDecoder(2)
    weight_assign(inv2, d2, {
        'conv2_1': 1,
        'conv1_2': 5,
        'conv1_1': 8,
    })
    torch.save(d2.state_dict(), 'pth_models/feature_invertor_conv2.pth')
    
    ## VGGEncoder3
    vgg3 = load_lua('models/vgg_normalised_conv3_1_mask.t7')
    e3 = VGGEncoder(3)
    weight_assign(vgg3, e3, {
        'conv0': 0,
        'conv1_1': 2,
        'conv1_2': 5,
        'conv2_1': 9,
        'conv2_2': 12,
        'conv3_1': 16,
    })
    torch.save(e3.state_dict(), 'pth_models/vgg_normalised_conv3.pth')
    
    ## VGGDecoder3
    inv3 = load_lua('models/feature_invertor_conv3_1_mask.t7')
    d3 = VGGDecoder(3)
    weight_assign(inv3, d3, {
        'conv3_1': 1,
        'conv2_2': 5,
        'conv2_1': 8,
        'conv1_2': 12,
        'conv1_1': 15,
    })
    torch.save(d3.state_dict(), 'pth_models/feature_invertor_conv3.pth')
    
    ## VGGEncoder4
    vgg4 = load_lua('models/vgg_normalised_conv4_1_mask.t7')
    e4 = VGGEncoder(4)
    weight_assign(vgg4, e4, {
        'conv0': 0,
        'conv1_1': 2,
        'conv1_2': 5,
        'conv2_1': 9,
        'conv2_2': 12,
        'conv3_1': 16,
        'conv3_2': 19,
        'conv3_3': 22,
        'conv3_4': 25,
        'conv4_1': 29,
    })
    torch.save(e4.state_dict(), 'pth_models/vgg_normalised_conv4.pth')
    
    ## VGGDecoder4
    inv4 = load_lua('models/feature_invertor_conv4_1_mask.t7')
    d4 = VGGDecoder(4)
    weight_assign(inv4, d4, {
        'conv4_1': 1,
        'conv3_4': 5,
        'conv3_3': 8,
        'conv3_2': 11,
        'conv3_1': 14,
        'conv2_2': 18,
        'conv2_1': 21,
        'conv1_2': 25,
        'conv1_1': 28,
    })
    torch.save(d4.state_dict(), 'pth_models/feature_invertor_conv4.pth')
    
    p_wct = PhotoWCT()
    photo_wct_loader(p_wct)
    torch.save(p_wct.state_dict(), 'PhotoWCTModels/photo_wct.pth')
