"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn as nn
import torch

class VGGEncoder1(nn.Module):
  def __init__(self, vgg1):
    super(VGGEncoder1, self).__init__()
    # 224 x 224
    self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
    self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
    self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
    # 224 x 224
    self.reflect_pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
    # 226 x 226
    self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
    self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
    self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())
    self.relu = nn.ReLU(inplace=True)
    # 224 x 224

  def forward(self, x):
    out = self.conv1(x)
    out = self.reflect_pad1(out)
    out = self.conv2(out)
    out = self.relu(out)
    return out


class VGGDecoder1(nn.Module):
  def __init__(self, d1):
    super(VGGDecoder1, self).__init__()
    self.reflect_pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
    # 226 x 226
    self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
    self.conv3.weight = torch.nn.Parameter(d1.get(1).weight.float())
    self.conv3.bias = torch.nn.Parameter(d1.get(1).bias.float())
    # 224 x 224

  def forward(self, x):
    out = self.reflect_pad2(x)
    out = self.conv3(out)
    return out


class VGGEncoder2(nn.Module):
  def __init__(self, vgg):
    super(VGGEncoder2, self).__init__()
    # 224 x 224
    self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
    self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
    self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
    self.reflect_pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
    # 226 x 226

    self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
    self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
    self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
    self.relu2 = nn.ReLU(inplace=True)
    # 224 x 224

    self.reflect_pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
    self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
    self.relu3 = nn.ReLU(inplace=True)
    # 224 x 224

    self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 112 x 112

    self.reflect_pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
    self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
    self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
    self.relu4 = nn.ReLU(inplace=True)
    # 112 x 112

  def forward(self, x):
    out = self.conv1(x)
    out = self.reflect_pad2(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.reflect_pad3(out)
    out = self.conv3(out)
    pool = self.relu3(out)
    out, pool_idx = self.maxPool(pool)
    out = self.reflect_pad4(out)
    out = self.conv4(out)
    out = self.relu4(out)
    return out, pool_idx, pool.size()


class VGGDecoder2(nn.Module):
  def __init__(self, d):
    super(VGGDecoder2, self).__init__()
    # decoder
    self.reflect_pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
    self.conv5.weight = torch.nn.Parameter(d.get(1).weight.float())
    self.conv5.bias = torch.nn.Parameter(d.get(1).bias.float())
    self.relu5 = nn.ReLU(inplace=True)
    # 112 x 112

    self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # self.unpool = nn.Upsample(2,2)
    # 224 x 224

    self.reflect_pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv6.weight = torch.nn.Parameter(d.get(5).weight.float())
    self.conv6.bias = torch.nn.Parameter(d.get(5).bias.float())
    self.relu6 = nn.ReLU(inplace=True)
    # 224 x 224

    self.reflect_pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
    self.conv7.weight = torch.nn.Parameter(d.get(8).weight.float())
    self.conv7.bias = torch.nn.Parameter(d.get(8).bias.float())

  def forward(self, x, pool_idx, pool):
    out = self.reflect_pad5(x)
    out = self.conv5(out)
    out = self.relu5(out)
    out = self.unpool(out, pool_idx, output_size=pool)
    out = self.reflect_pad6(out)
    out = self.conv6(out)
    out = self.relu6(out)
    out = self.reflect_pad7(out)
    out = self.conv7(out)
    return out


class VGGEncoder3(nn.Module):
  def __init__(self, vgg):
    super(VGGEncoder3, self).__init__()
    # 224 x 224
    self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
    self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
    self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
    self.reflect_pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
    # 226 x 226

    self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
    self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
    self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
    self.relu2 = nn.ReLU(inplace=True)
    # 224 x 224

    self.reflect_pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
    self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
    self.relu3 = nn.ReLU(inplace=True)
    # 224 x 224

    self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 112 x 112

    self.reflect_pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
    self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
    self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
    self.relu4 = nn.ReLU(inplace=True)
    # 112 x 112

    self.reflect_pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
    self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
    self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
    self.relu5 = nn.ReLU(inplace=True)
    # 112 x 112

    self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 56 x 56

    self.reflect_pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
    self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
    self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
    self.relu6 = nn.ReLU(inplace=True)
    # 56 x 56

  def forward(self, x):
    out = self.conv1(x)
    out = self.reflect_pad1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.reflect_pad3(out)
    out = self.conv3(out)
    pool1 = self.relu3(out)
    out, pool_idx = self.maxPool(pool1)
    out = self.reflect_pad4(out)
    out = self.conv4(out)
    out = self.relu4(out)
    out = self.reflect_pad5(out)
    out = self.conv5(out)
    pool2 = self.relu5(out)
    out, pool_idx2 = self.maxPool2(pool2)
    out = self.reflect_pad6(out)
    out = self.conv6(out)
    out = self.relu6(out)
    return out, pool_idx, pool1.size(), pool_idx2, pool2.size()


class VGGDecoder3(nn.Module):
  def __init__(self, d):
    super(VGGDecoder3, self).__init__()
    # decoder
    self.reflect_pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
    self.conv7.weight = torch.nn.Parameter(d.get(1).weight.float())
    self.conv7.bias = torch.nn.Parameter(d.get(1).bias.float())
    self.relu7 = nn.ReLU(inplace=True)
    # 56 x 56

    self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # self.unpool = nn.Upsample(2,2)
    # 112 x 112

    self.reflect_pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
    self.conv8.weight = torch.nn.Parameter(d.get(5).weight.float())
    self.conv8.bias = torch.nn.Parameter(d.get(5).bias.float())
    self.relu8 = nn.ReLU(inplace=True)
    # 112 x 112

    self.reflect_pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
    self.conv9.weight = torch.nn.Parameter(d.get(8).weight.float())
    self.conv9.bias = torch.nn.Parameter(d.get(8).bias.float())
    self.relu9 = nn.ReLU(inplace=True)

    self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # 224 x 224

    self.reflect_pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv10.weight = torch.nn.Parameter(d.get(12).weight.float())
    self.conv10.bias = torch.nn.Parameter(d.get(12).bias.float())
    self.relu10 = nn.ReLU(inplace=True)

    self.reflect_pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
    self.conv11.weight = torch.nn.Parameter(d.get(15).weight.float())
    self.conv11.bias = torch.nn.Parameter(d.get(15).bias.float())

  def forward(self, x, pool_idx, pool1, pool_idx2, pool2):
    out = self.reflect_pad7(x)
    out = self.conv7(out)
    out = self.relu7(out)
    out = self.unpool(out, pool_idx2, output_size=pool2)
    out = self.reflect_pad8(out)
    out = self.conv8(out)
    out = self.relu8(out)
    out = self.reflect_pad9(out)
    out = self.conv9(out)
    out = self.relu9(out)
    out = self.unpool2(out, pool_idx, output_size=pool1)
    out = self.reflect_pad10(out)
    out = self.conv10(out)
    out = self.relu10(out)
    out = self.reflect_pad11(out)
    out = self.conv11(out)
    return out


class VGGEncoder4(nn.Module):
  def __init__(self, vgg):
    super(VGGEncoder4, self).__init__()
    # vgg
    # 224 x 224
    self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
    self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
    self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
    self.reflect_pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
    # 226 x 226

    self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
    self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
    self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
    self.relu2 = nn.ReLU(inplace=True)
    # 224 x 224

    self.reflect_pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
    self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
    self.relu3 = nn.ReLU(inplace=True)
    # 224 x 224

    self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 112 x 112

    self.reflect_pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
    self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
    self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
    self.relu4 = nn.ReLU(inplace=True)
    # 112 x 112

    self.reflect_pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
    self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
    self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
    self.relu5 = nn.ReLU(inplace=True)
    # 112 x 112

    self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 56 x 56

    self.reflect_pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
    self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
    self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
    self.relu6 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
    self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
    self.relu7 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
    self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
    self.relu8 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad9 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
    self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
    self.relu9 = nn.ReLU(inplace=True)
    # 56 x 56

    self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    # 28 x 28

    self.reflect_pad10 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
    self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
    self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
    self.relu10 = nn.ReLU(inplace=True)
    # 28 x 28

  def forward(self, x):
    out = self.conv1(x)
    out = self.reflect_pad1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    out = self.reflect_pad3(out)
    out = self.conv3(out)
    pool1 = self.relu3(out)
    out, pool_idx = self.maxPool(pool1)
    out = self.reflect_pad4(out)
    out = self.conv4(out)
    out = self.relu4(out)
    out = self.reflect_pad5(out)
    out = self.conv5(out)
    pool2 = self.relu5(out)
    out, pool_idx2 = self.maxPool2(pool2)
    out = self.reflect_pad6(out)
    out = self.conv6(out)
    out = self.relu6(out)
    out = self.reflect_pad7(out)
    out = self.conv7(out)
    out = self.relu7(out)
    out = self.reflect_pad8(out)
    out = self.conv8(out)
    out = self.relu8(out)
    out = self.reflect_pad9(out)
    out = self.conv9(out)
    pool3 = self.relu9(out)
    out, pool_idx3 = self.maxPool3(pool3)
    out = self.reflect_pad10(out)
    out = self.conv10(out)
    out = self.relu10(out)
    return out, pool_idx, pool1.size(), pool_idx2, pool2.size(), pool_idx3, pool3.size()


  def forward_multiple(self, x):
    out0 = self.conv1(x)
    out0 = self.reflect_pad1(out0)
    out0 = self.conv2(out0)
    out0 = self.relu2(out0)
    out1 = self.reflect_pad3(out0)
    out1 = self.conv3(out1)
    pool1 = self.relu3(out1)
    out1, pool_idx = self.maxPool(pool1)
    out1 = self.reflect_pad4(out1)
    out1 = self.conv4(out1)
    out1 = self.relu4(out1)
    out2 = self.reflect_pad5(out1)
    out2 = self.conv5(out2)
    pool2 = self.relu5(out2)
    out2, pool_idx2 = self.maxPool2(pool2)
    out2 = self.reflect_pad6(out2)
    out2 = self.conv6(out2)
    out2 = self.relu6(out2)
    out = self.reflect_pad7(out2)
    out = self.conv7(out)
    out = self.relu7(out)
    out = self.reflect_pad8(out)
    out = self.conv8(out)
    out = self.relu8(out)
    out = self.reflect_pad9(out)
    out = self.conv9(out)
    pool3 = self.relu9(out)
    out, pool_idx3 = self.maxPool3(pool3)
    out = self.reflect_pad10(out)
    out = self.conv10(out)
    out = self.relu10(out)
    return out, out2, out1, out0

class VGGDecoder4(nn.Module):
  def __init__(self, d):
    super(VGGDecoder4, self).__init__()
    # decoder
    self.reflect_pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
    self.conv11.weight = torch.nn.Parameter(d.get(1).weight.float())
    self.conv11.bias = torch.nn.Parameter(d.get(1).bias.float())
    self.relu11 = nn.ReLU(inplace=True)
    # 28 x 28

    self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # 56 x 56

    self.reflect_pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv12.weight = torch.nn.Parameter(d.get(5).weight.float())
    self.conv12.bias = torch.nn.Parameter(d.get(5).bias.float())
    self.relu12 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad13 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv13.weight = torch.nn.Parameter(d.get(8).weight.float())
    self.conv13.bias = torch.nn.Parameter(d.get(8).bias.float())
    self.relu13 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad14 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
    self.conv14.weight = torch.nn.Parameter(d.get(11).weight.float())
    self.conv14.bias = torch.nn.Parameter(d.get(11).bias.float())
    self.relu14 = nn.ReLU(inplace=True)
    # 56 x 56

    self.reflect_pad15 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
    self.conv15.weight = torch.nn.Parameter(d.get(14).weight.float())
    self.conv15.bias = torch.nn.Parameter(d.get(14).bias.float())
    self.relu15 = nn.ReLU(inplace=True)
    # 56 x 56

    self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # 112 x 112

    self.reflect_pad16 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
    self.conv16.weight = torch.nn.Parameter(d.get(18).weight.float())
    self.conv16.bias = torch.nn.Parameter(d.get(18).bias.float())
    self.relu16 = nn.ReLU(inplace=True)
    # 112 x 112

    self.reflect_pad17 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
    self.conv17.weight = torch.nn.Parameter(d.get(21).weight.float())
    self.conv17.bias = torch.nn.Parameter(d.get(21).bias.float())
    self.relu17 = nn.ReLU(inplace=True)
    # 112 x 112

    self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    # 224 x 224

    self.reflect_pad18 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
    self.conv18.weight = torch.nn.Parameter(d.get(25).weight.float())
    self.conv18.bias = torch.nn.Parameter(d.get(25).bias.float())
    self.relu18 = nn.ReLU(inplace=True)
    # 224 x 224

    self.reflect_pad19 = nn.ReflectionPad2d((1, 1, 1, 1))
    self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)
    self.conv19.weight = torch.nn.Parameter(d.get(28).weight.float())
    self.conv19.bias = torch.nn.Parameter(d.get(28).bias.float())

  def forward(self, x, pool_idx, pool1, pool_idx2, pool2, pool_idx3, pool3):
    # decoder
    out = self.reflect_pad11(x)
    out = self.conv11(out)
    out = self.relu11(out)
    out = self.unpool(out, pool_idx3, output_size=pool3)
    out = self.reflect_pad12(out)
    out = self.conv12(out)

    out = self.relu12(out)
    out = self.reflect_pad13(out)
    out = self.conv13(out)
    out = self.relu13(out)
    out = self.reflect_pad14(out)
    out = self.conv14(out)
    out = self.relu14(out)
    out = self.reflect_pad15(out)
    out = self.conv15(out)
    out = self.relu15(out)
    out = self.unpool2(out, pool_idx2, output_size=pool2)
    out = self.reflect_pad16(out)
    out = self.conv16(out)
    out = self.relu16(out)
    out = self.reflect_pad17(out)
    out = self.conv17(out)
    out = self.relu17(out)
    out = self.unpool3(out, pool_idx, output_size=pool1)
    out = self.reflect_pad18(out)
    out = self.conv18(out)
    out = self.relu18(out)
    out = self.reflect_pad19(out)
    out = self.conv19(out)
    return out


