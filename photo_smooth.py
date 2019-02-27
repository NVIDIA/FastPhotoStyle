"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import division
import torch.nn as nn
import scipy.misc
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided
from PIL import Image


class Propagator(nn.Module):
    def __init__(self, beta=0.9999):
        super(Propagator, self).__init__()
        self.beta = beta

    def process(self, initImg, contentImg):

        if type(contentImg) == str:
            content = scipy.misc.imread(contentImg, mode='RGB')
        else:
            content = contentImg.copy()
        # content = scipy.misc.imread(contentImg, mode='RGB')

        if type(initImg) == str:
            B = scipy.misc.imread(initImg, mode='RGB').astype(np.float64) / 255
        else:
            B = scipy.asarray(initImg).astype(np.float64) / 255
            # B = self.
        # B = scipy.misc.imread(initImg, mode='RGB').astype(np.float64)/255
        h1,w1,k = B.shape
        h = h1 - 4
        w = w1 - 4
        B = B[int((h1-h)/2):int((h1-h)/2+h),int((w1-w)/2):int((w1-w)/2+w),:]
        content = scipy.misc.imresize(content,(h,w))
        B = self.__replication_padding(B,2)
        content = self.__replication_padding(content,2)
        content = content.astype(np.float64)/255
        B = np.reshape(B,(h1*w1,k))
        W = self.__compute_laplacian(content)
        W = W.tocsc()
        dd = W.sum(0)
        dd = np.sqrt(np.power(dd,-1))
        dd = dd.A.squeeze()
        D = scipy.sparse.csc_matrix((dd, (np.arange(0,w1*h1), np.arange(0,w1*h1)))) # 0.026
        S = D.dot(W).dot(D)
        A = scipy.sparse.identity(w1*h1) - self.beta*S
        A = A.tocsc()
        solver = scipy.sparse.linalg.factorized(A)
        V = np.zeros((h1*w1,k))
        V[:,0] = solver(B[:,0])
        V[:,1] = solver(B[:,1])
        V[:,2] = solver(B[:,2])
        V = V*(1-self.beta)
        V = V.reshape(h1,w1,k)
        V = V[2:2+h,2:2+w,:]
        
        img = Image.fromarray(np.uint8(np.clip(V * 255., 0, 255.)))
        return img

    # Returns sparse matting laplacian
    # The implementation of the function is heavily borrowed from
    # https://github.com/MarcoForte/closed-form-matting/blob/master/closed_form_matting.py
    # We thank Marco Forte for sharing his code.
    def __compute_laplacian(self, img, eps=10**(-7), win_rad=1):
            win_size = (win_rad*2+1)**2
            h, w, d = img.shape
            c_h, c_w = h - 2*win_rad, w - 2*win_rad
            win_diam = win_rad*2+1
            indsM = np.arange(h*w).reshape((h, w))
            ravelImg = img.reshape(h*w, d)
            win_inds = self.__rolling_block(indsM, block=(win_diam, win_diam))
            win_inds = win_inds.reshape(c_h, c_w, win_size)
            winI = ravelImg[win_inds]
            win_mu = np.mean(winI, axis=2, keepdims=True)
            win_var = np.einsum('...ji,...jk ->...ik', winI, winI)/win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)
            inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))
            X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
            vals = (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))
            nz_indsCol = np.tile(win_inds, win_size).ravel()
            nz_indsRow = np.repeat(win_inds, win_size).ravel()
            nz_indsVal = vals.ravel()
            L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
            return L

    def __replication_padding(self, arr,pad):
            h,w,c = arr.shape
            ans = np.zeros((h+pad*2,w+pad*2,c))
            for i in range(c):
                    ans[:,:,i] = np.pad(arr[:,:,i],pad_width=(pad,pad),mode='edge')
            return ans

    def __rolling_block(self, A, block=(3, 3)):
        shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
        strides = (A.strides[0], A.strides[1]) + A.strides
        return as_strided(A, shape=shape, strides=strides)