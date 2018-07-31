"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
src = '''
	#include "/usr/local/cuda/include/math_functions.h"
	#define TB 256
	#define EPS 1e-7

	__device__ bool InverseMat4x4(double m_in[4][4], double inv_out[4][4]) {
		double m[16], inv[16];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				m[i * 4 + j] = m_in[i][j];
			}
		}

	    inv[0] = m[5]  * m[10] * m[15] -
	             m[5]  * m[11] * m[14] -
	             m[9]  * m[6]  * m[15] +
	             m[9]  * m[7]  * m[14] +
	             m[13] * m[6]  * m[11] -
	             m[13] * m[7]  * m[10];

	    inv[4] = -m[4]  * m[10] * m[15] +
	              m[4]  * m[11] * m[14] +
	              m[8]  * m[6]  * m[15] -
	              m[8]  * m[7]  * m[14] -
	              m[12] * m[6]  * m[11] +
	              m[12] * m[7]  * m[10];

	    inv[8] = m[4]  * m[9] * m[15] -
	             m[4]  * m[11] * m[13] -
	             m[8]  * m[5] * m[15] +
	             m[8]  * m[7] * m[13] +
	             m[12] * m[5] * m[11] -
	             m[12] * m[7] * m[9];

	    inv[12] = -m[4]  * m[9] * m[14] +
	               m[4]  * m[10] * m[13] +
	               m[8]  * m[5] * m[14] -
	               m[8]  * m[6] * m[13] -
	               m[12] * m[5] * m[10] +
	               m[12] * m[6] * m[9];

	    inv[1] = -m[1]  * m[10] * m[15] +
	              m[1]  * m[11] * m[14] +
	              m[9]  * m[2] * m[15] -
	              m[9]  * m[3] * m[14] -
	              m[13] * m[2] * m[11] +
	              m[13] * m[3] * m[10];

	    inv[5] = m[0]  * m[10] * m[15] -
	             m[0]  * m[11] * m[14] -
	             m[8]  * m[2] * m[15] +
	             m[8]  * m[3] * m[14] +
	             m[12] * m[2] * m[11] -
	             m[12] * m[3] * m[10];

	    inv[9] = -m[0]  * m[9] * m[15] +
	              m[0]  * m[11] * m[13] +
	              m[8]  * m[1] * m[15] -
	              m[8]  * m[3] * m[13] -
	              m[12] * m[1] * m[11] +
	              m[12] * m[3] * m[9];

	    inv[13] = m[0]  * m[9] * m[14] -
	              m[0]  * m[10] * m[13] -
	              m[8]  * m[1] * m[14] +
	              m[8]  * m[2] * m[13] +
	              m[12] * m[1] * m[10] -
	              m[12] * m[2] * m[9];

	    inv[2] = m[1]  * m[6] * m[15] -
	             m[1]  * m[7] * m[14] -
	             m[5]  * m[2] * m[15] +
	             m[5]  * m[3] * m[14] +
	             m[13] * m[2] * m[7] -
	             m[13] * m[3] * m[6];

	    inv[6] = -m[0]  * m[6] * m[15] +
	              m[0]  * m[7] * m[14] +
	              m[4]  * m[2] * m[15] -
	              m[4]  * m[3] * m[14] -
	              m[12] * m[2] * m[7] +
	              m[12] * m[3] * m[6];

	    inv[10] = m[0]  * m[5] * m[15] -
	              m[0]  * m[7] * m[13] -
	              m[4]  * m[1] * m[15] +
	              m[4]  * m[3] * m[13] +
	              m[12] * m[1] * m[7] -
	              m[12] * m[3] * m[5];

	    inv[14] = -m[0]  * m[5] * m[14] +
	               m[0]  * m[6] * m[13] +
	               m[4]  * m[1] * m[14] -
	               m[4]  * m[2] * m[13] -
	               m[12] * m[1] * m[6] +
	               m[12] * m[2] * m[5];

	    inv[3] = -m[1] * m[6] * m[11] +
	              m[1] * m[7] * m[10] +
	              m[5] * m[2] * m[11] -
	              m[5] * m[3] * m[10] -
	              m[9] * m[2] * m[7] +
	              m[9] * m[3] * m[6];

	    inv[7] = m[0] * m[6] * m[11] -
	             m[0] * m[7] * m[10] -
	             m[4] * m[2] * m[11] +
	             m[4] * m[3] * m[10] +
	             m[8] * m[2] * m[7] -
	             m[8] * m[3] * m[6];

	    inv[11] = -m[0] * m[5] * m[11] +
	               m[0] * m[7] * m[9] +
	               m[4] * m[1] * m[11] -
	               m[4] * m[3] * m[9] -
	               m[8] * m[1] * m[7] +
	               m[8] * m[3] * m[5];

	    inv[15] = m[0] * m[5] * m[10] -
	              m[0] * m[6] * m[9] -
	              m[4] * m[1] * m[10] +
	              m[4] * m[2] * m[9] +
	              m[8] * m[1] * m[6] -
	              m[8] * m[2] * m[5];

	    double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	    if (abs(det) < 1e-9) {
	        return false;
	    }


	    det = 1.0 / det;

	    for (int i = 0; i < 4; i++) {
	    	for (int j = 0; j < 4; j++) {
	    		inv_out[i][j] = inv[i * 4 + j] * det;
	    	}
	    }

	    return true;
	}

  extern "C"
	__global__ void best_local_affine_kernel(
		float *output, float *input, float *affine_model,
		int h, int w, float epsilon, int kernel_radius
	)
	{
		int size = h * w;
		int id = blockIdx.x * blockDim.x + threadIdx.x;

		if (id < size) {
			int x = id % w, y = id / w;

			double Mt_M[4][4] = {}; // 4x4
			double invMt_M[4][4] = {};
			double Mt_S[3][4] = {}; // RGB -> 1x4
			double A[3][4] = {};
			for (int i = 0; i < 4; i++)
				for (int j = 0; j < 4; j++) {
					Mt_M[i][j] = 0, invMt_M[i][j] = 0;
					if (i != 3) {
						Mt_S[i][j] = 0, A[i][j] = 0;
						if (i == j)
				    		Mt_M[i][j] = 1e-3;
				    }
				}

			for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
				for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {

					int xx = x + dx, yy = y + dy;
					int id2 = yy * w + xx;

					if (0 <= xx && xx < w && 0 <= yy && yy < h) {

						Mt_M[0][0] += input[id2 + 2*size] * input[id2 + 2*size];
						Mt_M[0][1] += input[id2 + 2*size] * input[id2 + size];
						Mt_M[0][2] += input[id2 + 2*size] * input[id2];
						Mt_M[0][3] += input[id2 + 2*size];

						Mt_M[1][0] += input[id2 + size] * input[id2 + 2*size];
						Mt_M[1][1] += input[id2 + size] * input[id2 + size];
						Mt_M[1][2] += input[id2 + size] * input[id2];
						Mt_M[1][3] += input[id2 + size];

						Mt_M[2][0] += input[id2] * input[id2 + 2*size];
						Mt_M[2][1] += input[id2] * input[id2 + size];
						Mt_M[2][2] += input[id2] * input[id2];
						Mt_M[2][3] += input[id2];

						Mt_M[3][0] += input[id2 + 2*size];
						Mt_M[3][1] += input[id2 + size];
						Mt_M[3][2] += input[id2];
						Mt_M[3][3] += 1;

						Mt_S[0][0] += input[id2 + 2*size] * output[id2 + 2*size];
						Mt_S[0][1] += input[id2 + size] * output[id2 + 2*size];
						Mt_S[0][2] += input[id2] * output[id2 + 2*size];
						Mt_S[0][3] += output[id2 + 2*size];

						Mt_S[1][0] += input[id2 + 2*size] * output[id2 + size];
						Mt_S[1][1] += input[id2 + size] * output[id2 + size];
						Mt_S[1][2] += input[id2] * output[id2 + size];
						Mt_S[1][3] += output[id2 + size];

						Mt_S[2][0] += input[id2 + 2*size] * output[id2];
						Mt_S[2][1] += input[id2 + size] * output[id2];
						Mt_S[2][2] += input[id2] * output[id2];
						Mt_S[2][3] += output[id2];
					}
				}
			}

			bool success = InverseMat4x4(Mt_M, invMt_M);

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 4; j++) {
					for (int k = 0; k < 4; k++) {
						A[i][j] += invMt_M[j][k] * Mt_S[i][k];
					}
				}
			}

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 4; j++) {
					int affine_id = i * 4 + j;
					affine_model[12 * id + affine_id] = A[i][j];
				}
			}
		}
		return ;
	}

  extern "C"
	__global__ void bilateral_smooth_kernel(
		float *affine_model, float *filtered_affine_model, float *guide,
		int h, int w, int kernel_radius, float sigma1, float sigma2
	)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int size = h * w;
		if (id < size) {
			int x = id % w;
			int y = id / w;

			double sum_affine[12] = {};
			double sum_weight = 0;
			for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
				for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
					int yy = y + dy, xx = x + dx;
					int id2 = yy * w + xx;
					if (0 <= xx && xx < w && 0 <= yy && yy < h) {
						float color_diff1 = guide[yy*w + xx] - guide[y*w + x];
						float color_diff2 = guide[yy*w + xx + size] - guide[y*w + x + size];
						float color_diff3 = guide[yy*w + xx + 2*size] - guide[y*w + x + 2*size];
						float color_diff_sqr =
							(color_diff1*color_diff1 + color_diff2*color_diff2 + color_diff3*color_diff3) / 3;

						float v1 = exp(-(dx * dx + dy * dy) / (2 * sigma1 * sigma1));
						float v2 = exp(-(color_diff_sqr) / (2 * sigma2 * sigma2));
						float weight = v1 * v2;

						for (int i = 0; i < 3; i++) {
							for (int j = 0; j < 4; j++) {
								int affine_id = i * 4 + j;
								sum_affine[affine_id] += weight * affine_model[id2*12 + affine_id];
							}
						}
						sum_weight += weight;
					}
				}
			}

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 4; j++) {
					int affine_id = i * 4 + j;
					filtered_affine_model[id*12 + affine_id] = sum_affine[affine_id] / sum_weight;
				}
			}
		}
		return ;
	}


  extern "C"
	__global__ void reconstruction_best_kernel(
		float *input, float *filtered_affine_model, float *filtered_best_output,
		int h, int w
	)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int size = h * w;
		if (id < size) {
			double out1 =
				input[id + 2*size] * filtered_affine_model[id*12 + 0] + // A[0][0] +
				input[id + size]   * filtered_affine_model[id*12 + 1] + // A[0][1] +
				input[id]          * filtered_affine_model[id*12 + 2] + // A[0][2] +
									 filtered_affine_model[id*12 + 3]; //A[0][3];
			double out2 =
				input[id + 2*size] * filtered_affine_model[id*12 + 4] + //A[1][0] +
				input[id + size]   * filtered_affine_model[id*12 + 5] + //A[1][1] +
				input[id]          * filtered_affine_model[id*12 + 6] + //A[1][2] +
									 filtered_affine_model[id*12 + 7]; //A[1][3];
			double out3 =
				input[id + 2*size] * filtered_affine_model[id*12 + 8] + //A[2][0] +
				input[id + size]   * filtered_affine_model[id*12 + 9] + //A[2][1] +
				input[id]          * filtered_affine_model[id*12 + 10] + //A[2][2] +
									 filtered_affine_model[id*12 + 11]; // A[2][3];

			filtered_best_output[id] = out1;
			filtered_best_output[id + size] = out2;
			filtered_best_output[id + 2*size] = out3;
		}
		return ;
	}
	'''

import torch
import numpy as np
from PIL import Image
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


def smooth_local_affine(output_cpu, input_cpu, epsilon, patch, h, w, f_r, f_e):
    # program = Program(src.encode('utf-8'), 'best_local_affine_kernel.cu'.encode('utf-8'))
    # ptx = program.compile(['-I/usr/local/cuda/include'.encode('utf-8')])
    program = Program(src, 'best_local_affine_kernel.cu')
    ptx = program.compile(['-I/usr/local/cuda/include'])
    m = function.Module()
    m.load(bytes(ptx.encode()))

    _reconstruction_best_kernel = m.get_function('reconstruction_best_kernel')
    _bilateral_smooth_kernel = m.get_function('bilateral_smooth_kernel')
    _best_local_affine_kernel = m.get_function('best_local_affine_kernel')
    Stream = namedtuple('Stream', ['ptr'])
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

    filter_radius = f_r
    sigma1 = filter_radius / 3
    sigma2 = f_e
    radius = (patch - 1) / 2

    filtered_best_output = torch.zeros(np.shape(input_cpu)).cuda()
    affine_model =    torch.zeros((h * w, 12)).cuda()
    filtered_affine_model =torch.zeros((h * w, 12)).cuda()

    input_ = torch.from_numpy(input_cpu).cuda()
    output_ = torch.from_numpy(output_cpu).cuda()
    _best_local_affine_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[output_.data_ptr(), input_.data_ptr(), affine_model.data_ptr(),
             np.int32(h), np.int32(w), np.float32(epsilon), np.int32(radius)], stream=s
     )

    _bilateral_smooth_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[affine_model.data_ptr(), filtered_affine_model.data_ptr(), input_.data_ptr(), np.int32(h), np.int32(w), np.int32(f_r), np.float32(sigma1), np.float32(sigma2)], stream=s
    )

    _reconstruction_best_kernel(
        grid=(int((h * w) / 256 + 1), 1),
        block=(256, 1, 1),
        args=[input_.data_ptr(), filtered_affine_model.data_ptr(), filtered_best_output.data_ptr(),
        np.int32(h), np.int32(w)], stream=s
    )
    numpy_filtered_best_output = filtered_best_output.cpu().numpy()
    return numpy_filtered_best_output


def smooth_filter(initImg, contentImg, f_radius=15,f_edge=1e-1):
    '''
    :param initImg: intermediate output. Either image path or PIL Image
    :param contentImg: content image output. Either path or PIL Image
    :return: stylized output image. PIL Image
    '''
    if type(initImg) == str:
        initImg = Image.open(initImg).convert("RGB")
    best_image_bgr = np.array(initImg, dtype=np.float32)
    bW, bH, bC = best_image_bgr.shape
    best_image_bgr = best_image_bgr[:, :, ::-1]
    best_image_bgr = best_image_bgr.transpose((2, 0, 1))

    if type(contentImg) == str:
        contentImg = Image.open(contentImg).convert("RGB")
    content_input = contentImg.resize((bH,bW))
    content_input = np.array(content_input, dtype=np.float32)
    content_input = content_input[:, :, ::-1]
    content_input = content_input.transpose((2, 0, 1))
    input_ = np.ascontiguousarray(content_input, dtype=np.float32) / 255.
    _, H, W = np.shape(input_)
    output_ = np.ascontiguousarray(best_image_bgr, dtype=np.float32) / 255.
    best_ = smooth_local_affine(output_, input_, 1e-7, 3, H, W, f_radius, f_edge)
    best_ = best_.transpose(1, 2, 0)
    result = Image.fromarray(np.uint8(np.clip(best_ * 255., 0, 255.)))
    return result
