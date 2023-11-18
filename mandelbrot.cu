/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Utilities and system includes

#include <helper_cuda.h>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void cudaProcess(unsigned int *g_odata, int imgw) {
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    // if (tx == 0 and ty == 0) {
    //     printf("bw: %d, bh: %d\n", bw, bh);
    //     printf("bIdx.x: %d, bIdx.y: %d\n", blockIdx.x, blockIdx.y);
    // }

    // int block_lin_idx = blockIdx.y * gridDim.x + blockIdx.x;
    // int num_blocks = gridDim.x * gridDim.y;
    // float block_color = (float)block_lin_idx / num_blocks * 255;
    // g_odata[y * imgw + x] = rgbToInt(block_color, block_color, block_color);

    // uchar4 c4 = make_uchar4((x & 0x20) ? 255 : 0, 0, (y & 0x20) ? 255 : 0, 0);
    // g_odata[y * imgw + x] = rgbToInt(c4.z, c4.y, c4.x);
    // uchar4 c4 = make_uchar4(blockIdx.x * 8, blockIdx.y * 8, 0, 0);

    // g_odata[y * imgw + x] = rgbToInt(c4.x, c4.y, c4.z);

    // g_odata[y * imgw + x] = rgbToInt((float)x / imgw * 255, (float)y / imgw * 255, 0);


    int iter = 0;
    int maxIter = 100;
    float maxAmp = 1.0;
    float zr = 0.0;
    float zi = 0.0;
    float zr_temp;
    float a = (float)x / imgw;
    float b = (float)y / imgw;
    while (iter < maxIter and zr*zr + zi*zi < maxAmp) {
        zr_temp = zr*zr - zi*zi + a;
        zi = 2*zr*zi + b;
        zr = zr_temp;
        iter++;
    }
    float convergence = (float)iter / maxIter;
    float color = (1.0f - convergence) * 255;
    // printf("a: %.2f, b: %.2f, conv: %.2f\n", a, b, convergence);
    g_odata[y * imgw + x] = rgbToInt(color, color, color);
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                     unsigned int *g_odata, int imgw) {
    cudaProcess<<<grid, block, sbytes>>>(g_odata, imgw);
}
