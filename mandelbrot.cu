#include <helper_cuda.h>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ void mapValueToColor(float value, float *color) {
    float anchors[][4] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.9, .25, .0, .0},
        {0.95, .0, .5, .5},
        {0.99, .0, .0, .75},
        {1.0, 1.0, 1.0, 1.0}
    };

    // Find the segment in which the value lies
    int seg = 0;
    while (seg < sizeof(anchors) / sizeof(anchors[0]) - 1 && value > anchors[seg + 1][0]) {
        seg++;
    }

    // Interpolate between the colors of the two nearest anchor points
    float t = (value - anchors[seg][0]) / (anchors[seg + 1][0] - anchors[seg][0]);
    for (int i = 0; i < 3; i++) {
        color[i] = (1.0 - t) * anchors[seg][i + 1] + t * anchors[seg + 1][i + 1];
    }
    color[3] = 0.5;  // Alpha channel (opacity)
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void draw_image_kernel(unsigned int *g_odata, int2 image_size, 
                                  double2 image_center, double zoom_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= image_size.x | y >= image_size.y) {
        return;
    }

    // Find real and imaginary parts of c = a + bi
    int2 screen_center = {image_size.x / 2, image_size.y / 2};
    double a_norm = (double)(x - screen_center.x) / image_size.x;
    double b_norm = (double)(y - screen_center.y) / image_size.x;  // x and y are both normalized by image width (not a typo)
    double a = a_norm / zoom_factor + image_center.x;
    double b = b_norm / zoom_factor + image_center.y;

    int iter = 0;
    int maxIter = 1000;
    double maxAmp = 1000.0;
    double zr = 0.0;
    double zi = 0.0;
    double zr_temp;
    while (iter < maxIter and zr*zr + zi*zi < maxAmp) {
        zr_temp = zr*zr - zi*zi + a;
        zi = 2*zr*zi + b;
        zr = zr_temp;
        iter++;
    }
    double convergence = (double)iter / maxIter;
    float rgba[4];
    mapValueToColor(1.0 - (float)convergence, rgba);
    for (auto& val : rgba) {
        val *= 255;
    }
    g_odata[y * image_size.x + x] = rgbToInt(rgba[0], rgba[1], rgba[2]);

    // int block_lin_idx = blockIdx.y * gridDim.x + blockIdx.x;
    // int num_blocks = gridDim.x * gridDim.y;
    // float block_color = (float)block_lin_idx / num_blocks * 255;
    // g_odata[y * image_size.x + x] = rgbToInt(block_color, block_color, block_color);

}

extern "C" void draw_image(dim3 grid, dim3 block, 
                                   unsigned int *g_odata, int2 image_size,
                                   double2 image_center, double zoom_factor
                                   ) {
    draw_image_kernel<<<grid, block>>>(g_odata, image_size, image_center, zoom_factor);
}
