
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define PATCH_SIZE 3
#define SIGMA 10.0f

using namespace cv;

__global__ void nlm_kernel(const float* d_image, float* d_result, int width, int height, int patch_size, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half_patch_size = patch_size / 2;
    float h = sigma * sigma;
    float sum_weights = 0.0f;
    float sum_pixel_values = 0.0f;

    for (int dy = -half_patch_size; dy <= half_patch_size; ++dy) {
        for (int dx = -half_patch_size; dx <= half_patch_size; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            float diff = d_image[ny * width + nx] - d_image[y * width + x];
            float weight = expf(-(diff * diff) / h);

            sum_weights += weight;
            sum_pixel_values += weight * d_image[ny * width + nx];
        }
    }

    d_result[y * width + x] = sum_pixel_values / sum_weights;
}

void nlm_denoise(const Mat& input_image, Mat& output_image, int patch_size, float sigma) {
    int width = input_image.cols;
    int height = input_image.rows;

    // Allocate memory on the device
    float* d_image;
    float* d_result;
    size_t image_size = width * height * sizeof(float);
    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_result, image_size);

    // Copy the input image to the device
    cudaMemcpy(d_image, input_image.ptr<float>(), image_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    nlm_kernel<<<gridDim, blockDim>>>(d_image, d_result, width, height, patch_size, sigma);

    // Copy the result back to the host
    cudaMemcpy(output_image.ptr<float>(), d_result, image_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_result);
}

int main() {
    // Load the preprocessed image (assume the image is preprocessed and normalized)
    Mat input_image = imread("data/preprocessed/image.tiff", IMREAD_GRAYSCALE);
    input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);

    // Create an output image
    Mat output_image = input_image.clone();

    // Denoise the image using NLM with CUDA
    nlm_denoise(input_image, output_image, PATCH_SIZE, SIGMA);

    // Convert the result to 8-bit and save
    output_image.convertTo(output_image, CV_8U, 255.0);
    imwrite("data/preprocessed/denoised_image_gpu.tiff", output_image);

    // Display the original and denoised images
    imshow("Original Image", input_image);
    imshow("Denoised Image", output_image);
    waitKey(0);

    return 0;
}
