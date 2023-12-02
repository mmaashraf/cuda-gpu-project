#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

__global__ void boxBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    // Box blur kernel (same as previous example)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int channels = 3; // assuming 3 color channels (RGB)
        int blurRadius = 3; // adjust as needed

        int pixelIndex = (row * width + col) * channels;

        for (int c = 0; c < channels; ++c) {
            float result = 0.0f;

            for (int i = -blurRadius; i <= blurRadius; ++i) {
                for (int j = -blurRadius; j <= blurRadius; ++j) {
                    int curRow = min(max(row + i, 0), height - 1);
                    int curCol = min(max(col + j, 0), width - 1);

                    int curIndex = (curRow * width + curCol) * channels + c;
                    result += input[curIndex];
                }
            }

            output[pixelIndex + c] = static_cast<unsigned char>(result / ((2 * blurRadius + 1) * (2 * blurRadius + 1)));
        }
    }
}

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    // Gaussian blur kernel (same as previous example)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int channels = 3; // assuming 3 color channels (RGB)
        int blurRadius = 3; // adjust as needed

        int pixelIndex = (row * width + col) * channels;

        for (int c = 0; c < channels; ++c) {
            float result = 0.0f;
            float totalWeight = 0.0f;

            for (int i = -blurRadius; i <= blurRadius; ++i) {
                for (int j = -blurRadius; j <= blurRadius; ++j) {
                    int curRow = min(max(row + i, 0), height - 1);
                    int curCol = min(max(col + j, 0), width - 1);

                    int curIndex = (curRow * width + curCol) * channels + c;
                    float weight = expf(-(i * i + j * j) / (2.0f * blurRadius * blurRadius));
                    result += input[curIndex] * weight;
                    totalWeight += weight;
                }
            }

            output[pixelIndex + c] = static_cast<unsigned char>(result / totalWeight);
        }
    }
}

int main() {
    // Read an image from file
    cv::Mat inputImage = cv::imread("input.jpg");
    if (inputImage.empty()) {
        fprintf(stderr, "Could not open or find the image.\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Allocate device memory
    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * inputImage.channels());
    cudaMalloc((void**)&d_output, width * height * inputImage.channels());

    // Copy input image to device
    cudaMemcpy(d_input, inputImage.data, width * height * inputImage.channels(), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    // Launch the box blur kernel
    boxBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Wait for completion before proceeding

    // Copy the result back to the host
    unsigned char* boxBlurredImage = (unsigned char*)malloc(width * height * inputImage.channels());
    cudaMemcpy(boxBlurredImage, d_output, width * height * inputImage.channels(), cudaMemcpyDeviceToHost);

    // Create a box-blurred image using the output data
    cv::Mat boxBlurredMat(height, width, CV_8UC3, boxBlurredImage);

    // Display the box-blurred image
    // cv::imshow("Box-Blurred Image", boxBlurredMat);
    // cv::waitKey(0);

    // Save the box-blurred image
    cv::imwrite("box_blurred_output.jpg", boxBlurredMat);

    // Launch the Gaussian blur kernel
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Wait for completion before proceeding

    // Copy the result back to the host
    unsigned char* gaussianBlurredImage = (unsigned char*)malloc(width * height * inputImage.channels());
    cudaMemcpy(gaussianBlurredImage, d_output, width * height * inputImage.channels(), cudaMemcpyDeviceToHost);

    // Create a Gaussian-blurred image using the output data
    cv::Mat gaussianBlurredMat(height, width, CV_8UC3, gaussianBlurredImage);

    // Display the Gaussian-blurred image
    // cv::imshow("Gaussian-Blurred Image", gaussianBlurredMat);
    // cv::waitKey(0);

    // Save the Gaussian-blurred image
    cv::imwrite("gaussian_blurred_output.jpg", gaussianBlurredMat);

    // Free allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(boxBlurredImage);
    free(gaussianBlurredImage);

    return 0;
}
