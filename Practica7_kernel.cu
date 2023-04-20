#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void trans(int* a, int* b, int n);
__global__ void conv(int* a, int* b, int n);


int main()
{
    const int n = 4;
    int size = n * n * sizeof(int);

    int a[n * n];
    int b[n * n];

    // Assign random value to every position
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 256;
        b[i] = 0;
    }

    // Print assigned values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", a[i * n + j]);
        }
        printf("\n");
    }
    // Print separation line
    printf("\n");

    int* devA = 0;
    int* devB = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);

    // Declare grid
    dim3 block(32, 32);
    dim3 grid(32 / (n * n), 32 / (n * n));

    // Solve operations
    trans << <1, block >> > (devA, devB, n);
    // conv << <1, block >> > (devA, devB, n);
    cudaMemcpy(b, devB, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    dim3 block2(32, 32);
    dim3 grid2((64 + (n * n) - 1) / (n * n), (64 + (n * n) - 1) / (n * n));
    conv << <grid2, block2 >> > (devA, devB, n);
    cudaMemcpy(b, devB, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();


    // Print solution
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", b[i * n + j]);
        }
        printf("\n");
    }


    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);

    return 0;
}

__global__ void trans(int* a, int* b, int n) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;
    if (gid < n * n) {
        s[row * n + col] = a[row * n + col];
        __syncthreads();
        b[col * n + row] = s[row * n + col];
    }
}

__global__ void conv(int* a, int* b, int n) {
    __shared__ int s[64];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int offsetBlock = blockIdx.x * blockDim.x * blockDim.y;
    int offsetRow = blockIdx.y * blockDim.x * blockDim.y * gridDim.x;
    int gid = tid + offsetBlock + offsetRow;


    int row = gid / n;
    int col = gid - row * n;
}
