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

__global__ void unrolling(int* a, int *b, int n);

int main()
{
    const int n = 64;
    int size = n * n * sizeof(int);

    int* a, * b;

    // Allocate space for local variables
    a = (int*)malloc(size);
    b = (int*)malloc(size);

    // Assign and print a random value to every position
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 256;
        b[i] = 0;
        printf("%d ", a[i]);
    }

    printf("Assigned values\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", a[i * n + j]);
        }
        printf("\n");
    }

    int* devA, * devB;

    // Allocate Memory
    cudaMalloc(&devA, size);
    cudaMalloc(&devB, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);

    // Declare grid
    dim3 block(32, 32);

    // Solve operations SOA
    unrolling << <1, block >> > (devA, devB, n);
    cudaMemcpy(b, devB, size, cudaMemcpyDeviceToHost);

    // Print solution
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", b[i * n + j]);
        }
        printf("\n");
    }


    // Clean
    cudaDeviceSynchronize();

    cudaFree(devA);
    cudaFree(devB);

    return 0;
}

__global__ void unrolling(int* a, int *b, int n) {
    int gid = (threadIdx.x + threadIdx.y * blockDim.x) + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y);
    int offset = blockDim.x / 2;

    for (int i = 0; i < (n * n + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y); i += 2)
    {
        if (gid + blockDim.x * blockDim.y * i < n * n) {
            b[(gid % n * n + gid / n) + offset * i] = a[gid + blockDim.x * blockDim.y * i];
        }
        if (gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y < n * n) {
            b[(gid % n * n + gid / n) + offset * i + offset] = a[gid + blockDim.x * blockDim.y * i + blockDim.x * blockDim.y];
        }
    }

}
