#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <stdio.h>
#include <iostream>
#include <ctime>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void conv(int* a, int* b, int* ker, int n, int m, int kernelSize);

int main()
{
    /*int d_Count = 1;
    for (int devNo = 0; devNo < d_Count; devNo++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf(" Device Number: %d\n", devNo);
        printf(" Device name: %s\n", prop.name);
        printf(" No. of MultiProcessors: %d\n", prop.multiProcessorCount);
        printf(" Compute Capability: %d, %d\n", prop.major, prop.minor);
        printf(" Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf(" Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf(" Peak Memory Bandwidth (GB/s) : %8.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf(" Total amount of Global Memory : %dKB \n", prop.totalGlobalMem / 1024);
        printf(" Total amount of Const Memory: %dKB\n", prop.totalConstMem / 1024);
        printf(" Total of Shared Memory per block : %dKB\n", prop.sharedMemPerBlock / 1024);
        printf(" Total of Shared Memory per MP: %dKB\n", prop.sharedMemPerBlock / 1024);
        printf(" Warp Size: %d\n", prop.warpSize);
        printf(" Max. threds per block: %d\n", prop.maxThreadsPerBlock);
        printf(" Max.threds per MP : %d\n", prop.maxThreadsPerMultiProcessor);
        printf(" Maximum number of warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf(" Maximum Grid size: (%d, %d, %d) \n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf(" Maximum block dimension: (%d, %d , %d) \n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }*/

    const int len = 3;
    const int n = 8;
    int size = n * n * sizeof(int);
    int sizeLen = len * len * sizeof(int);

    int a[n * n];
    int b[n * n];
    int c[n * n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 3;
        b[i] = rand() % 3;
    }

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, sizeLen);

    c[0] = 0; c[1] = 1; c[2] = 0; c[3] = 0; c[4] = 0; c[5] = 0; c[6] = 0; c[7] = 0; c[8] = 0;

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, sizeLen, cudaMemcpyHostToDevice);

    // Solve Operation
    dim3 block(8, 8);
    conv << <1, block >> > (devA, devB, devC, n, n, len);

    // Print solution
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", c[i * n + j]);
        }
        printf("\n");
    }

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void conv(int* a, int* b, int* ker, int n, int m, int kernelSize) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    if (row > 0 && row < m - 1 && col>0 && col < n - 1) {
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                sum += (a[(row - 1) * m + i + (col - 1) + j] * ker[i * kernelSize + j]);
            }
        }

        b[row * m + col] = sum;
    }
}
