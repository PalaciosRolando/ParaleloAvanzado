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

__global__ void mulMatrixGPU(int* a, int* b, int* c, int width, int rows, int cols);

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

    const int n = 2;
    int size = n * sizeof(n);

    // Declare 2x2 Matrix
    int a[n*n];
    int b[n*n];
    int c[n*n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n*n; i++) {
        a[i] = rand() % 226;
        b[i] = rand() % 226;
    }

    int* devA = 0;
    int* devB = 0;
    int* devC = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, size, cudaMemcpyHostToDevice);

    // Solve Operation
    dim3 grid(8, 4, 4);
    dim3 block(8, 4, 4);
    mulMatrixGPU << <grid, block >> > (devA, devB, devC, 2, 2, 2);

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void mulMatrixGPU(int* a, int* b, int* c, int width, int rows, int cols) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int suma = 0;
    if (row < rows && col < cols) {
        for (int i = 0; i < width; i++) {
            suma += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = suma;
    }

}
