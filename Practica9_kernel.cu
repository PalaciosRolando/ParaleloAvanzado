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

__global__ void search(int* a, int n, int* pos, int searchNum);


int main()
{
    const int n = 8;
    int searchNum = 5;
    int size = n * sizeof(int);
    
    int* a, * ans, *pos;

    // Allocate space for local variables
    a = (int*)malloc(size);
    ans = (int*)malloc(size);
    pos = (int*)malloc(sizeof(int));

    // Initialilze before the array
    pos[0] = -1;

    // Assign and print a random value to every position
    printf("Assigned values\n");
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 256;
        printf("%d ", a[i]);
    }

    // Print separation line
    printf("\n");

    int* devA, *devPos;

    // Allocate Memory
    cudaMalloc(&devA, size);
    cudaMalloc(&devPos, sizeof(int));

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPos, pos, sizeof(int), cudaMemcpyHostToDevice);

    // Declare grid
    dim3 block(1024);
    dim3 grid(n >= 1024 ? n / 1024 : 1);
    /*if (n >= 1024) {
        dim3 grid(n / 1024);
    }
    else {
        dim3 grid(1);
    }*/


    // Solve operations
    search << <grid, block >> > (devA, n, devPos, searchNum);
    cudaDeviceSynchronize();
    cudaMemcpy(pos, devPos, sizeof(int), cudaMemcpyDeviceToHost);

    // Print solution
    if (pos[0] == -1) {
        printf("Element wasn't found\n");
    }
    else {
        printf("Element found at: %d postion\n", pos[0]);
    }


    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devPos);

    return 0;
}

__global__ void search(int* a, int n, int* pos, int searchNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        if (a[tid] == searchNum) {
            *pos = tid;
        }
    }
}
