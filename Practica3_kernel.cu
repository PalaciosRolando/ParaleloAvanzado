#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void idxCalc3D();
void sumArrayCPU(int* a, int* b, int* c, int size);

__global__ void sumArrayGPU(int* a, int* b, int* c, int size);

void sumArrayCPU3(int* a, int* b, int* c, int* d, int size);
__global__ void sumArrayGPU3(int* a, int* b, int* c, int* d, int size);

int main()
{
    //3D
    //dim3 grid(2, 2, 2);
    //dim3 block(2, 2, 2);
    //idxCalc3D << <grid, block >> > ();

    const int n = 10000;
    int size = n * sizeof(n);

    int a[n];
    int b[n];
    int c[n];
    int d[n];
    int comp[n];

    // Assign random value between 0-255 to every position
    for (int i = 0; i < n; i++) {
        a[i] = rand()%226;
        b[i] = rand()%226;
        c[i] = rand()%226;
    }
    
    int* devA = 0;
    int* devB = 0;
    int* devC = 0;
    int* devD = 0;

    // Allocate Memory
    cudaMalloc((void**)&devA, size);
    cudaMalloc((void**)&devB, size);
    cudaMalloc((void**)&devC, size);
    cudaMalloc((void**)&devD, size);

    // Copy to GPU
    cudaMemcpy(devA, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devC, c, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devD, d, size, cudaMemcpyHostToDevice);

    //Solve Operation CPU
    //sumArrayCPU(a, b, c, n);
    sumArrayCPU3(a, b, c, d, n);

    // Solve Operation GPU
    //sumArrayGPU << <79, 128 >> > (devA, devB, devC, size);
    dim3 grid(8,4,4);
    dim3 block(8,4,4);

    sumArrayGPU3 << <grid, block >> > (devA, devB, devC, devD, size);

    // Check Array 
    bool equal = true;
    cudaMemcpy(comp, devD, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (comp[i] != d[i]) {
            equal = false;
            printf("Different");
            return;
        }
    }
    if (equal)
        printf("Equal");

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}

__global__ void idxCalc3D() {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; //thread ID + offset

    printf("[DEVICE] gid: %d\n\r", gid);
}

void sumArrayCPU(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
        //printf("[HOST] %d + %d = %d\n\r", a[i], b[i], c[i]);
    }
}

__global__ void sumArrayGPU(int* a, int* b, int* c, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; //thread ID + offset

    if (gid < size) {
        c[gid] = a[gid] + b[gid];
        //printf("[DEVICE] %d + %d = %d\n\r", a[gid], b[gid], c[gid]);
    }

}

void sumArrayCPU3(int* a, int* b, int* c, int* d, int size) {
    for (int i = 0; i < size; i++) {
        d[i] = a[i] + b[i] + c[i];
        //printf("[HOST] %d + %d + %d = %d\n\r", a[i], b[i], c[i], d[i]);
    }
}

__global__ void sumArrayGPU3(int* a, int* b, int* c, int* d, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;

    int tid = threadIdx.x //1D
        + threadIdx.y * blockDim.x //2D
        + threadIdx.z * blockDim.x * blockDim.y; //3D

    int bid = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + blockIdx.z * gridDim.x * gridDim.y; //3D

    int gid = tid + bid * totalThreads; //thread ID + offset

    if (gid < size) {
        d[gid] = a[gid] + b[gid] + c[gid];
        //printf("[DEVICE] %d + %d + %d = %d\n\r", a[gid], b[gid], c[gid], d[gid]);
    }

}
