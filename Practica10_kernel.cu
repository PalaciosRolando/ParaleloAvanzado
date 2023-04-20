#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

//Desplazamiento de bits
#define DESP 1<<4
#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);};

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct myStruct {
    int x;
    int y;
};

struct despStruct {
    int x[DESP];
    int y[DESP];
};


__global__ void aos(myStruct* data, myStruct* ans, const int size);
__global__ void soa(despStruct* data, despStruct* ans, const int size);


int main()
{
    int despSize = DESP;
    int arrSize = sizeof(despStruct);

    despStruct* data, * ans;
    myStruct* dataAos, * ansAos;

    // Allocate space for local variables
    data = (despStruct*)malloc(sizeof(despStruct));
    dataAos = (myStruct*)malloc(sizeof(myStruct));
    ans = (despStruct*)malloc(sizeof(despStruct));
    ansAos = (myStruct*)malloc(sizeof(myStruct));

    // Set data
    for (int i = 0; i < despSize; i++) {
        data-> x[i] = 1;
        data-> y[i] = 2;
    }

    for (int i = 0; i < despSize; i++) {
        dataAos[i].x = 1;
        dataAos[i].y = 2;
    }

    // Print data
    printf("Soa\n");
    for (int i = 0; i < despSize; i++) {
        printf("x: %d y: %d\n", data-> x[i], data-> y[i]);
    }
    printf("\n");

    printf("Aos\n");
    for (int i = 0; i < despSize; i++) {
        printf("x: %d y: %d\n", dataAos[i].x, dataAos[i].y);
    }
    printf("\n");

    despStruct* devData, * devAns;
    myStruct* devAosData, * devAosAns;

    // Allocate Memory
    cudaMalloc(&devData, sizeof(despStruct));
    cudaMalloc(&devAns, sizeof(despStruct));

    cudaMalloc(&devAosData, sizeof(myStruct));
    cudaMalloc(&devAosAns, sizeof(myStruct));

    // Copy to GPU
    cudaMemcpy(devData, data, sizeof(despStruct), cudaMemcpyHostToDevice);
    cudaMemcpy(devAosData, dataAos, sizeof(myStruct), cudaMemcpyHostToDevice);

    // Declare grid
    dim3 block(32);
    dim3 grid((despSize + 32 - 1) / (block.x));

    // Solve operations SOA
    soa << <grid, block >> > (devData, devAns, despSize);
    cudaMemcpy(ans, devAns, sizeof(despStruct), cudaMemcpyDeviceToHost);

    // Print solution
    printf("SOA:\n");
    for (int i = 0; i < despSize; i++) {
        printf("x: %d y: %d\n", ans-> x[i], ans-> y[i]);
    }


    // Clean
    cudaDeviceSynchronize();

    cudaFree(devData);
    cudaFree(devAns);

    // Solve operations AOS
    aos << <grid, block >> > (devAosData, devAosAns, despSize);
    cudaMemcpy(ansAos, devAosAns, sizeof(myStruct), cudaMemcpyDeviceToHost);

    // Print solution
    printf("AOS:\n");
    for (int i = 0; i < despSize; i++) {
        printf("x: %d y: %d\n", ansAos[i].x, ansAos[i].y);
    }

    // Clean
    cudaDeviceSynchronize();
    cudaDeviceReset();

    cudaFree(devAosData);
    cudaFree(devAosAns);

    return 0;
}

__global__ void aos(myStruct* data, myStruct* ans, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        myStruct temp = data[gid];
        temp.x += 4;
        temp.y += 3;
        ans[gid] = temp;
    }
}

__global__ void soa(despStruct* data, despStruct* ans, const int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        int tmpX = data->x[gid];
        int tmpY = data->y[gid];

        tmpX += 4;
        tmpY += 3;
        ans->x[gid] = tmpX;
        ans->y[gid] = tmpY;
    }
}
