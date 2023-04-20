#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <time.h>

void bubble_sortCPU(int* a, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                int aux = a[j + 1];
                a[j + 1] = a[j];
                a[j] = aux;
            }
        }
    }
}
__global__ void bubble_sortGPU(int* a, int n) {

    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {

        int offset = i % 2;
        int leftInd = 2 * tid + offset;
        int rightInd = leftInd + 1;

        if (rightInd < n) {
            if (a[leftInd] > a[rightInd]) {
                int aux = a[leftInd];
                a[leftInd] = a[rightInd];
                a[rightInd] = aux;
            }
        }
        __syncthreads();
    }
}
int main() {

    int size = 10;
    int* host_a, * res;
    int* dev_a;

    host_a = (int*)malloc(size * sizeof(int));
    res = (int*)malloc(size * sizeof(int));

    cudaMalloc(&dev_a, size * sizeof(size));

    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (256));
        host_a[i] = r1;
        printf("%d ", host_a[i]);
    }

    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(size);

    bubble_sortGPU << <grid, block >> > (dev_a, size);
    cudaMemcpy(res, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    bubble_sortCPU(host_a, size);

    printf("CPU: \n");

    for (int i = 0; i < size; i++) {
        printf("%d ", host_a[i]);
    }

    printf("\n");
    printf("GPU\n");

    for (int i = 0; i < size; i++) {
        printf("%d ", res[i]);
    }

    printf("\n");

    return 0;

}
