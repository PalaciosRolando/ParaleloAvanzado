#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void pow(int *a, int *b, int *res){
  int i = threadIdx.x;
  res[i] = a[i] * b[i];
}

int main(){
  
  const int n = 3; 
  int size = n * sizeof(n);
  
  int a[n] = { 1, 7, 1 );
  int b[n] = { 4, 7, 1 );
  int res[n] = { 0, 0, 0 );
  
  int* devA = 0;
  int* devB = 0;
  int* devRes = 0;              
                
  cudaMalloc((void**)&devA, size);
  cudaMalloc((void**)&devB, size);
  cudaMalloc((void**)&devRes, size);
    
  cudaMemcpy(devA, a, size, cudaMempcyHostToDevice);
  cudaMemcpy(devB, b, size, cudaMempcyHostToDevice);
  cudaMemcpy(devRes, res, size, cudaMempcyHostToDevice);
  
  pow << <1, n >> > (devA, devB, devRes);
  cudaDeviceSynchronize();

  cudaMemcpy(res, devRes, size, cudaMemcpyDeviceToHost); 

  printf("{%d, %d, %d}", res[0], res[1], res[2]);
  cudaDeviceReset();
  
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devRes);
  
  return 0;
}
