#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void info() {
    printf("[DEVICE] ThreadIdx %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("[DEVICE] BlockIdx %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("[DEVICE] GridDim %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(){
  
    int idxX = 4;
    int idxY = 4;
    int idxZ = 4;
  
    dim3 block(2, 2, 2);
    dim3 grid(idxX / block.x, idxY / block.y, idxZ / block.z);
  
    info << <grid, block >> > ();

    return 0;
}
