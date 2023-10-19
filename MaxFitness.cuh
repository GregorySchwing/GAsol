//https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat/27928463#27928463%5B/url%5D
#include <iostream>
#include <stdlib.h>

#define DSIZE 10000
// nTPB should be a power-of-2
#define nTPB 256
#define MAX_KERNEL_BLOCKS 30
#define MAX_BLOCKS ((DSIZE/nTPB)+1)
#define MIN(a,b) ((a>b)?b:a)
#define FLOAT_MIN -999.9f

__device__ volatile float blk_vals[MAX_BLOCKS];
__device__ volatile int   blk_idxs[MAX_BLOCKS];
__device__ int   blk_num = 0;

__global__ void resetDeviceVariable(int resetValue) {
    blk_num = resetValue;
}


template <typename T>
__global__ void max_idx_kernel(const T *data, const int dsize, int *result){

  __shared__ volatile T   vals[nTPB];
  __shared__ volatile int idxs[nTPB];
  __shared__ volatile int last_block;
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  last_block = 0;
  T   my_val = FLOAT_MIN;
  int my_idx = -1;
  // sweep from global memory
  while (idx < dsize){
    if (data[idx] > my_val) {my_val = data[idx]; my_idx = idx;}
    idx += blockDim.x*gridDim.x;}
  // populate shared memory
  vals[threadIdx.x] = my_val;
  idxs[threadIdx.x] = my_idx;
  __syncthreads();
  // sweep in shared memory
  for (int i = (nTPB>>1); i > 0; i>>=1){
    if (threadIdx.x < i)
      if (vals[threadIdx.x] < vals[threadIdx.x + i]) {vals[threadIdx.x] = vals[threadIdx.x+i]; idxs[threadIdx.x] = idxs[threadIdx.x+i]; }
    __syncthreads();}
  // perform block-level reduction
  if (!threadIdx.x){
    blk_vals[blockIdx.x] = vals[0];
    blk_idxs[blockIdx.x] = idxs[0];
    if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
      last_block = 1;}
  __syncthreads();
  if (last_block){
    idx = threadIdx.x;
    my_val = FLOAT_MIN;
    my_idx = -1;
    while (idx < gridDim.x){
      if (blk_vals[idx] > my_val) {my_val = blk_vals[idx]; my_idx = blk_idxs[idx]; }
      idx += blockDim.x;}
  // populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx;
    __syncthreads();
  // sweep in shared memory
    for (int i = (nTPB>>1); i > 0; i>>=1){
      if (threadIdx.x < i)
        if (vals[threadIdx.x] < vals[threadIdx.x + i]) {vals[threadIdx.x] = vals[threadIdx.x+i]; idxs[threadIdx.x] = idxs[threadIdx.x+i]; }
      __syncthreads();}
    if (!threadIdx.x)
      *result = idxs[0];
    }
}
