#pragma once
#include <curand_kernel.h>
#define RANDBSIZE 512

__global__ void kernel_setup_prngs(uint64_t size, uint64_t seed, curandState *states){
    uint64_t id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id < size){
        curand_init(seed, id, 0, &states[id]);
    }
}

template <typename T>
__global__ void kernel_random_array(uint64_t size, float d, T max, curandState *states, T *array){
    uint64_t id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if(id >= size){ return; }
    float x = curand_uniform(&states[id]);
    if(x <= d){
        array[id] = 1;
    }
    else{
        array[id] = 0;
    }
}

curandState* setup_curand(uint64_t size, int seed, curandState *devStates) {
    dim3 block(RANDBSIZE, 1, 1);
    dim3 grid((size+RANDBSIZE-1)/RANDBSIZE, 1, 1); 
    kernel_setup_prngs<<<grid, block>>>(size, seed, devStates);
    cudaDeviceSynchronize();
    return devStates;
}

template <typename T>
T* genRandCA(T *d_array, float d, uint64_t size, T max, curandState* devStates){
    dim3 block(RANDBSIZE, 1, 1);
    dim3 grid((size+RANDBSIZE-1)/RANDBSIZE, 1, 1); 
    kernel_random_array<<<grid,block>>>(size, d, max, devStates, d_array);
    cudaDeviceSynchronize();
    return d_array;
}

template <typename T>
T* genRandCA_LTL(T *d_array, float d, uint64_t size, uint64_t seed){
    // devstates
    curandState *devStates;
    cudaMalloc((void **)&devStates, size * sizeof(curandState));
    devStates = setup_curand(size, seed, devStates);

    // random generation
    genRandCA<T>(d_array, d, size, 2, devStates);

    // clean memory
    cudaFree(devStates);
    return d_array;
}
