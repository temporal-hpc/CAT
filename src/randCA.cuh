#pragma once
#include <curand_kernel.h>

/* STEPS TO CREATE RANDOM ARRAYS 
   1) [Host] create the random PRNGs --> call Function: curandStateXORWOW_t* setup_curand(int n, int seed)
   2) [Host] create the random array on device passing the random states --> create_random_array_dev(....)
*/

__global__ void kernel_setup_prngs(uint64_t n, uint64_t seed, curandStateXORWOW_t *states){
    uint64_t id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id < n){
        curand_init(seed, id, 0, &state[id]);
    }
}

template <typename T>
__global__ void kernel_random_array(uint64_t n, T max, curandStateXORWOW_t *state, T *array){
    uint64_t id = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if(id >= n){ return; }
    uint32_t x = curand(&state[id]);
    array[id] = x % max;
}

curandStateXORWOW_t* setup_curand(uint64_t n, int seed) {
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();
    return devStates;
}

template <typename T>
T* genRandCA(T *d_array, uint64_t size, T max, curandStateXORWOW_t* devStates){
    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_random_array<<<grid,block>>>(n, max, devStates, darray);
    cudaDeviceSynchronize();
    return darray;
}

template <typename T>
T* genRandCA_LTL(T *d_array, uint64_t size, uint64_t seed){
    // devstates
    curandStateXORWOW_t *devStates;
    cudaMalloc((void **)&devStates, size * sizeof(curandStateXORWOW_t));
    curandStateXORWOW_t* devStates = setup_curand(size, seed);

    // random generation
    genRandCA<T>(d_array, size, 2, devStates);

    // clean memory
    cudaFree(devStates);
    return d_array;
}
