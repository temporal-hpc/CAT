#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#define RANDBSIZE_X 32
#define RANDBSIZE_Y 16

__global__ void kernel_setup_prngs(uint64_t n, uint64_t seed, curandState *states);

template <typename T>
__global__ void kernel_random_array(uint64_t n, int halo, float density, T max, curandState *states, T *array);
template <typename T> curandState *setup_curand(uint64_t n, int seed, curandState *devStates);
template <typename T> T *genRandCA(T *d_array, size_t n, int halo, float density, T max, curandState *devStates);

template <typename T> T *genRandCA_LTL(T *d_array, size_t n, int halo, float density, uint64_t seed);

#include "randCA.tpp"
