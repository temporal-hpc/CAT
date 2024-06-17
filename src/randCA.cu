#include "Defines.h"
#include "randCA.cuh"
#include <cuda.h>
#include <curand_kernel.h>

__global__ void kernel_setup_prngs(uint64_t n, uint64_t seed, curandState *states)
{
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint64_t id = (uint64_t)id_y * (uint64_t)gridDim.x * (uint64_t)blockDim.x + (uint64_t)id_x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if (id_x < n && id_y < n)
    {
        curand_init(seed, id, 0, &states[id]);
    }
}