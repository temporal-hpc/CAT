#pragma once

// Step function for a game of life CA in 2D
// This base solution uses shared memory
// Each block of threads loads into shared memory its corresponding region to work, including the halo
// The halo is loaded into the shared memory using the first 4 warps (one for each side), while the center
// is loaded by each thread using its local coordinate.
__global__ void gameOfLifeStep2D(MTYPE* pDataIn, MTYPE* pDataOut, uint32_t n, uint32_t nWithHalo) {

    __shared__ MTYPE shmem[(BSIZE3DX + HALO_SIZE)][(BSIZE3DY + HALO_SIZE)];
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    size_t dataCoord = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x;
    shmem[threadIdx.y, threadIdx.x] = pDataIn[]
}