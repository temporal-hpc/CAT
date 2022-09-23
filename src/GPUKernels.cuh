#pragma once

// rules
#define EL 2
#define EU 3
#define FL 3
#define FU 3
#define SHMEM_N BSIZE3DX + HALO_SIZE

#define HINDEX(x, y, nWithHalo) ((y + 1) * ((size_t)nWithHalo) + (x + 1))
#define GINDEX(x, y, nshmem) ((y) * (nshmem) + (x))

__device__ inline int h(int k, int a, int b) {
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__forceinline__ __device__ void workWithShmem(MTYPE* pDataOut, MTYPE* shmem, uint2 dataCoord, uint32_t nWithHalo, uint32_t nShmem) {
    // neighborhood count
    int nc
        = shmem[HINDEX(threadIdx.x - 1, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y - 1, nShmem)]
        + shmem[HINDEX(threadIdx.x - 1, threadIdx.y, nShmem)] /*                                                     */ + shmem[HINDEX(threadIdx.x + 1, threadIdx.y, nShmem)]
        + shmem[HINDEX(threadIdx.x - 1, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y + 1, nShmem)];

    unsigned int c = shmem[HINDEX(threadIdx.x, threadIdx.y, nShmem)];
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
}

// Step function for a game of life (GOL) CA in 2D VERSION 1
// This base solution uses shared memory
// Each block of threads loads into shared memory its corresponding region to work
// The halo is loaded into the shared memory using the first 4 warps (one for each side), while the center
// is loaded by each thread using its local coordinate.
// So that the worked region size == the block size
//__forceinline__ __device__ void loadDataToShmem(MTYPE* data, MTYPE* shmem, )
__global__ void ClassicV1GoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo) {

    __shared__ MTYPE shmem[(SHMEM_N) * (SHMEM_N)];
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };

    // haloSideToCopy assigns to each warp a side of the halo to be coppied to shared memory
    // when there are not enough warps (i.e. 4) it uses the thread.y coordinate, so that each
    // row of the block copies a side of the halo. this can be computed with a static if :)
    uint32_t haloSideToCopy = (BSIZE3DX * BSIZE3DY) / 32 < 4 ? threadIdx.y : tid / 32;
    // cacheSize is a helper variable used when there are not enough rows to copy all 4 sides
    // in that case, each row copies at least 2 sides. worst case scenario, the row 0 copies all 4 sides
    uint32_t cacheSize = min(BSIZE3DY, (uint32_t)n);

    if (dataCoord.x < n && dataCoord.y < n) {
        shmem[HINDEX(threadIdx.x, threadIdx.y, SHMEM_N)] = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];

        // TOP
        if (haloSideToCopy == (0 & (cacheSize - 1))) {
            shmem[HINDEX(threadIdx.x, -1, SHMEM_N)] = pDataIn[HINDEX(dataCoord.x, dataBlockCoord_y - 1, nWithHalo)];
        }
        // BOTTOM
        if (haloSideToCopy == (1 & (cacheSize - 1))) {
            shmem[HINDEX(threadIdx.x, BSIZE3DX, SHMEM_N)] = pDataIn[HINDEX(dataCoord.x, dataBlockCoord_y + BSIZE3DX, nWithHalo)];
        }
        // LEFT
        if (haloSideToCopy == (2 & (cacheSize - 1))) {
            shmem[HINDEX(-1, threadIdx.x, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x - 1, dataBlockCoord_y + threadIdx.x, nWithHalo)];
        }
        // RIGHT
        if (haloSideToCopy == (3 & (cacheSize - 1))) {
            shmem[HINDEX(BSIZE3DX, threadIdx.x, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x + BSIZE3DX, dataBlockCoord_y + threadIdx.x, nWithHalo)];
        }

        // Local thread 0 in charge of the four corners
        if (threadIdx.x + threadIdx.y == 0) {
            shmem[HINDEX(-1, -1, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x - 1, dataBlockCoord_y - 1, nWithHalo)];
            shmem[HINDEX(BSIZE3DX, -1, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x + BSIZE3DX, dataBlockCoord_y - 1, nWithHalo)];
            shmem[HINDEX(-1, BSIZE3DX, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x - 1, dataBlockCoord_y + BSIZE3DY, nWithHalo)];
            shmem[HINDEX(BSIZE3DX, BSIZE3DX, SHMEM_N)] = pDataIn[HINDEX(dataBlockCoord_x + BSIZE3DX, dataBlockCoord_y + BSIZE3DY, nWithHalo)];
        }
    }
    __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x + threadIdx.y == 0) {
    //     for (int k = 0; k < SHMEM_N; k++) {
    //         for (int l = 0; l < SHMEM_N; l++) {
    //             printf("%i ", shmem[k * SHMEM_N + l]);
    //         }
    //         printf("\n");
    //     }
    // }
    if (dataCoord.x < n && dataCoord.y < n) {
        workWithShmem(pDataOut, shmem, dataCoord, nWithHalo, SHMEM_N);
    }
}

// Step function for a game of life (GOL) CA in 2D VERSION 2
// This base solution uses shared memory
// Each block of threads loads into shared memory its corresponding region to work PLUS ITS HALO
// The halo is loaded into the shared memory trivially as the
// worked region size + halo == block size
//__forceinline__ __device__ void loadDataToShmem(MTYPE* data, MTYPE* shmem, )
__global__ void ClassicV2GoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo) {

    __shared__ MTYPE shmem[(BSIZE3DX) * (BSIZE3DY)];
    // Assuming that the total halo increase the size by 2
    uint32_t fixedBlockDim_x = blockDim.x - HALO_SIZE;
    uint32_t fixedBlockDim_y = blockDim.y - HALO_SIZE;

    // uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t dataBlockCoord_x = blockIdx.x * fixedBlockDim_x;
    uint32_t dataBlockCoord_y = blockIdx.y * fixedBlockDim_y;

    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };

    if (dataCoord.x < nWithHalo && dataCoord.y < nWithHalo) {
        shmem[GINDEX(threadIdx.x, threadIdx.y, BSIZE3DX)] = pDataIn[GINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
    }
    __syncthreads();
    // if (blockIdx.x == 1 && blockIdx.y == 3 && threadIdx.x + threadIdx.y == 0) {
    //     for (int k = 0; k < BSIZE3DX; k++) {
    //         for (int l = 0; l < BSIZE3DX; l++) {
    //             printf("%i ", shmem[k * BSIZE3DX + l]);
    //         }
    //         printf("\n");
    //     }
    // }
    if (dataCoord.x < nWithHalo - 1 && dataCoord.y < nWithHalo - 1) {
        if (threadIdx.x > 0 && threadIdx.x < BSIZE3DX - 1 && threadIdx.y > 0 && threadIdx.y < BSIZE3DY - 1) {
            // neighborhood count
            int nc
                = shmem[GINDEX(threadIdx.x - 1, threadIdx.y - 1, BSIZE3DX)] + shmem[GINDEX(threadIdx.x, threadIdx.y - 1, BSIZE3DX)] + shmem[GINDEX(threadIdx.x + 1, threadIdx.y - 1, BSIZE3DX)]
                + shmem[GINDEX(threadIdx.x - 1, threadIdx.y, BSIZE3DX)] /*                                                       */ + shmem[GINDEX(threadIdx.x + 1, threadIdx.y, BSIZE3DX)]
                + shmem[GINDEX(threadIdx.x - 1, threadIdx.y + 1, BSIZE3DX)] + shmem[GINDEX(threadIdx.x, threadIdx.y + 1, BSIZE3DX)] + shmem[GINDEX(threadIdx.x + 1, threadIdx.y + 1, BSIZE3DX)];

            unsigned int c = shmem[GINDEX(threadIdx.x, threadIdx.y, BSIZE3DX)];
            pDataOut[GINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
        }
    }
}