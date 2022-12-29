#pragma once
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;

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

__forceinline__ __device__ void workWithGbmem(MTYPE* pDataIn, MTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo) {
    // neighborhood count
    int nc
        = pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y - 1, nWithHalo)]
        + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y, nWithHalo)] /*                                                     */ + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y, nWithHalo)]
        + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y + 1, nWithHalo)];

    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
}

__global__ void ClassicGlobalMemGoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo) {
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };
    if (dataCoord.x < n && dataCoord.y < n) {
        workWithGbmem(pDataIn, pDataOut, dataCoord, nWithHalo);
    }
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

// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorV1GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {

    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t nFragmentsH = NREGIONS_H + 2; // ⚠️ The code relies on this being 8 !!
    const uint32_t nFragmentsV = NREGIONS_V + 2;

    // a total of nFragmentsH * nFragmentsV fragments of 16x16
    const uint32_t nShmemH = nFragmentsH * 16;
    const uint32_t nShmemV = nFragmentsV * 16;
    __shared__ FTYPE shmem[nShmemH * nShmemV];

    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    // printf("%i\n", tid);
    uint32_t wid = tid / 32;

    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
    for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
        // uint32_t index = (-i) % (16 + 1);
        // shmem_tridiag[i] = tridiagTemplate[index];
    }
    for (uint32_t i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);

        /*if (i == 15 * 16) {
            shmem_tridiag[i + 16 * 16] = 1;
        } else {
            shmem_tridiag[i + 16 * 16] = 0;
        }*/
    }
    /*
    if (tid < 16 * 16) {
        // if (tid < 16 * 16) {
        shmem_tridiag[tid] = tridiagTemplate[(1 - tid) % (16 + 1)];
        //} else {
        // This create the fragment T that has one '1' at the top right corner
        if (tid == 15) {
            shmem_tridiag[tid + 16 * 16] = 1;
        } else {
            shmem_tridiag[tid + 16 * 16] = 0;
        }

        //}
    }*/
    /*__syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("tridiag 1: \n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
            }
            printf("\n");
        }
    }*/
    __syncthreads();
    //__syncthreads();

    // Copying the corresponding global memory region to shared

    for (uint32_t index = tid; index < nShmemH * nShmemV; index += BSIZE3DX * BSIZE3DY) {
        // printf("%i\n", index);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_x = blockIdx.x * NREGIONS_H * 16 + (index % nShmemH); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_y = blockIdx.y * NREGIONS_V * 16 + (index / nShmemH); //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        // printf()
        // printf("%i -- x,y = (%i, %i) -> %llu\n", index, dataCoord_x, dataCoord_y, (dataCoord_y)*nWithHalo + (dataCoord_x));
        size_t dindex = (dataCoord_y)*nWithHalo + (dataCoord_x);

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is shmem to spare
        if (dataCoord_x < nWithHalo && dataCoord_y < nWithHalo) {
            shmem[index] = pDataIn[dindex];
        }
        // shmem[index] = pDataIn[HINDEX(dataCoord_x - 16, dataCoord_y - 16, nWithHalo)];
        //   }
    }
    __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    //     printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB; // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA; // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA; // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA; // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {

        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= nWithHalo / 16) {
            continue;
        }
        // printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);
        //  wmma::fill_fragment(c_frag, 0.0f);

        // Reducing horizontal neighbours
        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nShmemH * 16 + workFragment_x * 16], nShmemH);
        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);
        /*
        wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("LEFT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    //  printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
        wmma::load_matrix_sync(T_1_asB, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, a_frag, T_1_asB, c_frag);
        /*wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("Center wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 1) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nShmemH * 16 + (workFragment_x + 2) * 16], nShmemH);
        wmma::load_matrix_sync(T_2_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_2_asB, c_frag);
        /*wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("RIGHT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        // printf("%i, %i\n", (workFragment_x + 1) * 16, workFragment_y * 16);
        wmma::store_matrix_sync(&shmem[workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16], c_frag, nShmemH, wmma::mem_row_major);
        //__syncthreads();
        /*__syncthreads();

        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < nShmemV; i++) {
                for (int j = 0; j < nShmemH; j++) {
                    printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
                }
                printf("\n");
            }
        }
        //__syncthreads();*/
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

    /*
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("\n");
        printf("\n");

        // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
        for (int i = 0; i < nShmemV; i++) {
            for (int j = 0; j < nShmemH; j++) {
                printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
            }
            printf("\n");
        }
    }
    __syncthreads();
    // wmma::fill_fragment(c_frag, 0.0f);
    __syncthreads();*/
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= n / 16) {
            continue;
        }
        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        // Reducing horizontal neighbours
        wmma::load_matrix_sync(b_frag, &shmem[workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
        wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);
        /*wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("TOP wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    //  printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        wmma::load_matrix_sync(b_frag, &shmem[(workFragment_y + 1) * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
        wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);
        /*wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("Center wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 1) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/
        wmma::load_matrix_sync(b_frag, &shmem[(workFragment_y + 2) * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
        wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);
        /*wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("\n");
            printf("BOT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 1) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();
        __syncthreads();*/

        wmma::store_matrix_sync(&shmem[(workFragment_y + 1) * nShmemH * 16 + (workFragment_x + 1) * 16], c_frag, nShmemH, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
    // __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        //.printf("%i\n", tid);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_x = blockIdx.x * NREGIONS_H * 16 + (index % (NREGIONS_H * 16)); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_y = blockIdx.y * NREGIONS_V * 16 + (index / (NREGIONS_H * 16)); //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        // printf()
        // printf("%i, %i -- x,y = (%i, %i) -> %llu\n", tid, index, dataCoord_x, dataCoord_y, (dataCoord_y)*nWithHalo + (dataCoord_x));
        size_t dindex = (dataCoord_y + 16) * nWithHalo + (dataCoord_x + 16);

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is shmem to spare
        if (dataCoord_x < n && dataCoord_y < n) {
            uint32_t val = __half2uint_rn(shmem[((index / (NREGIONS_H * 16)) + 16) * nShmemH + index % (NREGIONS_H * 16) + 16]);
            float val2 = __half2float(pDataIn[dindex]);
            // printf("%f\n", (float)val);
            // printf("%i -> %llu = %i ----- val: %i\n", (((index / (NREGIONS_H * 16)) + 16) * nShmemH + index % (NREGIONS_H * 16) + 16), dindex, val2, val);

            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
        }
        // shmem[index] = pDataIn[HINDEX(dataCoord_x - 16, dataCoord_y - 16, nWithHalo)];
        //   }
    }
}
// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorCoalescedV1GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {

    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t nFragmentsH = NREGIONS_H + 2; // ⚠️ The code relies on this being 8 !!
    const uint32_t nFragmentsV = NREGIONS_V + 2;

    // a total of nFragmentsH * nFragmentsV fragments of 16x16
    const uint32_t nShmemH = nFragmentsH * 16;
    const uint32_t nShmemV = nFragmentsV * 16;
    __shared__ FTYPE shmem[nShmemH * nShmemV];

    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t wid = tid / 32;

    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
    for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
        // shmem_tridiag[i] = i / 16 == i % 16 ? 1 : 0;
        //  uint32_t index = (-i) % (16 + 1);
        //  shmem_tridiag[i] = tridiagTemplate[index];
    }
    for (uint32_t i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
        // shmem_tridiag[i + 16 * 16] = i / 16 == i % 16 ? 1 : 0;

        /*if (i == 15 * 16) {
            shmem_tridiag[i + 16 * 16] = 1;
        } else {
            shmem_tridiag[i + 16 * 16] = 0;
        }*/
    }
    ////for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
    //    shmem_tridiag[i + 256 * 2] = i / 16 == i % 16 ? 1 : 0;
    //}
    /*
    if (tid < 16 * 16) {
        // if (tid < 16 * 16) {
        shmem_tridiag[tid] = tridiagTemplate[(1 - tid) % (16 + 1)];
        //} else {
        // This create the fragment T that has one '1' at the top right corner
        if (tid == 15) {
            shmem_tridiag[tid + 16 * 16] = 1;
        } else {
            shmem_tridiag[tid + 16 * 16] = 0;
        }

        //}
    }*/
    /*__syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("tridiag 1: \n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
            }
            printf("\n");
        }
    }*/
    __syncthreads();
    //__syncthreads();

    // Copying the corresponding global memory region to shared

    for (uint32_t index = tid; index < nShmemH * nShmemV; index += BSIZE3DX * BSIZE3DY) {
        // printf("%i\n", index);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * nWithHalo / 16 + globalFragment_x * 256 + tid % 256;

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is shmem to spare
        if (globalFragment_x < nWithHalo / 16 && globalFragment_y < nWithHalo / 16) {
            // printf("%i -- (%i,%i) = (%i, %i) -> %llu\n", index, regionCoord_x, regionCoord_y, globalFragment_x, globalFragment_y, dindex);
            shmem[index] = pDataIn[dindex];
        }
        // shmem[index] = pDataIn[HINDEX(dataCoord_x - 16, dataCoord_y - 16, nWithHalo)];
        //   }
    }
    __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    //     printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB; // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA; // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA; // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA; // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    // int tts = 6;
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {

        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= nWithHalo / 16) {
            continue;
        }
        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        //  printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x * 16, workFragment_y * 16);
        //  wmma::fill_fragment(c_frag, 0.0f);

        // Reducing horizontal neighbours
        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + workFragment_x * 256], 16);
        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);

        /*__syncthreads();
        if (rid == tts) {
            wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
            printf("\n");
            printf("LEFT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nFragmentsH * 256 + workFragment_x * 256);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    //  printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();
        wmma::fill_fragment(c_frag, 0.0f);*/

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_1_asB, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, a_frag, T_1_asB, c_frag);

        /*
        __syncthreads();
        if (rid == tts) {
            wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
            printf("\n");
            printf("Center wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x + 1, workFragment_y, workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 1) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();
        wmma::fill_fragment(c_frag, 0.0f);
*/
        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 2) * 256], 16);
        wmma::load_matrix_sync(T_2_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_2_asB, c_frag);

        /*
        __syncthreads();
        if (rid == tts) {
            wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
            printf("\n");
            printf("RIGHT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x + 2, workFragment_y, workFragment_y * nFragmentsH * 256 + (workFragment_x + 2) * 256);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        // printf("%i, %i\n", (workFragment_x + 1) * 16, workFragment_y * 16);
        wmma::store_matrix_sync(&shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        //__syncthreads();
        /*__syncthreads();

        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < nShmemV; i++) {
                for (int j = 0; j < nShmemH; j++) {
                    printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
                }
                printf("\n");
            }
        }
        //__syncthreads();*/
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

    /*if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        printf("\n");
        printf("\n");

        // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
        for (int i = 0; i < nShmemV; i++) {
            for (int j = 0; j < nShmemH; j++) {
                printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
            }
            printf("\n");
        }
    }
    __syncthreads();
    // wmma::fill_fragment(c_frag, 0.0f);
    __syncthreads();*/
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= n / 16) {
            continue;
        }
        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        // Reducing horizontal neighbours   workFragment_y * nFragmentsH * 256 + workFragment_x * 256
        wmma::load_matrix_sync(b_frag, &shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);

        /*
        __syncthreads();
        if (rid == tts) {
            wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
            printf("\n");
            printf("top wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x + 1, workFragment_y, workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    //  printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();
        wmma::fill_fragment(c_frag, 0.0f);
*/
        wmma::load_matrix_sync(b_frag, &shmem[(workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);
        /*
                __syncthreads();
                if (rid == tts) {
                    wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
                }
                if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
                    printf("\n");
                    printf("Center wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, (workFragment_x + 1), (workFragment_y + 1), (workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256);

                    // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 16; j++) {
                            printf("%.0f ", __half2float(aux[i * 16 + j]));
                            // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 1) * 16 + j]));
                        }
                        printf("\n");
                    }
                }
                __syncthreads();
                wmma::fill_fragment(c_frag, 0.0f);*/

        wmma::load_matrix_sync(b_frag, &shmem[(workFragment_y + 2) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);

        /*__syncthreads();
        if (rid == tts) {
            wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        }
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 32 * tts) {
            printf("\n");
            printf("bot wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x + 1, workFragment_y+2, (workFragment_y + 2) * nFragmentsH * 256 + (workFragment_x + 1) * 256);

            // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", __half2float(aux[i * 16 + j]));
                    // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        wmma::store_matrix_sync(&shmem[(workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
    // __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {

        // printf("%i\n", index);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH) + 1;
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH)) + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * nWithHalo / 16 + globalFragment_x * 256 + tid % 256;

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is shmem to spare
        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {
            uint32_t val = __half2uint_rn(shmem[index + 256 * (nFragmentsH + 1)]);
            float val2 = __half2float(pDataIn[dindex]);
            // printf("%i -- (%i,%i) = (%i, %i) -> %llu\n", index, regionCoord_x, regionCoord_y, globalFragment_x, globalFragment_y, dindex);

            // shmem[index] = pDataIn[dindex];
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
        }
    }
}
__global__ void convertFp32ToFp16(FTYPE* out, MTYPE* in, int nWithHalo) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo) {
        out[tx + ty * nWithHalo] = __uint2half_rn(in[tx + ty * nWithHalo]);
    }
}
__global__ void convertFp16ToFp32(MTYPE* out, FTYPE* in, int nWithHalo) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo) {
        out[tx + ty * nWithHalo] = __half2uint_rn(in[tx + ty * nWithHalo]);
    }
}

__global__ void convertFp32ToFp16AndDoChangeLayout(FTYPE* out, MTYPE* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);

    if (tx < nWithHalo && ty < nWithHalo) {
        out[bid * 256 + tid] = __uint2half_rn(in[ty * nWithHalo + tx]);
        // out[bid * 256 + tid] = __uint2half_rn(ty * nWithHalo + tx);
    }
}
__global__ void convertFp16ToFp32AndUndoChangeLayout(MTYPE* out, FTYPE* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);

    if (tx < nWithHalo && ty < nWithHalo) {
        out[ty * nWithHalo + tx] = __half2uint_rn(in[bid * 256 + tid]);
    }
}