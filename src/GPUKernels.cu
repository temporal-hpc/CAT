#ifndef _CLASSIC_GOL_KERNELS_H_
#define _CLASSIC_GOL_KERNELS_H_
#include "GPUKernels.cuh"

#include "Defines.h"

#include <cuda.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define SHMEM_N (BSIZE3DX + HALO_SIZE)
#define BMAXLLSHMEM_N (80 + HALO_SIZE)

#define HINDEX(x, y, nWithHalo) ((y + RADIUS) * ((size_t)nWithHalo) + (x + RADIUS))
#define GINDEX(x, y, nshmem) ((y) * ((size_t)nshmem) + (x))

__device__ inline int h(int k, int a, int b) {
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__forceinline__ __device__ void workWithShmem(MTYPE* pDataOut, MTYPE* shmem, uint2 dataCoord, uint32_t nWithHalo, uint32_t nShmem) {
    // neighborhood count
    int nc = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            // nc += pDataIn[i+j];
            nc += shmem[HINDEX(threadIdx.x + j, threadIdx.y + i, nShmem)];
        }
    }
    // int nc
    //    	= shmem[HINDEX(threadIdx.x - 1, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y - 1, nShmem)]
    //     + shmem[HINDEX(threadIdx.x - 1, threadIdx.y    , nShmem)] /*                                                 */ + shmem[HINDEX(threadIdx.x + 1, threadIdx.y,     nShmem)]
    //     + shmem[HINDEX(threadIdx.x - 1, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y + 1, nShmem)];

    unsigned int c = shmem[HINDEX(threadIdx.x, threadIdx.y, nShmem)];
    nc -= c;
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
    // pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;//c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
}

__forceinline__ __device__ void workWithGbmem(MTYPE* pDataIn, MTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo) {
    // neighborhood count
    // int nc
    //    = pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y - 1, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y, nWithHalo)] /*                                                     */ + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y + 1, nWithHalo)];
    int nc = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            // nc += pDataIn[i+j];
            nc += pDataIn[HINDEX(dataCoord.x + j, dataCoord.y + i, nWithHalo)];
        }
    }

    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
    nc -= c;
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
    // pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;
}

__global__ void ClassicGlobalMemGoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo) {
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = {dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y};
    if (dataCoord.x < n && dataCoord.y < n) {
        workWithGbmem(pDataIn, pDataOut, dataCoord, nWithHalo);
    }
}
__forceinline__ __device__ void workWithGbmemHALF(FTYPE* pDataIn, FTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo) {
    // neighborhood count
    int nc = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            // nc += pDataIn[i+j];
            nc += __half2int_rd(pDataIn[HINDEX(dataCoord.x + j, dataCoord.y + i, nWithHalo)]);
        }
    }

    // int nc
    //     = pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y - 1, nWithHalo)]
    //     + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y, nWithHalo)] /*                                                     */ + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y, nWithHalo)]
    //     + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y + 1, nWithHalo)];

    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
    nc -= c;
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
    // pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;
}

__global__ void ClassicGlobalMemHALFGoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = {dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y};
    if (dataCoord.x < n && dataCoord.y < n) {
        workWithGbmemHALF(pDataIn, pDataOut, dataCoord, nWithHalo);
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
    __shared__ MTYPE shmem[(BMAXLLSHMEM_N) * (BMAXLLSHMEM_N)];
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t dataBlockCoord_x = blockIdx.x * 80;
    uint32_t dataBlockCoord_y = blockIdx.y * 80;

    for (uint32_t i = tid; i < BMAXLLSHMEM_N * BMAXLLSHMEM_N; i += BSIZE3DX * BSIZE3DY) {
        uint32_t shmem_x = i % BMAXLLSHMEM_N;
        uint32_t shmem_y = i / BMAXLLSHMEM_N;
        uint32_t data_x = dataBlockCoord_x + shmem_x;
        uint32_t data_y = dataBlockCoord_y + shmem_y;
        if (data_x < nWithHalo && data_y < nWithHalo) {
            shmem[GINDEX(shmem_x, shmem_y, BMAXLLSHMEM_N)] = pDataIn[GINDEX(data_x, data_y, nWithHalo)];
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < 80 * 80; i += BSIZE3DX * BSIZE3DY) {
        uint32_t shmem_x = i % 80;
        uint32_t shmem_y = i / 80;
        uint32_t data_x = dataBlockCoord_x + shmem_x;
        uint32_t data_y = dataBlockCoord_y + shmem_y;
        uint2 dataCoord = {data_x, data_y};
        if (dataCoord.x < n && dataCoord.y < n) {
            int nc = 0;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    // nc += pDataIn[i+j];
                    nc += shmem[HINDEX(shmem_x + j, shmem_y + i, BMAXLLSHMEM_N)];
                }
            }
            unsigned int c = shmem[HINDEX(shmem_x, shmem_y, BMAXLLSHMEM_N)];
            nc -= c;
            pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
            // pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;//c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
        }
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

    uint2 dataCoord = {dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y};

    if (dataCoord.x < nWithHalo && dataCoord.y < nWithHalo) {
        shmem[GINDEX(threadIdx.x, threadIdx.y, BSIZE3DX)] = pDataIn[GINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
    }
    __syncthreads();
    // if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x + threadIdx.y == 0) {
    //     for (int k = 0; k < BSIZE3DY; k++) {
    //         for (int l = 0; l < BSIZE3DX; l++) {
    //             printf("%i ", shmem[k * BSIZE3DX + l]);
    //         }
    //         printf("\n");
    //     }
    // }
    if (dataCoord.x < nWithHalo - RADIUS && dataCoord.y < nWithHalo - RADIUS) {
        if (threadIdx.x > RADIUS - 1 && threadIdx.x < BSIZE3DX - RADIUS && threadIdx.y > RADIUS - 1 && threadIdx.y < BSIZE3DY - RADIUS) {
            // printf("threadIdx.x: %u, threadIdx.y: %u\n", threadIdx.x, threadIdx.y);
            //  neighborhood count
            int nc = 0;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    // nc += pDataIn[i+j];
                    nc += shmem[GINDEX(threadIdx.x + j, threadIdx.y + i, BSIZE3DX)];
                }
            }

            unsigned int c = shmem[GINDEX(threadIdx.x, threadIdx.y, BSIZE3DX)];
            nc -= c;
            pDataOut[GINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
            // pDataOut[GINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;//c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
        }
    }
}

// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorV1GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t nFragmentsH = NREGIONS_H + 2;  // ⚠️ The code relies on this being 8 !!
    const uint32_t nFragmentsV = NREGIONS_V + 2;

    // a total of nFragmentsH * nFragmentsV fragments of 16x16
    const uint32_t nShmemH = nFragmentsH * 16;
    const uint32_t nShmemV = nFragmentsV * 16;
    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;
    FTYPE* shmem2 = (FTYPE*)((nShmemH * nShmemV) * sizeof(FTYPE) + totalshmem);
    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    // printf("%i\n", tid);
    uint32_t wid = tid / 32;

    int i;
    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
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
        uint32_t dataCoord_x = blockIdx.x * NREGIONS_H * 16 + (index % nShmemH);  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_y = blockIdx.y * NREGIONS_V * 16 + (index / nShmemH);  //  = nShmemH = (6+2)*16
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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
    //     printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //             if ((j + 1) % 16 == 0) {
    //                 printf(" ");
    //             }
    //         }
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB;  // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA;  // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;
        // printf("0: rid: %i, wfx, wfy --> (%i,%i) -> global(%i,%i)\n", rid, workFragment_x, workFragment_y, globalFragment_x, globalFragment_y);

        if (globalFragment_x >= (n / 16) || globalFragment_y >= nWithHalo / 16) {
            continue;
        }
        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);
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

        wmma::store_matrix_sync(&shmem2[workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16], c_frag, nShmemH, wmma::mem_row_major);
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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem2[i * nShmemH + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    // // wmma::fill_fragment(c_frag, 0.0f);
    // __syncthreads();
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;
        // printf("0: rid: %i, wfx, wfy --> (%i,%i) -> global(%i,%i)\n", rid, workFragment_x, workFragment_y, globalFragment_x, globalFragment_y);

        if (globalFragment_x >= (n / 16) || globalFragment_y >= n / 16) {
            continue;
        }
        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);
        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        // Reducing horizontal neighbours
        wmma::load_matrix_sync(b_frag, &shmem2[workFragment_y * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
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

        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 1) * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
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
        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 2) * nShmemH * 16 + (workFragment_x + 1) * 16], nShmemH);
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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nShmemV; i++) {
    //         for (int j = 0; j < nShmemH; j++) {
    //             printf("%.0f ", __half2float(shmem[i * nShmemH + j]));
    //             if ((j + 1) % 16 == 0) {
    //                 printf(" ");
    //             }
    //         }
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        //.printf("%i\n", tid);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_x = blockIdx.x * NREGIONS_H * 16 + (index % (NREGIONS_H * 16));  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t dataCoord_y = blockIdx.y * NREGIONS_V * 16 + (index / (NREGIONS_H * 16));  //  = nShmemH = (6+2)*16
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

            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            // pDataOut[dindex] =val;// __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
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
    const uint32_t nFragmentsH = NREGIONS_H + 2;  // ⚠️ The code relies on this being 8 !!
    const uint32_t nFragmentsV = NREGIONS_V + 2;

    // a total of nFragmentsH * nFragmentsV fragments of 16x16
    const uint32_t nShmemH = nFragmentsH * 16;
    const uint32_t nShmemV = nFragmentsV * 16;
    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;
    FTYPE* shmem2 = (FTYPE*)((nShmemH * nShmemV) * sizeof(FTYPE) + totalshmem);

    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t wid = tid / 32;

    int i;
    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
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
        // uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        // uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // // for (char fragRow = 0; i < 8; i += 1) {
        // uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        // uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));
        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y) * 256 * nWithHalo / 16 + (globalFragment_x) * 256 + tid % 256;
        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        // size_t dindex = (globalFragment_y)*256 * nWithHalo / 16 + globalFragment_x * 256 + tid % 256;

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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
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

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB;  // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA;  // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    // int tts = 6;

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= nWithHalo / 16) {
            continue;
        }
        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }

        //  printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);
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
        wmma::store_matrix_sync(&shmem2[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
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
    // // wmma::fill_fragment(c_frag, 0.0f);
    // __syncthreads();
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= n / 16) {
            continue;
        }

        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);

        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        // Reducing horizontal neighbours   workFragment_y * nFragmentsH * 256 + workFragment_x * 256
        wmma::load_matrix_sync(b_frag, &shmem2[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
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
        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
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

        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 2) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
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

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nFragmentsH * nFragmentsV; i++) {
    //         uint32_t fid = i;
    //         uint32_t fx = fid % nFragmentsH;
    //         uint32_t fy = fid / nFragmentsH;
    //         printf("%u, %u\n", fx, fy);
    //         for (int ei = 0; ei < 16; ei++) {
    //             for (int ej = 0; ej < 16; ej++) {
    //                 printf("%.0f ", __half2float(shmem[fy * 256 * nFragmentsH + fx * 256 + ei * 16 + ej]));
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        // printf("%i\n", index);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1

        uint32_t fid = index / 256;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y) * 256 * (nWithHalo / 16) + (globalFragment_x) * 256 + index % 256;
        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     //     continue;
        //     printf("%i -- (%i,%i) = (%i, %i) -> %llu\n", index, regionCoord_x, regionCoord_y, globalFragment_x, globalFragment_y, dindex);

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is

        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {
            size_t ind = (fy + 1) * 256 * nFragmentsH + (fx + 1) * 256 + index % 256;
            uint32_t val = __half2uint_rn(shmem[ind]);
            // uint32_t val = __half2uint_rn(shmem[index + 16*nShmemH+256]);
            float val2 = __half2float(pDataIn[dindex]);
            // if (blockIdx.x == 1 && blockIdx.y == 0 && index % 256 == 0)

            //     printf("%llu -- (%i,%i) = (%i, %i) -> %llu\n", ind, fx, fy, globalFragment_x, globalFragment_y, dindex);

            // shmem[index] = pDataIn[dindex];
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            // pDataOut[dindex] = val;//__uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
    }
}

// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorCoalescedV2GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t nFragmentsH = NREGIONS_H + 2;  // ⚠️ The code relies on this being 8 !!
    const uint32_t nFragmentsV = NREGIONS_V + 2;

    // a total of nFragmentsH * nFragmentsV fragments of 16x16
    const uint32_t nShmemH = nFragmentsH * 16;
    const uint32_t nShmemV = nFragmentsV * 16;
    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;
    FTYPE* shmem2 = (FTYPE*)((nShmemH * nShmemV) * sizeof(FTYPE) + totalshmem);

    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t wid = tid / 32;

    int i;
    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    // Copying the corresponding global memory region to shared

    for (uint32_t index = tid; index < nShmemH * nShmemV; index += BSIZE3DX * BSIZE3DY) {
        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));

        size_t dindex = (globalFragment_y) * 256 * nWithHalo / 16 + (globalFragment_x) * 256 + tid % 256;
        if (globalFragment_x < nWithHalo / 16 && globalFragment_y < nWithHalo / 16) {
            shmem[index] = pDataIn[dindex];
        }
    }
    __syncthreads();

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
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

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB;  // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA;  // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    // int tts = 6;
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= nWithHalo / 16) {
            continue;
        }

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + workFragment_x * 256], 16);
        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_1_asB, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, a_frag, T_1_asB, c_frag);

        wmma::load_matrix_sync(a_frag, &shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 2) * 256], 16);
        wmma::load_matrix_sync(T_2_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_2_asB, c_frag);

        wmma::store_matrix_sync(&shmem2[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);

        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= n / 16) {
            continue;
        }
        wmma::load_matrix_sync(b_frag, &shmem2[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem2[(workFragment_y + 2) * nFragmentsH * 256 + (workFragment_x + 1) * 256], 16);
        wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);

        wmma::store_matrix_sync(&shmem[(workFragment_y + 1) * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        uint32_t fid = index / 256;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y) * 256 * (nWithHalo / 16) + (globalFragment_x) * 256 + index % 256;

        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {
            size_t ind = (fy + 1) * 256 * nFragmentsH + (fx + 1) * 256 + index % 256;
            uint32_t val = __half2uint_rn(shmem[ind]);
            // uint32_t val = __half2uint_rn(shmem[index + 16*nShmemH+256]);
            float val2 = __half2float(pDataIn[dindex]);

            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            // pDataOut[dindex] = val;// __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
    }
}

static inline bool is_aligned(const void* pointer, size_t byte_count) {
    return (uintptr_t)pointer % byte_count == 0;
}

__global__ void TensorCoalescedV3GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    const uint32_t nFragmentsH = NREGIONS_H + 2;

    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;

    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t wid = tid / 32;

    int i;
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag3;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB;  // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA;  // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;

    const uint32_t n16 = n >> 4;
    const uint32_t nWithHalo16 = nWithHalo >> 4;
#pragma unroll

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;
        // for (char fragRow = 0; i < 8; i += 1) {
        const uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        const uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (!(globalFragment_x < n16 && globalFragment_y < nWithHalo16)) {
            continue;
        }

        size_t globalFragment_p = (globalFragment_y * nWithHalo16 + globalFragment_x) << 8;

        wmma::load_matrix_sync(a_frag, &pDataIn[globalFragment_p], 16);
        wmma::load_matrix_sync(a_frag2, &pDataIn[globalFragment_p + 256], 16);
        wmma::load_matrix_sync(a_frag3, &pDataIn[globalFragment_p + 512], 16);

        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::load_matrix_sync(T_2_asB, &shmem_tridiag[256], 16);
        wmma::load_matrix_sync(T_1_asB, shmem_tridiag, 16);

        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);
        wmma::mma_sync(c_frag, a_frag2, T_1_asB, c_frag);
        wmma::mma_sync(c_frag, a_frag3, T_2_asB, c_frag);

        wmma::store_matrix_sync(&shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
#pragma unroll

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint32_t workFragment_x = rid % NREGIONS_H;
        const uint32_t workFragment_y = rid / NREGIONS_H;
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n16 || globalFragment_y >= n16) {
            continue;
        }
        size_t globalFragment_p = (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 256;
        wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p], 16);
        wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p + nFragmentsH * 256], 16);
        wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p + nFragmentsH * 512], 16);
        wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);

        wmma::store_matrix_sync(&pDataOut[((globalFragment_y + 1) * nWithHalo16 + (globalFragment_x + 1)) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
#pragma unroll

    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        uint32_t fid = index >> 8;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y * nWithHalo16 + globalFragment_x) * 256 + (index & 255);
        if (globalFragment_x < (nWithHalo16)-1 && globalFragment_y < (nWithHalo16)-1) {
            uint32_t val = __half2uint_rn(pDataOut[dindex]);
            float val2 = __half2float(pDataIn[dindex]);
            // pDataOut[dindex] = val;//__uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
    }
}

__global__ void TensorCoalescedV4GoLStep_Step1(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    // a total of nFragmentsH * nFragmentsV fragments of 16x16

    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t wid = tid / 32;

    int i;
    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();
    //__syncthreads();

    // Copying the corresponding global memory region to shared

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
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
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB;  // Col major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    // int tts = 6;

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= nWithHalo / 16) {
            continue;
        }

        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }

        //  printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);
        //  wmma::fill_fragment(c_frag, 0.0f);

        // Reducing horizontal neighbours
        wmma::load_matrix_sync(a_frag, &pDataIn[globalFragment_y * nWithHalo / 16 * 256 + globalFragment_x * 256], 16);
        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);

        // __syncthreads();
        // if (rid == tts) {
        //     wmma::store_matrix_sync(aux, c_frag, 16, wmma::mem_row_major);
        // }
        // __syncthreads();
        // // printf("LEFT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nFragmentsH * 256 + workFragment_x * 256);
        // if (blockIdx.x == 0 && blockIdx.y == 0 && tid % 32 == 0 && globalFragment_y == 1 && globalFragment_x == 1) {
        //     printf("\n");
        //     printf("LEFT wid: %i - rid: %i - (%i, %i)=%i\n", wid, rid, workFragment_x, workFragment_y, workFragment_y * nFragmentsH * 256 + workFragment_x * 256);

        //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
        //     for (int i = 0; i < 16; i++) {
        //         for (int j = 0; j < 16; j++) {
        //             printf("%.0f ", __half2float(pDataIn[globalFragment_y * nWithHalo / 16 * 256 + (globalFragment_x)*256 + i * 16 + j]));
        //             //  printf("%.0f ", __half2float(shmem_tridiag[i * 16 + j + 256]));
        //             // printf("%.0f ", __half2float(shmem[workFragment_y * nShmemH * 16 + i * nShmemH + (workFragment_x + 0) * 16 + j]));
        //         }
        //         printf("\n");
        //     }
        // }
        // __syncthreads();
        // wmma::fill_fragment(c_frag, 0.0f);

        wmma::load_matrix_sync(a_frag, &pDataIn[globalFragment_y * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], 16);
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
        wmma::load_matrix_sync(a_frag, &pDataIn[globalFragment_y * nWithHalo / 16 * 256 + (globalFragment_x + 2) * 256], 16);
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
        wmma::store_matrix_sync(&pDataOut[globalFragment_y * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
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
}

__global__ void TensorCoalescedV4GoLStep_Step2(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // Here goes the tridiagonal matrices. Only 2 will be generated as T_0 = T_2^t
    __shared__ FTYPE shmem_tridiag[16 * 16 * 2];

    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t wid = tid / 32;
    uint32_t val2s[1 + NREGIONS_H * 16 * NREGIONS_V * 16 / (BSIZE3DX * BSIZE3DY)];
    // uint32_t val2s[NREGIONS_H * 16 * NREGIONS_V * 16 / (BSIZE3DX * BSIZE3DY)];
    //  printf("%i\n", NREGIONS_H * 16 * NREGIONS_V * 16 / (BSIZE3DX * BSIZE3DY));
    //   Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    //   printf("%.f\n", __half2float())
    int i;
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }
    for (uint32_t val2id = 0; val2id < NREGIONS_H * 16 * NREGIONS_V * 16 / (BSIZE3DX * BSIZE3DY); val2id += 1) {
        const int t = tid + BSIZE3DX * BSIZE3DY * val2id;
        uint32_t fid = t / 256;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y) * 256 * (nWithHalo / 16) + (globalFragment_x) * 256 + t % 256;
        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {
            val2s[val2id] = __half2uint_rn(pDataOut[dindex]);
            // printf("%u -> %i\n", val2id, val2s[val2id]);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA;  // Row major
    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n / 16 || globalFragment_y >= n / 16) {
            continue;
        }

        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     printf("0: rid: %i, wfx, wfy --> (%i,%i)\n", rid, workFragment_x, workFragment_y);

        // if (workFragment_x >= nWithHalo / 16 || workFragment_y >= nWithHalo / 16) {
        //     continue;
        // }
        // Reducing horizontal neighbours   workFragment_y * nFragmentsH * 256 + workFragment_x * 256
        wmma::load_matrix_sync(b_frag, &pDataIn[globalFragment_y * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], 16);
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
        wmma::load_matrix_sync(b_frag, &pDataIn[(globalFragment_y + 1) * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], 16);
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

        wmma::load_matrix_sync(b_frag, &pDataIn[(globalFragment_y + 2) * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], 16);
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

        wmma::store_matrix_sync(&pDataOut[(globalFragment_y + 1) * nWithHalo / 16 * 256 + (globalFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
    // __syncthreads();

    // if (blockIdx.x == 1 && blockIdx.y == 0 && tid == 0) {
    //     printf("\n");
    //     printf("\n");

    //     // printf("nShmemV: %i, nShmemH: %i\n", nShmemV, nShmemH);
    //     for (int i = 0; i < nFragmentsH * nFragmentsV; i++) {
    //         uint32_t fid = i;
    //         uint32_t fx = fid % nFragmentsH;
    //         uint32_t fy = fid / nFragmentsH;
    //         printf("%u, %u\n", fx, fy);
    //         for (int ei = 0; ei < 16; ei++) {
    //             for (int ej = 0; ej < 16; ej++) {
    //                 printf("%.0f ", __half2float(shmem[fy * 256 * nFragmentsH + fx * 256 + ei * 16 + ej]));
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    int c = 0;
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        // printf("%i\n", index);
        //  uint8_t dataCoord_x = blockIdx.x * blockDim.x + (index & 127); // ⚠️ bc of this hardcoded 127 !! nShmemH-1

        uint32_t fid = index / 256;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y) * 256 * (nWithHalo / 16) + (globalFragment_x) * 256 + index % 256;
        // if (blockIdx.x == 1 && blockIdx.y == 0)
        //     //     continue;
        //     printf("%i -- (%i,%i) = (%i, %i) -> %llu\n", index, regionCoord_x, regionCoord_y, globalFragment_x, globalFragment_y, dindex);

        // Ideally I could remove this check if (number regions*16)%n == 0
        // if (dindex < nWithHalo * nWithHalo) { //this works but data is repeated along the x axis when there is

        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {
            // size_t ind = (fy + 1) * 256 * nFragmentsH + (fx + 1) * 256 + index % 256;
            uint32_t val = __half2uint_rn(pDataOut[dindex]);
            // uint32_t val = __half2uint_rn(shmem[index + 16*nShmemH+256]);
            uint32_t val2 = (val2s[c]);
            // uint32_t val2 = val;//(val2s[c]);
            // printf("%i\n", c);

            // if (blockIdx.x == 1 && blockIdx.y == 0 && index % 256 == 0)

            //     printf("%llu -- (%i,%i) = (%i, %i) -> %llu\n", ind, fx, fy, globalFragment_x, globalFragment_y, dindex);

            // shmem[index] = pDataIn[dindex];
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            // pDataOut[dindex] = val;//__uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
        c += 1;
    }
}

__device__ __inline__ uint32_t addInt4(int i, char int4index, int* shmem) {
    int oldval = shmem[i / 8];
    int newval = ((32 + RADIUS - abs((i >> 5) - (i & 31))) >> 5);
    oldval = oldval | (newval << (int4index * 4));
    return oldval;
}

__device__ __inline__ uint32_t addInt4left(int i, char int4index, int* shmem) {
    int oldval = shmem[i / 8];
    int newval = (32 + (i & 31) - (i >> 5)) / (64 - RADIUS);
    oldval = oldval | (newval << (int4index * 4));
    return oldval;
}
__device__ __inline__ uint32_t addInt4right(int i, char int4index, int* shmem) {
    int oldval = shmem[i / 8];
    int newval = (24 + (32 - (i & 31)) + (i >> 5)) / (64 - RADIUS);
    oldval = oldval | (newval << (int4index * 4));
    return oldval;
}
__global__ void TensorCoalescedSubTypeGoLStep(int* pDataIn, size_t n, size_t nWithHalo, MTYPE* buffer) {
    const uint32_t nFragmentsV = NREGIONS_V + 2;
    const uint32_t nFragmentsH = NREGIONS_H + 2;

    extern __shared__ char totalshmem[];
    size_t regionsize = nFragmentsV * nFragmentsH * 32 * 32 * sizeof(int);
    int* shmem = (int*)totalshmem;
    int* shmemComp = (int*)&totalshmem[regionsize];
    int* shmem_tridiag = (int*)&totalshmem[regionsize + regionsize / 8];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int wid = tid / 32;
    const int wtid = tid & 31;

    int i;

    for (i = tid; i < 1024 / 8; i += BSIZE3DX * BSIZE3DY) {
        int val = 0;
        for (int j = 0; j < 8; j++) {
            int tridiag_index = i * 8 + j;
            int minival = ((32 + RADIUS - abs((tridiag_index >> 5) - (tridiag_index & 31))) >> 5);
            val = val | (minival << (j * 4));
        }
        shmem_tridiag[i] = val;
    }
#pragma unroll
    for (i = tid + 1024 / 8; i < 1280 / 8; i += BSIZE3DX * BSIZE3DY) {
        int val = 0;
        for (int j = 0; j < 8; j++) {
            int tridiag_index = tid * 8 + j;
            int minival = (32 + (tridiag_index & 31) - (tridiag_index >> 5)) / (64 - RADIUS);
            val = val | (minival << (j * 4));
        }
        shmem_tridiag[i] = val;
    }
#pragma unroll
    for (i = tid + 1280 / 8; i < 1280 / 8 + 256 / 8; i += BSIZE3DX * BSIZE3DY) {
        int val = 0;
        for (int j = 0; j < 8; j++) {
            int tridiag_index = tid * 8 + j;
            int minival = (24 + (32 - (tridiag_index & 31)) + (tridiag_index >> 5)) / (64 - RADIUS);
            val = val | (minival << (j * 4));
        }
        shmem_tridiag[i] = val;
    }
    // for (int ki=tid; ki<nFragmentsH*nFragmentsV*32*32; ki+=BSIZE3DX*BSIZE3DY){
    //     shmem[ki] = 0;
    // }
    // for (int ki=tid; ki<nFragmentsH*nFragmentsV*32*32/8; ki+=BSIZE3DX*BSIZE3DY){
    //     shmemComp[ki] = 0;
    // }

    // __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(i=0; i<(1024/8); i++){
    //         printf("%x, ", shmem_tridiag[i]);
    //         if(i%4 == 3){

    //             printf("\n");
    //         }
    //     }
    //                     printf("\n");

    // }

    // __syncthreads();
    // __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(i=(1024/8); i<256/8+(1024/8); i++){
    //         printf("%x, ", shmem_tridiag[i]);
    //         if(i%4 == 3){

    //             printf("\n");
    //         }
    //     }
    //             printf("\n");
    //             printf("\n");

    // }

    // __syncthreads();
    //  __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(i=256/8+(1024/8); i<512/8+(1024/8); i++){
    //         printf("%x, ", shmem_tridiag[i]);
    //         if(i%4 == 3){

    //             printf("\n");
    //         }
    //     }
    // }

    // __syncthreads();
    // __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(i=0; i<nWithHalo*nWithHalo/8; i++){
    //         printf("%x ", pDataIn[i]);
    //         if(i%(nWithHalo/8) == (nWithHalo/8)-1){

    //             printf("\n");
    //         }
    //         if(i%(nWithHalo) == (nWithHalo)-1){
    //             printf("\n");
    //         }
    //     }
    //             printf("\n");
    //             printf("\n");
    //             printf("\n");

    // }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 32, wmma::experimental::precision::u4, wmma::row_major> a_frag0;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, wmma::experimental::precision::u4, wmma::col_major> b_frag0;

    wmma::fragment<wmma::matrix_a, 8, 8, 32, wmma::experimental::precision::u4, wmma::row_major> a_frag1;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, wmma::experimental::precision::u4, wmma::col_major> b_frag1;

    wmma::fill_fragment(c_frag, 0);

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;

    const uint32_t n32 = n >> 5;
    const uint32_t nWithHalo32 = nWithHalo >> 5;

#pragma unroll
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;
        // for (char fragRow = 0; i < 8; i += 1) {
        const uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        const uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (!(globalFragment_x < n32 && globalFragment_y < nWithHalo32)) {
            continue;
        }

        size_t globalFragment_p = (globalFragment_y * nWithHalo32 + globalFragment_x) << (7);

        for (char minifrag_i = 0; minifrag_i < 4; minifrag_i++) {
            for (char minifrag_j = 0; minifrag_j < 4; minifrag_j++) {
                if (minifrag_j == 0) {
                    wmma::load_matrix_sync(a_frag0, &pDataIn[globalFragment_p + (minifrag_i * 256) / 8], 32);
                    wmma::load_matrix_sync(b_frag0, &shmem_tridiag[(4 * 256 / 8)], 32);
                    wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);
                }
                wmma::load_matrix_sync(a_frag0, &pDataIn[globalFragment_p + (1024 / 8) + (minifrag_i * 256) / 8], 32);
                wmma::load_matrix_sync(b_frag0, &shmem_tridiag[(minifrag_j * 256 / 8)], 32);
                wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);

                if (minifrag_j == 3) {
                    wmma::load_matrix_sync(a_frag0, &pDataIn[globalFragment_p + 2 * (1024 / 8) + (minifrag_i * 256) / 8], 32);
                    wmma::load_matrix_sync(b_frag0, &shmem_tridiag[(5 * 256 / 8)], 32);
                    wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);
                }
                // printf("%i\n", (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 1024+ minifrag_i*256 + minifrag_j*8);
                wmma::store_matrix_sync(&shmem[(workFragment_y * nFragmentsH + (workFragment_x + 1)) * 1024 + minifrag_j * 256 + minifrag_i * 8], c_frag, 32, wmma::mem_col_major);
                wmma::fill_fragment(c_frag, 0.0f);
            }
        }
    }

    __syncthreads();
    for (int i = tid; i < (nFragmentsV * nFragmentsH) * 32 * 32 / 8; i += BSIZE3DX * BSIZE3DY) {
        int val = 0;
        for (int j = 0; j < 8; j++) {
            int tridiag_index = i * 8 + j;
            int minival = shmem[tridiag_index];
            val = val | (minival << (j * 4));
        }
        shmemComp[i] = val;
    }
    __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(int ii=0; ii<nFragmentsV*32; ii++){
    //         for (int j=0; j< nFragmentsH*32; j++){
    //             printf("%i ", shmem[ii*nFragmentsH*32 + j]);
    //             if ((ii*nFragmentsH*32 + j )%1024 == 1023){
    //                 printf("\n");
    //                 for (int jj=0;jj<nFragmentsH*32-j; j++){
    //                     printf("  ");
    //                 }
    //             }
    //         }
    //         printf("\n");
    //     }

    // }
    // __syncthreads();
    // __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(int ii=0; ii<nFragmentsV*32; ii++){
    //         for (int j=0; j< nFragmentsH*32/8; j++){
    //             printf("%x ", shmemComp[ii*nFragmentsH*32/8 + j]);
    //             if ((ii*nFragmentsH*32 + j )%(nWithHalo) == nWithHalo-1){
    //                 printf("\n");

    //             }
    //             // if((ii*nFragmentsH*32 + j )%(nWithHalo) == (nWithHalo)-1){
    //             //     printf("\n");
    //             // }
    //         }
    //         printf("\n");
    //     }

    // }
    // __syncthreads();

#pragma unroll
    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;
        // for (char fragRow = 0; i < 8; i += 1) {
        const uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        const uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (!(globalFragment_x < n32 && globalFragment_y < nWithHalo32)) {
            continue;
        }

        size_t globalFragment_p = (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 1024 / 8;

        for (char minifrag_i = 0; minifrag_i < 4; minifrag_i++) {
            for (char minifrag_j = 0; minifrag_j < 4; minifrag_j++) {
                if (minifrag_i == 0) {
                    wmma::load_matrix_sync(a_frag0, &shmem_tridiag[(4 * 256 / 8)], 32);
                    wmma::load_matrix_sync(b_frag0, &shmemComp[(workFragment_y * nFragmentsH + (workFragment_x + 1)) * 1024 / 8 + (minifrag_j * 256) / 8], 32);
                    wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);
                }
                wmma::load_matrix_sync(a_frag0, &shmem_tridiag[(minifrag_i * 256 / 8)], 32);
                wmma::load_matrix_sync(b_frag0, &shmemComp[((workFragment_y + 1) * nFragmentsH + (workFragment_x + 1)) * 1024 / 8 + (minifrag_j) * 256 / 8], 32);
                wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);

                if (minifrag_i == 3) {
                    wmma::load_matrix_sync(a_frag0, &shmem_tridiag[(5 * 256 / 8)], 32);
                    wmma::load_matrix_sync(b_frag0, &shmemComp[((workFragment_y + 2) * nFragmentsH + (workFragment_x + 1)) * 1024 / 8 + (minifrag_j * 256) / 8], 32);
                    wmma::mma_sync(c_frag, a_frag0, b_frag0, c_frag);
                }
                // printf("%i\n", (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 1024+ minifrag_i*256 + minifrag_j*8);
                wmma::store_matrix_sync(&shmem[((workFragment_y + 1) * nFragmentsH + (workFragment_x + 1)) * 1024 + minifrag_i * 256 + minifrag_j * 8], c_frag, 32, wmma::mem_row_major);
                wmma::fill_fragment(c_frag, 0.0f);
            }
        }
    }

    //     __syncthreads();
    // if (tid ==0 && blockIdx.x ==0 && blockIdx.y == 0){
    //     for(int ii=0; ii<nFragmentsV*32; ii++){
    //         for (int j=0; j< nFragmentsH*32; j++){
    //             printf("%i ", shmem[ii*nFragmentsH*32 + j]);
    //         }
    //         printf("\n");
    //     }

    // }
    // __syncthreads();

    // for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
    //     const uint32_t workFragment_x = rid % NREGIONS_H;
    //     const uint32_t workFragment_y = rid / NREGIONS_H;
    //     const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
    //     const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16

    //     uint32_t globalFragment_x = regionCoord_x + workFragment_x;
    //     uint32_t globalFragment_y = regionCoord_y + workFragment_y;

    //     if (globalFragment_x >= n16 || globalFragment_y >= n16) {
    //         continue;
    //     }
    //     size_t globalFragment_p = (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 256;
    //     wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p], 16);
    //     wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
    //     wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);

    //     wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p + nFragmentsH * 256], 16);
    //     wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
    //     wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);

    //     wmma::load_matrix_sync(b_frag, &shmem[globalFragment_p + nFragmentsH * 512], 16);
    //     wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
    //     wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);

    //     wmma::store_matrix_sync(&pDataOut[((globalFragment_y + 1) * nWithHalo16 + (globalFragment_x + 1)) * 256], c_frag, 16, wmma::mem_row_major);
    //     wmma::fill_fragment(c_frag, 0.0f);
    // }

    __syncthreads();

#pragma unroll
    for (uint32_t index = tid; index < NREGIONS_H * NREGIONS_V * 32 * 32; index += BSIZE3DX * BSIZE3DY) {
        uint32_t fragId = index >> 10;
        uint32_t fx = fragId % NREGIONS_H;
        uint32_t fy = fragId / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y * nWithHalo32 + globalFragment_x) * 1024 + (index & 1023);
        size_t shindex = (fy + 1) * nFragmentsH * 1024 + (fx + 1) * 1024 + (index & 1023);
        if (globalFragment_x < (nWithHalo32 - 1) && globalFragment_y < (nWithHalo32 - 1)) {
            uint32_t val = shmem[shindex];
            uint32_t i = index % 8;
            uint32_t val2 = (pDataIn[dindex / 8] >> (i * 4)) & 0b1111;
            // printf("%u\n", pDataIn[dindex/8]);
            // buffer[dindex] = val;//__uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
            buffer[dindex] = (val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
    }
}

__global__ void convertFp32ToFp16(FTYPE* out, int* in, int nWithHalo) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo) {
        out[tx + ty * (size_t)nWithHalo] = __uint2half_rn(in[tx + ty * (size_t)nWithHalo]);
    }
}
__global__ void convertFp16ToFp32(int* out, FTYPE* in, int nWithHalo) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo) {
        out[tx + ty * (size_t)nWithHalo] = __half2uint_rn(in[tx + ty * (size_t)nWithHalo]);
    }
}

__global__ void convertFp32ToFp16AndDoChangeLayout(FTYPE* out, int* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo) {
        out[bid * 256 + tid] = __uint2half_rn(in[ty * nWithHalo + tx]);
    }
}
__global__ void convertFp16ToFp32AndUndoChangeLayout(int* out, FTYPE* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo) {
        out[ty * nWithHalo + tx] = __half2uint_rn(in[bid * 256 + tid]);
    }
}

__global__ void convertUInt32ToUInt4AndDoChangeLayout(int* out, MTYPE* in, size_t nWithHalo) {
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (tx < nWithHalo && ty < nWithHalo) {
        int val = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            val |= (in[ty * nWithHalo + (tx) * 8 + i] & 0b1111) << (i * 4);
        }
        out[bid * 1024 / 8 + tid] = val;
    }
}
__global__ void convertUInt4ToUInt32AndUndoChangeLayout(MTYPE* out, int* in, size_t nWithHalo) {
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (tx < nWithHalo && ty < nWithHalo) {
        int val = in[(bid * 1024 / 8 + tid)];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            out[ty * nWithHalo + (tx) * 8 + i] = (val >> (i * 4)) & 0b1111;
        }
    }
}
__global__ void UndoChangeLayout(MTYPE* out, MTYPE* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);

    if (tx < nWithHalo && ty < nWithHalo) {
        out[ty * nWithHalo + tx] = in[bid * 1024 + tid];
    }
}

__global__ void onlyConvertUInt32ToUInt4(int* out, MTYPE* in, size_t nWithHalo) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nWithHalo * nWithHalo / 8) {
        int val = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            val |= (in[tid * 8 + i] & 0b1111) << (i * 4);
        }
        out[tid] = val;
    }
}

__global__ void convertInt32ToInt8AndDoChangeLayout(unsigned char* out, int* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo) {
        out[bid * 256 + tid] = (unsigned char)(in[ty * nWithHalo + tx]);
    }
}
__global__ void convertInt8ToInt32AndUndoChangeLayout(int* out, unsigned char* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo) {
        out[ty * nWithHalo + tx] = (int)(in[bid * 256 + tid]);
    }
}

__global__ void TensorCoalescedInt8(unsigned char* pDataIn, unsigned char* pDataOut, size_t n, size_t nWithHalo) {
    const uint32_t nFragmentsH = NREGIONS_H + 2;

    extern __shared__ char totalshmem[];
    int* shmem = (int*)totalshmem;
    unsigned char* shmem_char = (unsigned char*)&totalshmem[(NREGIONS_H + 2) * (NREGIONS_V + 2) * 16 * 16 * 4];

    __shared__ unsigned char shmem_tridiag[16 * 16 * 2];

    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t wid = tid / 32;

    int i;
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (16 + RADIUS - abs((i >> 4) - (i & 15))) >> 4;  // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i & 15) + (i >> 4)) / (32 - RADIUS);  //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> a_frag3;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> T_0_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::row_major> T_1_asB;  // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, unsigned char, wmma::col_major> T_2_asB;  // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::col_major> T_0_asA;  // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> T_1_asA;  // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, unsigned char, wmma::row_major> T_2_asA;  // Row major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;

    const uint32_t n16 = n >> 4;
    const uint32_t nWithHalo16 = nWithHalo >> 4;
#pragma unroll

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {
        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;
        // for (char fragRow = 0; i < 8; i += 1) {
        const uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        const uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (!(globalFragment_x < n16 && globalFragment_y < nWithHalo16)) {
            continue;
        }

        size_t globalFragment_p = (globalFragment_y * nWithHalo16 + globalFragment_x) << 8;

        wmma::load_matrix_sync(a_frag, &pDataIn[globalFragment_p], 16);
        wmma::load_matrix_sync(a_frag2, &pDataIn[globalFragment_p + 256], 16);
        wmma::load_matrix_sync(a_frag3, &pDataIn[globalFragment_p + 512], 16);

        wmma::load_matrix_sync(T_0_asB, &shmem_tridiag[256], 16);
        wmma::load_matrix_sync(T_2_asB, &shmem_tridiag[256], 16);
        wmma::load_matrix_sync(T_1_asB, shmem_tridiag, 16);

        wmma::mma_sync(c_frag, a_frag, T_0_asB, c_frag);
        wmma::mma_sync(c_frag, a_frag2, T_1_asB, c_frag);
        wmma::mma_sync(c_frag, a_frag3, T_2_asB, c_frag);

        wmma::store_matrix_sync(&shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

#pragma unroll
    for (uint32_t i = tid; i < (NREGIONS_H + 2) * (NREGIONS_V + 2) * 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_char[i] = shmem[i];
    }
    __syncthreads();
#pragma unroll

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint32_t workFragment_x = rid % NREGIONS_H;
        const uint32_t workFragment_y = rid / NREGIONS_H;
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V;  //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n16 || globalFragment_y >= n16) {
            continue;
        }
        size_t globalFragment_p = (workFragment_y * nFragmentsH + (workFragment_x + 1)) * 256;
        wmma::load_matrix_sync(b_frag, &shmem_char[globalFragment_p], 16);
        wmma::load_matrix_sync(T_0_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_0_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem_char[globalFragment_p + nFragmentsH * 256], 16);
        wmma::load_matrix_sync(T_1_asA, shmem_tridiag, 16);
        wmma::mma_sync(c_frag, T_1_asA, b_frag, c_frag);

        wmma::load_matrix_sync(b_frag, &shmem_char[globalFragment_p + nFragmentsH * 512], 16);
        wmma::load_matrix_sync(T_2_asA, &shmem_tridiag[256], 16);
        wmma::mma_sync(c_frag, T_2_asA, b_frag, c_frag);

        wmma::store_matrix_sync(&shmem[((workFragment_y + 1) * nFragmentsH + (workFragment_x + 1)) * 256], c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

#pragma unroll
    for (uint32_t index = tid; index < NREGIONS_H * 16 * NREGIONS_V * 16; index += BSIZE3DX * BSIZE3DY) {
        uint32_t fid = index >> 8;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H;  // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V;  //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y * nWithHalo16 + globalFragment_x) * 256 + (index & 255);
        if (globalFragment_x < (nWithHalo16)-1 && globalFragment_y < (nWithHalo16)-1) {
            size_t ind = (fy + 1) * 256 * nFragmentsH + (fx + 1) * 256 + index % 256;
            pDataOut[dindex] = shmem[ind];

            unsigned char val = (pDataOut[dindex]);
            unsigned char val2 = (pDataIn[dindex]);
            // pDataOut[dindex] = val;//__uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
            // pDataOut[dindex] = (val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
        }
    }
}
__global__ void copyHorizontalHalo(MTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [1,dim]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
#pragma unroll

        for (int i = 0; i < RADIUS; i++) {
            // Copy first real row to bottom ghost row
            data[(nWithHalo * (n + RADIUS + i)) + (id + RADIUS)] = data[(nWithHalo * (RADIUS + i)) + id + RADIUS];
            // Copy last real row to top ghost row
            data[nWithHalo * i + id + RADIUS] = data[(nWithHalo) * (n + i) + id + RADIUS];
        }
    }
}

__global__ void copyVerticalHalo(MTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [0,dim+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < nWithHalo) {
#pragma unroll
        for (int i = 0; i < RADIUS; i++) {
            // Copy first real column to right most ghost column
            data[(id) * (nWithHalo) + (n + RADIUS + i)] = data[(id) * (nWithHalo) + (RADIUS + i)];
            // Copy last real column to left most ghost column
            data[(id) * (nWithHalo) + i] = data[(id) * (nWithHalo) + (n + i)];
        }
    }
}

__global__ void copyHorizontalHaloCoalescedVersion(FTYPE* data, size_t n, size_t nWithHalo) {
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (bid < n / 16) {
        data[(bid + 1) * 256 + tid] = data[(bid + 1 + nWithHalo / 16 * n / 16) * 256 + tid];
    } else if (bid < 2 * (n / 16)) {
        bid -= n / 16;
        data[(bid + 1 + nWithHalo / 16 * (nWithHalo / 16 - 1)) * 256 + tid] = data[(bid + 1 + nWithHalo / 16) * 256 + tid];
    }
}

__global__ void copyVerticalHaloCoalescedVersion(FTYPE* data, size_t n, size_t nWithHalo) {
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (bid < nWithHalo / 16) {
        data[(bid * (nWithHalo / 16) * 256) + tid] = data[(bid * (nWithHalo / 16) * 256) + (n / 16) * 256 + tid];
    } else if (bid < 2 * (nWithHalo / 16)) {
        bid -= nWithHalo / 16;
        // printf("ASD\n");
        data[(bid * (nWithHalo / 16) * 256) + (n / 16 + 1) * 256 + tid] = data[(bid * (nWithHalo / 16) * 256) + tid + 256];
    }
}
__global__ void copyHorizontalHaloHalf(FTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [1,dim]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
#pragma unroll

        for (int i = 0; i < RADIUS; i++) {
            // Copy first real row to bottom ghost row
            data[(nWithHalo * (n + RADIUS + i)) + (id + RADIUS)] = data[(nWithHalo * (RADIUS + i)) + id + RADIUS];
            // Copy last real row to top ghost row
            data[nWithHalo * i + id + RADIUS] = data[(nWithHalo) * (n + i) + id + RADIUS];
        }
    }
}

__global__ void copyVerticalHaloHalf(FTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [0,dim+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < nWithHalo) {
#pragma unroll
        for (int i = 0; i < RADIUS; i++) {
            // Copy first real column to right most ghost column
            data[(id) * (nWithHalo) + (n + RADIUS + i)] = data[(id) * (nWithHalo) + (RADIUS + i)];
            // Copy last real column to left most ghost column
            data[(id) * (nWithHalo) + i] = data[(id) * (nWithHalo) + (n + i)];
        }
    }
}

__global__ void copyHorizontalHaloTensor(FTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [1,dim]
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < n) {
#pragma unroll

        for (int h = 0; h < RADIUS; h++) {
            // Copy last real row to top ghost row
            data[(nWithHalo * (h + 16 - RADIUS)) + j + 16] = data[(nWithHalo) * (n + (h + 16 - RADIUS)) + j + 16];
            // Copy first real row to bottom ghost row
            data[(nWithHalo * (n + h + 16)) + (j + 16)] = data[(nWithHalo * (16 + h)) + j + 16];
        }
    }
}

__global__ void copyVerticalHaloTensor(FTYPE* data, size_t n, size_t nWithHalo) {
    // We want id ∈ [0,dim+1]
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nWithHalo) {
#pragma unroll
        for (int h = 0; h < RADIUS; h++) {
            // Copy first real column to right most ghost column
            data[(i) * (nWithHalo) + (n + 16 + h)] = data[(i) * (nWithHalo) + (16 + h)];
            // Copy last real column to left most ghost column
            data[(i) * (nWithHalo) + (h + 16 - RADIUS)] = data[(i) * (nWithHalo) + (n + (h + 16 - RADIUS))];
        }
    }
}

__global__ void copyFromMTYPEAndCast(MTYPE* from, int* to, size_t nWithHalo) {
    size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tid = tid_y * blockDim.x * gridDim.x + tid_x;
    for (size_t index = tid; index < nWithHalo * nWithHalo; index += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
        to[index] = (int)from[index];
    }
}
__global__ void copyToMTYPEAndCast(int* from, MTYPE* to, size_t nWithHalo) {
    size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tid = tid_y * blockDim.x * gridDim.x + tid_x;
    for (size_t index = tid; index < nWithHalo * nWithHalo; index += blockDim.x * blockDim.y * gridDim.x * gridDim.y) {
        to[index] = (MTYPE)from[index];
    }
}

///////////////////////////////////////////////////////////
#define sh_row threadIdx.y
#define sh_col (threadIdx.x * cellsPerThread)
#define x2 (x * cellsPerThread)
#define sh_size_x (blockDim.x * cellsPerThread)
__forceinline__ __device__ int count_neighs(int my_id, int size_i, MTYPE* lattice, int neighs, int halo);

__global__ void moveKernel(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int cellsPerThread, int neighs, int halo) {
    int count = 0, k;
    int x = (blockDim.x - halo) * blockIdx.x + threadIdx.x;
    int y = (blockDim.y - halo) * blockIdx.y + threadIdx.y;
    int my_sh_id;
    size_t my_id;

    extern __shared__ MTYPE sh_lattice[];

    for (k = 0; k < cellsPerThread; k++) {
        my_sh_id = sh_row * sh_size_x + sh_col + k;
        my_id = y * (size_t)(size_i + halo) + x2 + k;

        if (y < size_i + halo && (x2 + k) < size_j + halo) {
            sh_lattice[my_sh_id] = d_lattice[my_id];
        }
    }
    __syncthreads();

    for (k = 0; k < cellsPerThread; k++) {
        my_sh_id = sh_row * sh_size_x + sh_col + k;
        my_id = y * (size_t)(size_i + halo) + x2 + k;
        MTYPE c = sh_lattice[my_sh_id];
        if (y < size_i + neighs && (x2 + k) < size_j + neighs && sh_row >= neighs && sh_row < blockDim.y - neighs && (sh_col + k) >= neighs && (sh_col + k) < (blockDim.x * cellsPerThread) - neighs) {
            count = count_neighs(my_sh_id, sh_size_x - halo, sh_lattice, neighs, halo);  // decrease sh_size_x by 2 to use the same count_neighs function than the rest of the implementations
            d_lattice_new[my_id] = c * h(count, SMIN, SMAX) + (1 - c) * h(count, BMIN, BMAX);
            // check_rules(my_id, count, d_lattice, d_lattice_new);
        }
    }
}
#define NEIGHS1
__forceinline__ __device__ int count_neighs(int my_id, int size_i, MTYPE* lattice, int neighs, int halo) {
    int size = size_i + halo;
    int count = 0;

#if RADIUS > 5
    for (int i = -RADIUS; i <= RADIUS; i++) {
#pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++) {
            count += lattice[my_id + i * size + j];
        }
    }
    return count;
#endif
#if RADIUS > 0
    count = lattice[my_id - size - 1];
    count += lattice[my_id - size];
    count += lattice[my_id - size + 1];
    count += lattice[my_id - 1];
    count += lattice[my_id + 1];
    count += lattice[my_id + size - 1];
    count += lattice[my_id + size];
    count += lattice[my_id + size + 1];
#endif

#if RADIUS > 1
    int size2 = 2 * size;

    count += lattice[my_id - size2 - 2];
    count += lattice[my_id - size2 - 1];
    count += lattice[my_id - size2];
    count += lattice[my_id - size2 + 1];
    count += lattice[my_id - size2 + 2];

    count += lattice[my_id - size - 2];
    count += lattice[my_id - size + 2];

    count += lattice[my_id - 2];
    count += lattice[my_id + 2];

    count += lattice[my_id + size - 2];
    count += lattice[my_id + size + 2];

    count += lattice[my_id + size2 - 2];
    count += lattice[my_id + size2 - 1];
    count += lattice[my_id + size2];
    count += lattice[my_id + size2 + 1];
    count += lattice[my_id + size2 + 2];
#endif

#if RADIUS > 2
    int size3 = 3 * size;
    count += lattice[my_id - size3 - 3];
    count += lattice[my_id - size3 - 2];
    count += lattice[my_id - size3 - 1];
    count += lattice[my_id - size3];
    count += lattice[my_id - size3 + 1];
    count += lattice[my_id - size3 + 2];
    count += lattice[my_id - size3 + 3];

    count += lattice[my_id - size2 - 3];
    count += lattice[my_id - size2 + 3];

    count += lattice[my_id - size - 3];
    count += lattice[my_id - size + 3];

    count += lattice[my_id - 3];
    count += lattice[my_id + 3];

    count += lattice[my_id + size - 3];
    count += lattice[my_id + size + 3];

    count += lattice[my_id + size2 - 3];
    count += lattice[my_id + size2 + 3];

    count += lattice[my_id + size3 - 3];
    count += lattice[my_id + size3 - 2];
    count += lattice[my_id + size3 - 1];
    count += lattice[my_id + size3];
    count += lattice[my_id + size3 + 1];
    count += lattice[my_id + size3 + 2];
    count += lattice[my_id + size3 + 3];
#endif

#if RADIUS > 3
    int size4 = 4 * size;

    count += lattice[my_id - size4 - 4];
    count += lattice[my_id - size4 - 3];
    count += lattice[my_id - size4 - 2];
    count += lattice[my_id - size4 - 1];
    count += lattice[my_id - size4];
    count += lattice[my_id - size4 + 1];
    count += lattice[my_id - size4 + 2];
    count += lattice[my_id - size4 + 3];
    count += lattice[my_id - size4 + 4];

    count += lattice[my_id - size3 - 4];
    count += lattice[my_id - size3 + 4];

    count += lattice[my_id - size2 - 4];
    count += lattice[my_id - size2 + 4];

    count += lattice[my_id - size - 4];
    count += lattice[my_id - size + 4];

    count += lattice[my_id - 4];
    count += lattice[my_id + 4];

    count += lattice[my_id + size - 4];
    count += lattice[my_id + size + 4];

    count += lattice[my_id + size2 - 4];
    count += lattice[my_id + size2 + 4];

    count += lattice[my_id + size3 - 4];
    count += lattice[my_id + size3 + 4];

    count += lattice[my_id + size4 - 4];
    count += lattice[my_id + size4 - 3];
    count += lattice[my_id + size4 - 2];
    count += lattice[my_id + size4 - 1];
    count += lattice[my_id + size4];
    count += lattice[my_id + size4 + 1];
    count += lattice[my_id + size4 + 2];
    count += lattice[my_id + size4 + 3];
    count += lattice[my_id + size4 + 4];
#endif

#if RADIUS > 4
    int size5 = 5 * size;

    count += lattice[my_id - size5 - 5];
    count += lattice[my_id - size5 - 4];
    count += lattice[my_id - size5 - 3];
    count += lattice[my_id - size5 - 2];
    count += lattice[my_id - size5 - 1];
    count += lattice[my_id - size5];
    count += lattice[my_id - size5 + 1];
    count += lattice[my_id - size5 + 2];
    count += lattice[my_id - size5 + 3];
    count += lattice[my_id - size5 + 4];
    count += lattice[my_id - size5 + 5];

    count += lattice[my_id - size4 - 5];
    count += lattice[my_id - size4 + 5];

    count += lattice[my_id - size3 - 5];
    count += lattice[my_id - size3 + 5];

    count += lattice[my_id - size2 - 5];
    count += lattice[my_id - size2 + 5];

    count += lattice[my_id - size - 5];
    count += lattice[my_id - size + 5];

    count += lattice[my_id - 5];
    count += lattice[my_id + 5];

    count += lattice[my_id + size - 5];
    count += lattice[my_id + size + 5];

    count += lattice[my_id + size2 - 5];
    count += lattice[my_id + size2 + 5];

    count += lattice[my_id + size3 - 5];
    count += lattice[my_id + size3 + 5];

    count += lattice[my_id + size4 - 5];
    count += lattice[my_id + size4 + 5];

    count += lattice[my_id + size5 - 5];
    count += lattice[my_id + size5 - 4];
    count += lattice[my_id + size5 - 3];
    count += lattice[my_id + size5 - 2];
    count += lattice[my_id + size5 - 1];
    count += lattice[my_id + size5];
    count += lattice[my_id + size5 + 1];
    count += lattice[my_id + size5 + 2];
    count += lattice[my_id + size5 + 3];
    count += lattice[my_id + size5 + 4];
    count += lattice[my_id + size5 + 5];
#endif

    return count;
}

__global__ void copy_Rows(int size_i, MTYPE* d_lattice, int neighs, int halo) {
    size_t my_id = (size_t)blockDim.x * blockIdx.x + threadIdx.x + neighs;
    int i = 0;
    size_t size = size_i + halo;

    if (my_id < (size_i + neighs)) {
        for (i = 0; i < neighs; i++) {
            d_lattice[size * (size_i + (i + neighs)) + my_id] = d_lattice[(i + neighs) * size + my_id];  // copia primeras filas en ultimas
            d_lattice[i * size + my_id] = d_lattice[size * (size_i + i) + my_id];                        // copia ultimas filas en primeras
        }
    }
}

__global__ void copy_Cols(int size_i, MTYPE* d_lattice, int neighs, int halo) {
    size_t my_id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    int i = 0;
    // Al haber copiado la primer fila en la ultima columna, se puede directamente copiar la primer columna completa,
    // incluidas las ghost cells, en la ultima columna ghost, y las esquinas van a tener el valor apropiado, la esquina
    // diagonal opuesta.
    size_t size = size_i + halo;

    if (my_id < size) {
        for (i = 0; i < neighs; i++) {
            d_lattice[my_id * size + (size_i + (i + neighs))] = d_lattice[my_id * size + (i + neighs)];  // copia primeras columnas en ultimas
            d_lattice[my_id * size + i] = d_lattice[my_id * size + (size_i + i)];                        // copia ultimas columnas en primeras
        }
    }
}

#define my_id_topa (y * (size_i + halo) + x)
#define col_topa (threadIdx.x + neighs)
#define row_topa (threadIdx.y + neighs)
#define my_sh_id_topa ((row_topa) * (blockDim.x + halo) + (col_topa))

__global__ void moveKernelTopa(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int neighs, int halo) {
    int count = 0;
    int x = blockDim.x * blockIdx.x + threadIdx.x + neighs;
    int y = blockDim.y * blockIdx.y + threadIdx.y + neighs;
    int v = 0;

    extern __shared__ MTYPE sh_lattice[];

    if (y < size_i + neighs && x < size_j + neighs) {
        sh_lattice[my_sh_id_topa] = d_lattice[my_id_topa];
    }

    if (row_topa == neighs || row_topa == neighs + 1) {
        for (v = 0; v < neighs; v++) {
            int gy = y - (row_topa - neighs);
            int up_or_down = ((blockDim.x + neighs) * (row_topa - neighs)) + v;

            sh_lattice[(up_or_down) * (blockDim.x + halo) + col_topa] = d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + x];
            // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row,col,   up_or_down,col, d_lattice[(gy - neighs + (up_or_down)) * (size_i + halo) + x]);

            // Corner Halos: left-up and left-down
            if ((col_topa - neighs) < neighs) {
                sh_lattice[(up_or_down) * (blockDim.x + halo) + (col_topa - neighs)] = d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + (x - neighs)];
                // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row, col,  up_or_down, col-neighs, d_lattice[(gy - neighs + (up_or_down)) * (size_i + halo) + (x-neighs)]);
            }

            // Corner Halos: right-up and right-down
            if ((col_topa + neighs) >= blockDim.y + neighs) {
                sh_lattice[(up_or_down) * (blockDim.x + halo) + (col_topa + neighs)] = d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + (x + neighs)];
                // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row, col,  up_or_down, col+neighs, sh_lattice[(up_or_down) * (blockDim.x+halo) + (col+neighs)] );
            }
        }

    } else if (row_topa == neighs + 2 || row_topa == neighs + 3) {
        for (v = 0; v < neighs; v++) {
            int gy = y - (row_topa - neighs);
            int gx = x - (col_topa - neighs);
            int lr = ((blockDim.y + neighs) * (row_topa & 1)) + v;

            // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, col, row,  col, lr, d_lattice[(gx - neighs + lr) + (gy + (col-neighs)) * (size_i + halo)]);
            sh_lattice[col_topa * (blockDim.x + halo) + lr] = d_lattice[(gx - neighs + lr) + (gy + (col_topa - neighs)) * (size_i + halo)];
        }
    }

    __syncthreads();

    if (x < size_i + neighs && y < size_j + neighs) {
        // if (i <= size_i && j <= size_j && (ii-1) != 0 && (ii-1) != blockDim.x && (jj-1) != 0 && (jj-1) != blockDim.y) {
        MTYPE c = sh_lattice[my_sh_id_topa];

        count = count_neighs(my_sh_id_topa, blockDim.x, sh_lattice, neighs, halo);  // decrease sh_size_x by 2 to use the same count_neighs function than the rest of the implementations
        d_lattice_new[my_id_topa] = c * h(count, SMIN, SMAX) + (1 - c) * h(count, BMIN, BMAX);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_lookup_table(int* GPU_lookup_table) {
    int(*lookup_table)[CAGIGAS_CELL_NEIGHBOURS + 1] = (int(*)[CAGIGAS_CELL_NEIGHBOURS + 1]) GPU_lookup_table;

    if (threadIdx.y < 2 && blockIdx.x < (CAGIGAS_CELL_NEIGHBOURS + 1)) {
        if (threadIdx.y == 0)
            if (blockIdx.x >= BMIN && blockIdx.x <= BMAX)
                lookup_table[threadIdx.y][blockIdx.x] = 1;
            else
                lookup_table[threadIdx.y][blockIdx.x] = 0;

        if (threadIdx.y == 1)
            if (blockIdx.x >= SMIN && blockIdx.x <= SMAX)
                lookup_table[threadIdx.y][blockIdx.x] = 1;
            else
                lookup_table[threadIdx.y][blockIdx.x] = 0;
    }
}

__global__ void ghostRows(uint64_t* grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
    // We want id ∈ [1,GRID_SIZE]
    size_t my_id = blockDim.x * blockIdx.x + threadIdx.x + horizontalHaloWidth;
    int fullHorizontalSize = ROW_SIZE + 2 * horizontalHaloWidth;

    if (my_id < (ROW_SIZE + horizontalHaloWidth)) {
        for (int currentHalo = 0; currentHalo < verticalHaloSize; currentHalo++) {
            // fill bottom halo
            grid[(currentHalo + verticalHaloSize + GRID_SIZE) * fullHorizontalSize + my_id] = grid[(currentHalo + verticalHaloSize) * fullHorizontalSize + my_id];

            // fill top halo
            grid[currentHalo * fullHorizontalSize + my_id] = grid[(currentHalo + GRID_SIZE) * fullHorizontalSize + my_id];
        }
    }
}

// __global__ void copy_Cols(int size_i, MTYPE* d_lattice, int neighs, int halo) {
//     int my_id = blockDim.x * blockIdx.x + threadIdx.x;
//     int i = 0;
//     // Al haber copiado la primer fila en la ultima columna, se puede directamente copiar la primer columna completa,
//     // incluidas las ghost cells, en la ultima columna ghost, y las esquinas van a tener el valor apropiado, la esquina
//     // diagonal opuesta.
//     int size = size_i + halo;

//     if (my_id < size) {
//         for (i = 0; i < neighs; i++) {
//             d_lattice[my_id * size + (size_i + (i + neighs))] = d_lattice[my_id * size + (i + neighs)];  // copia primeras columnas en ultimas
//             d_lattice[my_id * size + i] = d_lattice[my_id * size + (size_i + i)];                        // copia ultimas columnas en primeras
//         }
//     }
// }
__global__ void ghostCols(uint64_t* grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
    // We want id ∈ [0,SIZE+1]
    size_t my_id = blockDim.x * blockIdx.x + threadIdx.x;
    int fullHorizontalSize = ROW_SIZE + 2 * horizontalHaloWidth;
    int fullVerticalSize = GRID_SIZE + 2 * verticalHaloSize;

    if (my_id < fullVerticalSize) {
        for (int currentHalo = 0; currentHalo < horizontalHaloWidth; currentHalo++) {
            // Copy first real column to right most ghost column
            grid[(my_id) * (fullHorizontalSize) + horizontalHaloWidth + currentHalo + ROW_SIZE] = grid[(my_id) * (fullHorizontalSize) + horizontalHaloWidth + currentHalo];
            // Copy last real column to left most ghost column
            grid[my_id * (fullHorizontalSize) + currentHalo] = grid[my_id * (fullHorizontalSize) + currentHalo + ROW_SIZE];
        }
    }
}

__device__ inline int dist(int x0, int x1) {
    return abs(x0 - x1);
}

__global__ void GOL(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
    // We want id ∈ [1,SIZE]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + verticalHaloSize;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + horizontalHaloWidth;
    int fullHorizontalSize = ROW_SIZE + 2 * horizontalHaloWidth;
    int fullVerticalSize = GRID_SIZE + 2 * verticalHaloSize;

    size_t fullSharedWidth = blockDim.x + 2 * horizontalHaloWidth;
    size_t id = iy * (fullHorizontalSize) + ix;

    int current_cell_idx = threadIdx.x + horizontalHaloWidth;
    int current_cell_idy = threadIdx.y + verticalHaloSize;

    size_t sh_id = (current_cell_idy) * (fullSharedWidth) + current_cell_idx;

    uint64_t center_cell, new_cell = 0;
    unsigned char subcell;

    int k, numNeighbors;
    int(*lookup_table)[CAGIGAS_CELL_NEIGHBOURS + 1] = (int(*)[CAGIGAS_CELL_NEIGHBOURS + 1]) GPU_lookup_table;

    extern __shared__ uint64_t sh_grid[];

    int blockStart_x = blockIdx.x * blockDim.x;
    int blockStart_y = blockIdx.y * blockDim.y;

    for (int i = threadIdx.y; i < BSIZE3DY + 2 * verticalHaloSize; i += BSIZE3DY) {
        for (int j = threadIdx.x; j < BSIZE3DX + 2 * horizontalHaloWidth; j += BSIZE3DX) {
            if ((blockStart_y + i) < fullVerticalSize && blockStart_x + j < fullHorizontalSize) {
                sh_grid[i * (BSIZE3DX + 2 * horizontalHaloWidth) + j] = grid[(blockStart_y + i) * fullHorizontalSize + blockStart_x + j];
            }
        }
    }
    // __syncthreads();
    // if (threadIdx.x + threadIdx.y == 0) {
    //     // print the whole shared memory
    //     for (int i = 0; i < BSIZE3DY + 2 * verticalHaloSize; i++) {
    //         for (int j = 0; j < BSIZE3DX + 2 * horizontalHaloWidth; j++) {
    //             uint64_t v = sh_grid[i * (BSIZE3DX + 2 * horizontalHaloWidth) + j];
    //             printf("%d %d %d %d %d %d %d %d ", getSubCellD(v, 0), getSubCellD(v, 1), getSubCellD(v, 2), getSubCellD(v, 3), getSubCellD(v, 4), getSubCellD(v, 5), getSubCellD(v, 6), getSubCellD(v, 7));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
    // if (threadIdx.x + threadIdx.y == 0) {
    //     // print the whole grid
    //     for (int i = 0; i < fullVerticalSize; i++) {
    //         for (int j = 0; j < fullHorizontalSize; j++) {
    //             uint64_t v = grid[i * fullHorizontalSize + j];
    //             printf("%d %d %d %d %d %d %d %d ", getSubCellD(v, 0), getSubCellD(v, 1), getSubCellD(v, 2), getSubCellD(v, 3), getSubCellD(v, 4), getSubCellD(v, 5), getSubCellD(v, 6), getSubCellD(v, 7));
    //         }
    //         printf("\n");
    //     }
    // }
    __syncthreads();

    // uint64_t cells[(2 * RADIUS + 1) * (2 * ceil(RADIUS / 8.0f) + 1) + 1];
    unsigned char subcells[ELEMENTS_PER_CELL];
    for (int i = 0; i < ELEMENTS_PER_CELL; i++) {
        subcells[i] = 0;
    }
    if (iy >= verticalHaloSize && iy < GRID_SIZE + verticalHaloSize && ix >= horizontalHaloWidth && ix < ROW_SIZE + horizontalHaloWidth) {
        // center_cell = sh_grid[sh_id];

        numNeighbors = 0;
        for (int i = -RADIUS; i <= RADIUS; i++) {
#pragma unroll
            for (int j = -RADIUS; j < 8 + RADIUS; j++) {
                int currentNeighSubcellIndex = (j) & (ELEMENTS_PER_CELL - 1);
                int currentNeighPosition_y = threadIdx.y + verticalHaloSize + i;
                int currentNeighUnpackedPosition_x = (threadIdx.x + horizontalHaloWidth) * 8 + j;

                int currentNeighPosition_x = currentNeighUnpackedPosition_x / 8;
                // print the variables above

                // print all info19
                int currentNeighCellIndex = currentNeighPosition_y * fullSharedWidth + currentNeighPosition_x;

                // int currentNeighCellIndex = (i + cell_idy) * fullSharedWidth + (cell_idx + j);
                // print i, j and ((j + 1) % 2) - 1
                // printf("i=%d j=%d -> %d\n", i, j, (current_id_x + j) / 8);
                uint64_t currentNeighCell = sh_grid[currentNeighCellIndex];
                // if (threadIdx.x == 0 && threadIdx.y == 0)
                //     printf("threadIdx.x=%d threadIdx.y=%d i=%d j=%d currentNeighSubcellIndex=%d currentNeighPosition_y=%d currentNeighUnpackedPosition_x=%d currentNeighPosition_x=%d, currentNeighCell=%d %d %d %d %d %d %d %d \n", threadIdx.x, threadIdx.y, i, j, currentNeighSubcellIndex, currentNeighPosition_y, currentNeighUnpackedPosition_x, currentNeighPosition_x, getSubCellD(currentNeighCell, 0), getSubCellD(currentNeighCell, 1), getSubCellD(currentNeighCell, 2), getSubCellD(currentNeighCell, 3), getSubCellD(currentNeighCell, 4), getSubCellD(currentNeighCell, 5), getSubCellD(currentNeighCell, 6), getSubCellD(currentNeighCell, 7));

                unsigned char subcell = getSubCellD(currentNeighCell, currentNeighSubcellIndex);
                for (int k = 0; k < ELEMENTS_PER_CELL; k++) {
                    int currentSubCellPosition_x = current_cell_idx * 8 + k;
                    if (currentSubCellPosition_x == currentNeighUnpackedPosition_x && currentNeighPosition_y == current_cell_idy) {
                        continue;
                    }
                    if (dist(currentSubCellPosition_x, currentNeighUnpackedPosition_x) <= RADIUS && dist(current_cell_idy, currentNeighPosition_y) <= RADIUS) {
                        // if (threadIdx.x == 0 && threadIdx.y == 0)
                        //     printf("i=%d j=%d k=%d -> %d, subcell=%d, ty=%d\n", i, j, k, currentSubCellPosition_x, subcell, threadIdx.y);

                        subcells[k] += subcell;
                    }
                }
                // numNeighbors += getSubCellD(current_cell, subcell_idx);
            }
        }
        // if (threadIdx.x == 0 && threadIdx.y == 0)
        //     printf("subcells: %d %d %d %d %d %d %d %d\n", subcells[0], subcells[1], subcells[2], subcells[3], subcells[4], subcells[5], subcells[6], subcells[7]);
        for (int i = 0; i < ELEMENTS_PER_CELL; i++) {
            setSubCellD(&new_cell, i, lookup_table[getSubCellD(sh_grid[sh_id], i)][subcells[i]]);
        }
        // printf("%d %d %d %d %d %d %d %d\n", lookup_table[getSubCellD(sh_grid[sh_id], 0)][subcells[0]], lookup_table[getSubCellD(sh_grid[sh_id], 1)][subcells[1]], lookup_table[getSubCellD(sh_grid[sh_id], 2)][subcells[2]], lookup_table[getSubCellD(sh_grid[sh_id], 3)][subcells[3]], lookup_table[getSubCellD(sh_grid[sh_id], 4)][subcells[4]], lookup_table[getSubCellD(sh_grid[sh_id], 5)][subcells[5]], lookup_table[getSubCellD(sh_grid[sh_id], 6)][subcells[6]], lookup_table[getSubCellD(sh_grid[sh_id], 7)][subcells[7]]);
        // setSubCellD(&new_cell, 0, lookup_table[getSubCellD(center_cell, 0)][numNeighbors]);
        // First (0) subcell:
        // up_cell = sh_grid[sh_id - (fullSharedWidth)];
        // down_cell = sh_grid[sh_id + (fullSharedWidth)];
        // left_cell = sh_grid[sh_id - 1];
        // upleft_cell = sh_grid[sh_id - (fullSharedWidth + 1)];
        // downleft_cell = sh_grid[sh_id + (fullSharedWidth - 1)];
        // right_cell = sh_grid[sh_id + 1];
        // upright_cell = sh_grid[sh_id - (fullSharedWidth - 1)];
        // downright_cell = sh_grid[sh_id + (fullSharedWidth + 1)];

        // numNeighbors = getSubCellD(up_cell, 0) + getSubCellD(down_cell, 0);                                                   // upper lower
        // numNeighbors += getSubCellD(left_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(center_cell, 1);                          // left right
        // numNeighbors += getSubCellD(upleft_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(downleft_cell, ELEMENTS_PER_CELL - 1);  // diagonals left=
        // numNeighbors += getSubCellD(up_cell, 1) + getSubCellD(down_cell, 1);                                                  // diagonals right
        // subcell = getSubCellD(center_cell, 0);
        // setSubCellD(&new_cell, 0, lookup_table[subcell][numNeighbors]);

        // // Middle subcells:
        // for (k = 1; k < CAGIGAS_CELL_NEIGHBOURS - 1; k++) {
        //     numNeighbors = getSubCellD(up_cell, k) + getSubCellD(down_cell, k);                 // upper lower
        //     numNeighbors += getSubCellD(center_cell, k - 1) + getSubCellD(center_cell, k + 1);  // left right
        //     numNeighbors += getSubCellD(up_cell, k - 1) + getSubCellD(down_cell, k - 1);        // diagonals left
        //     numNeighbors += getSubCellD(up_cell, k + 1) + getSubCellD(down_cell, k + 1);        // diagonals right
        //     subcell = getSubCellD(center_cell, k);
        //     setSubCellD(&new_cell, k, lookup_table[subcell][numNeighbors]);
        // }

        // numNeighbors = getSubCellD(up_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(down_cell, ELEMENTS_PER_CELL - 1);   // upper lower
        // numNeighbors += getSubCellD(center_cell, ELEMENTS_PER_CELL - 2) + getSubCellD(right_cell, 0);                 // left right
        // numNeighbors += getSubCellD(up_cell, ELEMENTS_PER_CELL - 2) + getSubCellD(down_cell, ELEMENTS_PER_CELL - 2);  // diagonals left
        // numNeighbors += getSubCellD(upright_cell, 0) + getSubCellD(downright_cell, 0);                                // diagonals right
        // subcell = getSubCellD(center_cell, ELEMENTS_PER_CELL - 1);
        // setSubCellD(&new_cell, ELEMENTS_PER_CELL - 1, lookup_table[subcell][numNeighbors]);

        // Copy new_cell to newGrid:
        newGrid[id] = new_cell;

        /*
                // Get the number of neighbors for a given grid point
                numNeighbors = grid[id+(SIZE+2)] + grid[id-(SIZE+2)] //upper lower
                             + grid[id+1] + grid[id-1]             //right left
                             + grid[id+(SIZE+3)] + grid[id-(SIZE+3)] //diagonals
                             + grid[id-(SIZE+1)] + grid[id+(SIZE+1)];

                uint64_t center_cell = grid[id];
                newGrid[id] = lookup_table[center_cell][numNeighbors];
        */
    }
}

__forceinline__ unsigned char getSubCellH(uint64_t cell, char pos) {
    return (cell >> (ELEMENTS_PER_CELL - 1 - pos) * 8);
}

__forceinline__ void setSubCellH(uint64_t* cell, char pos, unsigned char subcell) {
    uint64_t mask = 0xFF;
    uint64_t maskNewCell = subcell;

    // Erase pos content in cell:
    mask = mask << (ELEMENTS_PER_CELL - 1 - pos) * 8;
    mask = ~mask;
    *cell = *cell & mask;

    // Add subcell content to cell in pos:
    *cell = *cell | (maskNewCell << (ELEMENTS_PER_CELL - 1 - pos) * 8);
}

__device__ unsigned char getSubCellD(uint64_t cell, char pos) {
    return (cell >> (ELEMENTS_PER_CELL - 1 - pos) * 8);
}

__device__ void setSubCellD(uint64_t* cell, char pos, unsigned char subcell) {
    uint64_t mask = 0xFF;
    uint64_t maskNewCell = subcell;

    // Erase pos content in cell:
    mask = mask << (ELEMENTS_PER_CELL - 1 - pos) * 8;
    mask = ~mask;
    *cell = *cell & mask;

    // Add subcell content to cell in pos:
    *cell = *cell | (maskNewCell << (ELEMENTS_PER_CELL - 1 - pos) * 8);
}

__global__ void unpackState(uint64_t* from, int* to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
    // We want id ∈ [1,SIZE]
    size_t unpacked_x = (blockDim.x * blockIdx.x + threadIdx.x) * 8 + verticalHaloSize;
    size_t unpacked_y = blockDim.y * blockIdx.y + threadIdx.y + verticalHaloSize;

    size_t packed_x = (blockDim.x * blockIdx.x + threadIdx.x) + horizontalHaloWidth;

    size_t unpackedIndex = unpacked_y * (GRID_SIZE + 2 * verticalHaloSize) + unpacked_x;
    size_t packedIndex = unpacked_y * (ROW_SIZE + 2 * horizontalHaloWidth) + packed_x;
    // print all i in one line

    uint64_t cellValue;
    unsigned char subcell;

    if (unpacked_y < GRID_SIZE + 2 * verticalHaloSize && unpacked_x < GRID_SIZE + 2 * verticalHaloSize) {
        cellValue = from[packedIndex];
        for (int i = 0; i < ELEMENTS_PER_CELL; i++) {
            subcell = getSubCellD(cellValue, i);
            to[unpackedIndex + i] = subcell;
        }
    }
}

__global__ void packState(int* from, uint64_t* to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
    // We want id ∈ [1,SIZE]
    size_t unpacked_x = (blockDim.x * blockIdx.x + threadIdx.x) * 8 + verticalHaloSize;
    size_t unpacked_y = blockDim.y * blockIdx.y + threadIdx.y + verticalHaloSize;

    size_t packed_x = (blockDim.x * blockIdx.x + threadIdx.x) + horizontalHaloWidth;

    size_t unpackedIndex = unpacked_y * (GRID_SIZE + 2 * verticalHaloSize) + unpacked_x;
    size_t packedIndex = unpacked_y * (ROW_SIZE + 2 * horizontalHaloWidth) + packed_x;
    // print all i in one line
    uint64_t cellValue;
    // if (threadIdx.x + threadIdx.y == 0) {
    //     for (int i = 0; i < GRID_SIZE + 2 * verticalHaloSize; i++) {
    //         for (int j = 0; j < GRID_SIZE + 2 * verticalHaloSize; j++) {
    //             printf("%d ", from[i * (GRID_SIZE + 2 * verticalHaloSize) + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // printf("(%d, %d) = %d\n", unpacked_x, unpacked_y, unpackedIndex);
    if (unpacked_y < GRID_SIZE + 2 * verticalHaloSize && unpacked_x < GRID_SIZE + 2 * verticalHaloSize) {
        cellValue = 0;
        // print

        // printf("%d, %d -> %d %d %d %d %d %d %d %d\n", unpacked_y, unpacked_x, from[unpackedIndex], from[unpackedIndex + 1], from[unpackedIndex + 2], from[unpackedIndex + 3], from[unpackedIndex + 4], from[unpackedIndex + 5], from[unpackedIndex + 6], from[unpackedIndex + 7]);

        for (int i = 0; i < ELEMENTS_PER_CELL; i++) {
            setSubCellD(&cellValue, i, from[unpackedIndex + i]);
        }
        to[packedIndex] = cellValue;
        // printf("%d, %d -> cellValue=%lx\n", unpacked_y, unpacked_x, cellValue);
    }
}

// __global__ void GOL(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize) {
//     // We want id ∈ [1,SIZE]
//     int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
//     int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
//     size_t id = iy * (ROW_SIZE + 2) + ix;
//     uint64_t cell, new_cell = 0;
//     uint64_t up_cell, down_cell, right_cell, left_cell;                 // Up,down,right,left cells
//     uint64_t upleft_cell, downleft_cell, upright_cell, downright_cell;  // Diagonal cells
//     unsigned char subcell;

//     int k, numNeighbors;
//     int(*lookup_table)[CAGIGAS_CELL_NEIGHBOURS + 1] = (int(*)[CAGIGAS_CELL_NEIGHBOURS + 1]) GPU_lookup_table;

//     if (iy > 0 && iy <= GRID_SIZE && ix > 0 && ix <= ROW_SIZE) {
//         cell = grid[id];

//         // First (0) subcell:
//         up_cell = grid[id - (ROW_SIZE + 2)];
//         down_cell = grid[id + (ROW_SIZE + 2)];
//         left_cell = grid[id - 1];
//         upleft_cell = grid[id - (ROW_SIZE + 3)];
//         downleft_cell = grid[id + (ROW_SIZE + 1)];

//         numNeighbors = getSubCellD(up_cell, 0) + getSubCellD(down_cell, 0);                                                   // upper lower
//         numNeighbors += getSubCellD(left_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(cell, 1);                                 // left right
//         numNeighbors += getSubCellD(upleft_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(downleft_cell, ELEMENTS_PER_CELL - 1);  // diagonals left
//         numNeighbors += getSubCellD(up_cell, 1) + getSubCellD(down_cell, 1);                                                  // diagonals right
//         subcell = getSubCellD(cell, 0);
//         setSubCellD(&new_cell, 0, lookup_table[subcell][numNeighbors]);

//         // Middle subcells:
//         for (k = 1; k < CAGIGAS_CELL_NEIGHBOURS - 1; k++) {
//             numNeighbors = getSubCellD(up_cell, k) + getSubCellD(down_cell, k);           // upper lower
//             numNeighbors += getSubCellD(cell, k - 1) + getSubCellD(cell, k + 1);          // left right
//             numNeighbors += getSubCellD(up_cell, k - 1) + getSubCellD(down_cell, k - 1);  // diagonals left
//             numNeighbors += getSubCellD(up_cell, k + 1) + getSubCellD(down_cell, k + 1);  // diagonals right
//             subcell = getSubCellD(cell, k);
//             setSubCellD(&new_cell, k, lookup_table[subcell][numNeighbors]);
//         }

//         // Last (CAGIGAS_CELL_NEIGHBOURS-1) subcell:
//         right_cell = grid[id + 1];
//         upright_cell = grid[id - (ROW_SIZE + 1)];
//         downright_cell = grid[id + (ROW_SIZE + 3)];

//         numNeighbors = getSubCellD(up_cell, ELEMENTS_PER_CELL - 1) + getSubCellD(down_cell, ELEMENTS_PER_CELL - 1);   // upper lower
//         numNeighbors += getSubCellD(cell, ELEMENTS_PER_CELL - 2) + getSubCellD(right_cell, 0);                        // left right
//         numNeighbors += getSubCellD(up_cell, ELEMENTS_PER_CELL - 2) + getSubCellD(down_cell, ELEMENTS_PER_CELL - 2);  // diagonals left
//         numNeighbors += getSubCellD(upright_cell, 0) + getSubCellD(downright_cell, 0);                                // diagonals right
//         subcell = getSubCellD(cell, ELEMENTS_PER_CELL - 1);
//         setSubCellD(&new_cell, ELEMENTS_PER_CELL - 1, lookup_table[subcell][numNeighbors]);

//         // Copy new_cell to newGrid:
//         newGrid[id] = new_cell;

//         /*
//                 // Get the number of neighbors for a given grid point
//                 numNeighbors = grid[id+(SIZE+2)] + grid[id-(SIZE+2)] //upper lower
//                              + grid[id+1] + grid[id-1]             //right left
//                              + grid[id+(SIZE+3)] + grid[id-(SIZE+3)] //diagonals
//                              + grid[id-(SIZE+1)] + grid[id+(SIZE+1)];

//                 uint64_t cell = grid[id];
//                 newGrid[id] = lookup_table[cell][numNeighbors];
//         */
//     }
// }

#endif
