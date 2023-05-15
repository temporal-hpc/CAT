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

#define HINDEX(x, y, nWithHalo) ((y + R) * ((size_t)nWithHalo) + (x + R))
#define GINDEX(x, y, nshmem) ((y) * (nshmem) + (x))

__device__ inline int h(int k, int a, int b) {
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__forceinline__ __device__ void workWithShmem(MTYPE* pDataOut, MTYPE* shmem, uint2 dataCoord, uint32_t nWithHalo, uint32_t nShmem) {
    // neighborhood count
	int nc = 0;
	for (int i=-R; i<=R; i++){
		for (int j=-R; j<=R; j++){
			//nc += pDataIn[i+j];
			nc += shmem[HINDEX(threadIdx.x+j, threadIdx.y+i, nShmem)];
		}
	}
    //int nc
    //   	= shmem[HINDEX(threadIdx.x - 1, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y - 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y - 1, nShmem)]
    //    + shmem[HINDEX(threadIdx.x - 1, threadIdx.y    , nShmem)] /*                                                 */ + shmem[HINDEX(threadIdx.x + 1, threadIdx.y,     nShmem)]
    //    + shmem[HINDEX(threadIdx.x - 1, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x, threadIdx.y + 1, nShmem)] + shmem[HINDEX(threadIdx.x + 1, threadIdx.y + 1, nShmem)];

    unsigned int c = shmem[HINDEX(threadIdx.x, threadIdx.y, nShmem)];
	//nc -= c;
    //pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;//c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
}

__forceinline__ __device__ void workWithGbmem(MTYPE* pDataIn, MTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo) {
    // neighborhood count
    //int nc
    //    = pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y - 1, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y, nWithHalo)] /*                                                     */ + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y + 1, nWithHalo)];
	int nc = 0;
	for (int i=-R; i<=R; i++){
		for (int j=-R; j<=R; j++){
			//nc += pDataIn[i+j];
			nc += pDataIn[HINDEX(dataCoord.x+j, dataCoord.y+i, nWithHalo)];
		}
	}

    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
	//nc -= c;
    //pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;
}

__global__ void ClassicGlobalMemGoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo) {
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };
    if (dataCoord.x < n && dataCoord.y < n) {
        workWithGbmem(pDataIn, pDataOut, dataCoord, nWithHalo);
    }
}
__forceinline__ __device__ void workWithGbmemHALF(FTYPE* pDataIn, FTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo) {
    // neighborhood count
	int nc = 0;
	for (int i=-R; i<=R; i++){
		for (int j=-R; j<=R; j++){
			//nc += pDataIn[i+j];
			nc += __half2int_rd(pDataIn[HINDEX(dataCoord.x+j, dataCoord.y+i, nWithHalo)]);
		}
	}

    //int nc
    //    = pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y - 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y - 1, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y, nWithHalo)] /*                                                     */ + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y, nWithHalo)]
    //    + pDataIn[HINDEX(dataCoord.x - 1, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x, dataCoord.y + 1, nWithHalo)] + pDataIn[HINDEX(dataCoord.x + 1, dataCoord.y + 1, nWithHalo)];

    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)];
	//nc -= c;
    //pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = c * h(nc, EL, EU) + (1 - c) * h(nc, FL, FU);
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo)] = nc;
}

__global__ void ClassicGlobalMemHALFGoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };
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

    __shared__ MTYPE shmem[(SHMEM_N) * (SHMEM_N)];
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };

	for (uint32_t i = tid; i < SHMEM_N*SHMEM_N; i += BSIZE3DX * BSIZE3DY){
		uint32_t shmem_x = i % SHMEM_N;
		uint32_t shmem_y = i / SHMEM_N;
		uint32_t data_x = dataBlockCoord_x + shmem_x;
		uint32_t data_y = dataBlockCoord_y + shmem_y;
		//uint2 dataCoord = { dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y };
	
		printf("SHMEM_N: %u, i:%u,  sh %u,%u  ---  data %u,%u\n", SHMEM_N, i, shmem_x, shmem_y, data_x, data_y);
    	if (data_x < nWithHalo && data_y < nWithHalo) {
        	shmem[GINDEX(shmem_x, shmem_y, SHMEM_N)] = pDataIn[GINDEX(data_x, data_y, nWithHalo)];
		}

    }
    __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x + threadIdx.y == 0) {
        for (int k = 0; k < SHMEM_N; k++) {
            for (int l = 0; l < SHMEM_N; l++) {
                printf("%i ", shmem[k * SHMEM_N + l]);
            }
            printf("\n");
        }
    }
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
    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;
    FTYPE* shmem2 = (FTYPE*)((nShmemH * nShmemV) * sizeof(FTYPE) + totalshmem);
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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
    extern __shared__ char totalshmem[];
    FTYPE* shmem = (FTYPE*)totalshmem;
    FTYPE* shmem2 = (FTYPE*)((nShmemH * nShmemV) * sizeof(FTYPE) + totalshmem);

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
        // uint32_t regionCoord_x = blockIdx.x * NREGIONS_H * 16; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        // uint32_t regionCoord_y = blockIdx.y * NREGIONS_V * 16; //  = nShmemH = (6+2)*16
        // // for (char fragRow = 0; i < 8; i += 1) {
        // uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        // uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));
        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * nWithHalo / 16 + (globalFragment_x)*256 + tid % 256;
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * (nWithHalo / 16) + (globalFragment_x)*256 + index % 256;
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
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
        }
    }
}

// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorCoalescedV2GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo) {

    // A typical 'row' of T_0. Each warp store a row into shared mem T_0, starting at a different position
    // and cycling like a circle array. Ej: ▽ is the starting position of warp 0
    // const half tridiagTemplate[17] = { 1.f, 1.f, 1.f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    const uint32_t nFragmentsH = NREGIONS_H + 2; // ⚠️ The code relies on this being 8 !!
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

    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
    }
    for (uint32_t i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    // Copying the corresponding global memory region to shared

    for (uint32_t index = tid; index < nShmemH * nShmemV; index += BSIZE3DX * BSIZE3DY) {
        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + ((index / 256) % nFragmentsH);
        uint32_t globalFragment_y = regionCoord_y + (index / (256 * nFragmentsH));

        size_t dindex = (globalFragment_y)*256 * nWithHalo / 16 + (globalFragment_x)*256 + tid % 256;
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y)*256 * (nWithHalo / 16) + (globalFragment_x)*256 + index % 256;

        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {

            size_t ind = (fy + 1) * 256 * nFragmentsH + (fx + 1) * 256 + index % 256;
            uint32_t val = __half2uint_rn(shmem[ind]);
            // uint32_t val = __half2uint_rn(shmem[index + 16*nShmemH+256]);
            float val2 = __half2float(pDataIn[dindex]);

            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
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
        shmem_tridiag[i] = (16+R - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (16 - (i&15) + (i>>4)) /(32-R); //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> a_frag3;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB; // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA; // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA; // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA; // Row major

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
        const uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        const uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16

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

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16

        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y * nWithHalo16 + globalFragment_x) * 256 + (index & 255);
        if (globalFragment_x < (nWithHalo16)-1 && globalFragment_y < (nWithHalo16)-1) {

            uint32_t val = __half2uint_rn(pDataOut[dindex]);
            float val2 = __half2float(pDataIn[dindex]);
            pDataOut[dindex] = val;//__uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
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

    // Procedurally generating T_0 and T_1. T_2 is just T_0 transposed.
    // printf("%.f\n", __half2float())
    for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
    }
    for (uint32_t i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
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

    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_0_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> T_1_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::col_major> T_2_asB; // Col major

    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;
    // if (tid==0)
    // printf("%i\n", wcount);
    //__shared__ half aux[256];
    // int tts = 6;

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V + 2); rid += wcount) {

        const uint32_t workFragment_x = (rid % NREGIONS_H);
        const uint32_t workFragment_y = (rid / NREGIONS_H);
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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
    for (int i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 - abs((i >> 4) - (i & 15))) >> 4; // tridiagTemplate[index];
    }
    for (uint32_t i = tid; i < 256; i += BSIZE3DX * BSIZE3DY) {
        shmem_tridiag[i + 16 * 16] = (((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }
    for (uint32_t val2id = 0; val2id < NREGIONS_H * 16 * NREGIONS_V * 16 / (BSIZE3DX * BSIZE3DY); val2id += 1) {
        const int t = tid + BSIZE3DX * BSIZE3DY * val2id;
        uint32_t fid = t / 256;
        uint32_t fx = fid % NREGIONS_H;
        uint32_t fy = fid / NREGIONS_H;

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * (nWithHalo / 16) + (globalFragment_x)*256 + t % 256;
        if (globalFragment_x < (nWithHalo / 16) - 1 && globalFragment_y < (nWithHalo / 16) - 1) {

            val2s[val2id] = __half2uint_rn(pDataOut[dindex]);
            // printf("%u -> %i\n", val2id, val2s[val2id]);
        }
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, FTYPE_ACC> c_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, FTYPE, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::col_major> T_0_asA; // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_1_asA; // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, FTYPE, wmma::row_major> T_2_asA; // Row major
    const uint8_t wcount = (BSIZE3DX * BSIZE3DY) / 32;

    for (uint32_t rid = wid; rid < NREGIONS_H * (NREGIONS_V); rid += wcount) {
        const uint8_t workFragment_x = rid % NREGIONS_H;
        const uint8_t workFragment_y = rid / NREGIONS_H;
        uint32_t regionCoord_x = blockIdx.x * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = blockIdx.y * NREGIONS_V; //  = nShmemH = (6+2)*16
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

        uint32_t regionCoord_x = (blockIdx.x) * NREGIONS_H; // ⚠️ bc of this hardcoded 127 !! nShmemH-1
        uint32_t regionCoord_y = (blockIdx.y) * NREGIONS_V; //  = nShmemH = (6+2)*16
        // for (char fragRow = 0; i < 8; i += 1) {
        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        // uint32_t globalFragmentLinear = blockIdx.y * NREGIONS_V * 16 + regionCoord_x + ((index / 256) % nFragmentsH) * 256 + (tid % 256);

        // printf()
        size_t dindex = (globalFragment_y)*256 * (nWithHalo / 16) + (globalFragment_x)*256 + index % 256;
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
            // printf("%i\n", c);

            // if (blockIdx.x == 1 && blockIdx.y == 0 && index % 256 == 0)

            //     printf("%llu -- (%i,%i) = (%i, %i) -> %llu\n", ind, fx, fy, globalFragment_x, globalFragment_y, dindex);

            // shmem[index] = pDataIn[dindex];
            pDataOut[dindex] = __uint2half_rn(val2 * h(val - val2, EL, EU) + (1 - val2) * h(val - val2, FL, FU));
        }
        c += 1;
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

    uint32_t xx = tid % 16;
    uint32_t yy = tid / 16;

    uint32_t xxx = xx / 8;
    uint32_t yyy = yy / 8;

    uint32_t reg = yyy * 2 + xxx;

    uint32_t outt = reg * 16 * 4 + tid % 8 + (yy % 8) * 8;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);
    // printf("%u\n", outt);
    if (tx < nWithHalo && ty < nWithHalo) {
        // out[bid * blockDim.x * blockDim.y + outt] = __uint2half_rn(in[ty * nWithHalo + tx]);
        out[bid * 256 + tid] = __uint2half_rn(in[ty * nWithHalo + tx]);
    }
}
__global__ void convertFp16ToFp32AndUndoChangeLayout(MTYPE* out, FTYPE* in, size_t nWithHalo) {
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);
    uint32_t xx = tid % 16;
    uint32_t yy = tid / 16;

    uint32_t xxx = xx / 8;
    uint32_t yyy = yy / 8;

    uint32_t reg = yyy * 2 + xxx;

    uint32_t outt = reg * 16 * 4 + tid % 8 + (yy % 8) * 8;
    if (tx < nWithHalo && ty < nWithHalo) {
        out[ty * nWithHalo + tx] = __half2uint_rn(in[bid * 256 + tid]);
        // out[ty * nWithHalo + tx] = __half2uint_rn(in[bid * blockDim.x * blockDim.y + outt]);
    }
}
