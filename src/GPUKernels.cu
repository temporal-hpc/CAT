#ifndef _CLASSIC_GOL_KERNELS_H_
#define _CLASSIC_GOL_KERNELS_H_
#include "GPUKernels.cuh"

#include <cuda.h>
#include <stdio.h>

#include <mma.h>
using namespace nvcuda;

#define CELL_NEIGHBOURS 8

__device__ inline int h(int k, int a, int b)
{
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__forceinline__ __device__ void workWithShmem(unsigned char *pDataOut, unsigned char *shmem, uint2 dataCoord,
                                              uint32_t nWithHalo, uint32_t nShmem, int radius)
{
    int nc = 0;
#pragma unroll
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            nc += shmem[HINDEX(threadIdx.x + j, threadIdx.y + i, nShmem, radius)];
        }
    }
    unsigned int c = shmem[HINDEX(threadIdx.x, threadIdx.y, nShmem, radius)];
    nc -= c;
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo, radius)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
}

__forceinline__ __device__ void workWithGbmem(unsigned char *pDataIn, unsigned char *pDataOut, uint2 dataCoord,
                                              uint32_t nWithHalo, int radius)
{
    int nc = 0;
#pragma unroll
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            nc += pDataIn[HINDEX(dataCoord.x + j, dataCoord.y + i, nWithHalo, radius)];
        }
    }
    unsigned int c = pDataIn[HINDEX(dataCoord.x, dataCoord.y, nWithHalo, radius)];
    nc -= c;
    pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo, radius)] = c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
}

__global__ void BASE_KERNEL(unsigned char *pDataIn, unsigned char *pDataOut, size_t n, size_t nWithHalo, int radius)
{
    uint32_t dataBlockCoord_x = blockIdx.x * blockDim.x;
    uint32_t dataBlockCoord_y = blockIdx.y * blockDim.y;
    uint2 dataCoord = {dataBlockCoord_x + threadIdx.x, dataBlockCoord_y + threadIdx.y};
    if (dataCoord.x < n && dataCoord.y < n)
    {
        workWithGbmem(pDataIn, pDataOut, dataCoord, nWithHalo, radius);
    }
}

__global__ void COARSE_KERNEL(unsigned char *pDataIn, unsigned char *pDataOut, size_t n, size_t nWithHalo, int radius)
{
    __shared__ unsigned char shmem[(80) * (80)];
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    // uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t dataBlockCoord_x = blockIdx.x * 80;
    uint32_t dataBlockCoord_y = blockIdx.y * 80;

    for (uint32_t i = tid; i < 80 * 80; i += blockDim.x * blockDim.y)
    {
        uint32_t shmem_x = i % 80;
        uint32_t shmem_y = i / 80;
        uint32_t data_x = dataBlockCoord_x + shmem_x;
        uint32_t data_y = dataBlockCoord_y + shmem_y;
        if (data_x < nWithHalo && data_y < nWithHalo)
        {
            shmem[GINDEX(shmem_x, shmem_y, 80)] = pDataIn[GINDEX(data_x, data_y, nWithHalo)];
        }
    }
    __syncthreads();
    for (uint32_t i = tid; i < 80 * 80; i += blockDim.x * blockDim.y)
    {
        uint32_t shmem_x = i % 80;
        uint32_t shmem_y = i / 80;
        uint32_t data_x = dataBlockCoord_x + shmem_x;
        uint32_t data_y = dataBlockCoord_y + shmem_y;
        uint2 dataCoord = {data_x, data_y};
        if (dataCoord.x < n && dataCoord.y < n)
        {
            int nc = 0;
#pragma unroll
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    nc += shmem[HINDEX(shmem_x + j, shmem_y + i, 80, radius)];
                }
            }
            unsigned int c = shmem[HINDEX(shmem_x, shmem_y, 80, radius)];
            nc -= c;
            pDataOut[HINDEX(dataCoord.x, dataCoord.y, nWithHalo, radius)] =
                c * h(nc, SMIN, SMAX) + (1 - c) * h(nc, BMIN, BMAX);
        }
    }
}

__global__ void CAT_KERNEL(half *pDataIn, half *pDataOut, size_t n, size_t nWithHalo, int radius, int nRegionsH,
                           int nRegionsV)
{
    const uint32_t nFragmentsH = nRegionsH + 2;

    extern __shared__ unsigned char totalshmem[];
    half *shmem = (half *)totalshmem;

    __shared__ half shmem_tridiag[16 * 16 * 2];

    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t wid = tid / 32;

    int i;
#pragma unroll
    for (i = tid; i < 256; i += blockDim.x * blockDim.y)
    {
        //  printf("%u,%u = %.0f\n", i, index, __half2float(tridiagTemplate[index]));
        shmem_tridiag[i] = (17 + radius - abs((i >> 4) - (i & 15))) / 17; // tridiagTemplate[index];
    }
#pragma unroll
    for (i = tid; i < 256; i += blockDim.x * blockDim.y)
    {
        shmem_tridiag[i + 16 * 16] =
            (16 - (i & 15) + (i >> 4)) / (32 - radius); //(((i >> 4) + 1) >> 4) * ((16 - (i & 15)) >> 4);
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag3;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fill_fragment(c_frag, 0);

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> T_0_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> T_1_asB; // Row major
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> T_2_asB; // Col major

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> T_0_asA; // Col major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> T_1_asA; // Row major
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> T_2_asA; // Row major

    const uint8_t wcount = (blockDim.x * blockDim.y) / 32;

    const uint32_t n16 = n >> 4;
    const uint32_t nWithHalo16 = nWithHalo >> 4;
#pragma unroll

    for (uint32_t rid = wid; rid < nRegionsH * (nRegionsV + 2); rid += wcount)
    {
        const uint32_t workFragment_x = (rid % nRegionsH);
        const uint32_t workFragment_y = (rid / nRegionsH);
        const uint32_t regionCoord_x = blockIdx.x * nRegionsH;
        const uint32_t regionCoord_y = blockIdx.y * nRegionsV;
        // for (unsigned char fragRow = 0; i < 8; i += 1) {
        const uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        const uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (!(globalFragment_x < n16 && globalFragment_y < nWithHalo16))
        {
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

        wmma::store_matrix_sync(&shmem[workFragment_y * nFragmentsH * 256 + (workFragment_x + 1) * 256], c_frag, 16,
                                wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
#pragma unroll

    for (uint32_t rid = wid; rid < nRegionsH * (nRegionsV); rid += wcount)
    {
        const uint32_t workFragment_x = rid % nRegionsH;
        const uint32_t workFragment_y = rid / nRegionsH;
        const uint32_t regionCoord_x = blockIdx.x * nRegionsH;
        const uint32_t regionCoord_y = blockIdx.y * nRegionsV;

        uint32_t globalFragment_x = regionCoord_x + workFragment_x;
        uint32_t globalFragment_y = regionCoord_y + workFragment_y;

        if (globalFragment_x >= n16 || globalFragment_y >= n16)
        {
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

        wmma::store_matrix_sync(&pDataOut[((globalFragment_y + 1) * nWithHalo16 + (globalFragment_x + 1)) * 256],
                                c_frag, 16, wmma::mem_row_major);
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();
#pragma unroll

    for (uint32_t index = tid; index < nRegionsH * 16 * nRegionsV * 16; index += blockDim.x * blockDim.y)
    {
        uint32_t fid = index >> 8;
        uint32_t fx = fid % nRegionsH;
        uint32_t fy = fid / nRegionsH;

        uint32_t regionCoord_x = (blockIdx.x) * nRegionsH;
        uint32_t regionCoord_y = (blockIdx.y) * nRegionsV;

        uint32_t globalFragment_x = regionCoord_x + fx + 1;
        uint32_t globalFragment_y = regionCoord_y + fy + 1;

        size_t dindex = (globalFragment_y * nWithHalo16 + globalFragment_x) * 256 + (index & 255);
        if (globalFragment_x < (nWithHalo16)-1 && globalFragment_y < (nWithHalo16)-1)
        {
            uint32_t val = __half2uint_rn(pDataOut[dindex]);
            float val2 = __half2float(pDataIn[dindex]);
            pDataOut[dindex] =
                __uint2half_rn(val2 * h(val - val2, SMIN, SMAX) + (1 - val2) * h(val - val2, BMIN, BMAX));
        }
    }
}

__global__ void convertFp32ToFp16(half *out, int *in, int nWithHalo)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo)
    {
        out[tx + ty * (size_t)nWithHalo] = __uint2half_rn(in[tx + ty * (size_t)nWithHalo]);
    }
}
__global__ void convertFp16ToFp32(int *out, half *in, int nWithHalo)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < nWithHalo && ty < nWithHalo)
    {
        out[tx + ty * (size_t)nWithHalo] = __half2uint_rn(in[tx + ty * (size_t)nWithHalo]);
    }
}

__global__ void convertFp32ToFp16AndDoChangeLayout(half *out, unsigned char *in, size_t nWithHalo)
{
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo)
    {
        out[bid * 256 + tid] = __uint2half_rd(in[ty * nWithHalo + tx]);
    }
}
__global__ void convertFp16ToFp32AndUndoChangeLayout(unsigned char *out, half *in, size_t nWithHalo)
{
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    if (tx < nWithHalo && ty < nWithHalo)
    {
        out[ty * nWithHalo + tx] = __half2uint_rn(in[bid * 256 + tid]);
    }
}

__global__ void UndoChangeLayout(unsigned char *out, unsigned char *in, size_t nWithHalo)
{
    uint32_t tx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ty = blockDim.y * blockIdx.y + threadIdx.y;
    size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.y * gridDim.x + blockIdx.x;

    // printf("%i, %i -> %i, %i\n", tx, ty, in_x, in_y);
    // printf("%llu -> %llu\n", tx + ty * nWithHalo, bid*256+tid);

    if (tx < nWithHalo && ty < nWithHalo)
    {
        out[ty * nWithHalo + tx] = in[bid * 1024 + tid];
    }
}

__global__ void copyHorizontalHalo(unsigned char *data, size_t n, size_t nWithHalo, int radius)
{
    // We want id ∈ [1,dim]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n)
    {
#pragma unroll
        for (int i = 0; i < radius; i++)
        {
            // Copy first real row to bottom ghost row
            data[(nWithHalo * (n + radius + i)) + (id + radius)] = data[(nWithHalo * (radius + i)) + id + radius];
            // Copy last real row to top ghost row
            data[nWithHalo * i + id + radius] = data[(nWithHalo) * (n + i) + id + radius];
        }
    }
}

__global__ void copyVerticalHalo(unsigned char *data, size_t n, size_t nWithHalo, int radius)
{
    // We want id ∈ [0,dim+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < nWithHalo)
    {
#pragma unroll
        for (int i = 0; i < radius; i++)
        {
            // Copy first real column to right most ghost column
            data[(id) * (nWithHalo) + (n + radius + i)] = data[(id) * (nWithHalo) + (radius + i)];
            // Copy last real column to left most ghost column
            data[(id) * (nWithHalo) + i] = data[(id) * (nWithHalo) + (n + i)];
        }
    }
}

__global__ void copyFromMTYPEAndCast(unsigned char *from, int *to, size_t nWithHalo)
{
    size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tid = tid_y * blockDim.x * gridDim.x + tid_x;
    for (size_t index = tid; index < nWithHalo * nWithHalo; index += blockDim.x * blockDim.y * gridDim.x * gridDim.y)
    {
        to[index] = (int)from[index];
    }
}
__global__ void copyToMTYPEAndCast(int *from, unsigned char *to, size_t nWithHalo)
{
    size_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t tid = tid_y * blockDim.x * gridDim.x + tid_x;
    for (size_t index = tid; index < nWithHalo * nWithHalo; index += blockDim.x * blockDim.y * gridDim.x * gridDim.y)
    {
        to[index] = (unsigned char)from[index];
    }
}

///////////////////////////////////////////////////////////
#define sh_row (size_t) threadIdx.y
#define sh_col ((size_t)threadIdx.x * cellsPerThread)
#define x2 ((size_t)x * cellsPerThread)
#define sh_size_x (blockDim.x * cellsPerThread)
__forceinline__ __device__ int count_neighs(unsigned char c, int my_id, int size_i, unsigned char *lattice, int neighs,
                                            int halo);

__global__ void MCELL_KERNEL(unsigned char *d_lattice, unsigned char *d_lattice_new, int size_i, int size_j,
                             int cellsPerThread, int neighs, int halo)
{

    const size_t totalShmem = ((blockDim.x * 2 + 2 * neighs) * (blockDim.y + 2 * neighs));
    const size_t sh_stride = ((blockDim.x * 2 + 2 * neighs));
    extern __shared__ unsigned char sh_lattice[];

    size_t global_id;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int blockStart_x = blockIdx.x * blockDim.x * 2;
    int blockStart_y = blockIdx.y * blockDim.y;

    for (int sh_id = tid; sh_id < totalShmem; sh_id += blockDim.x * blockDim.y)
    {
        int shmem_y = sh_id / (sh_stride);
        int shmem_x = sh_id % (sh_stride);

        global_id = (blockStart_y + shmem_y) * (size_t)(size_i + halo) + blockStart_x + shmem_x;
        if (blockStart_y + shmem_y < size_i + halo && blockStart_x + shmem_x < size_j + halo)
        {
            sh_lattice[sh_id] = d_lattice[global_id];
        }
    }

    __syncthreads();

    uint32_t subcell[2] = {0, 0};

    // col izq
    for (int ry = -neighs; ry <= neighs; ry++)
    {
        int y = threadIdx.y + ry + neighs;
        int x = threadIdx.x * 2;
        int sh_id = y * sh_stride + x;
        int c = sh_lattice[sh_id];
        subcell[0] += c;
    }

// centro comun
#pragma unroll
    for (int ry = -neighs; ry <= neighs; ry++)
    {
        for (int rx = -neighs + 1; rx <= neighs; rx++)
        {
            int y = threadIdx.y + ry + neighs;
            int x = threadIdx.x * 2 + rx + neighs;
            int sh_id = y * sh_stride + x;
            int c = sh_lattice[sh_id];
            subcell[0] += c;
            subcell[1] += c;
        }
    }

    // col der
    for (int ry = -neighs; ry <= neighs; ry++)
    {
        int y = threadIdx.y + ry + neighs;
        int x = threadIdx.x * 2 + 2 * neighs + 1;
        int sh_id = y * sh_stride + x;
        int c = sh_lattice[sh_id];
        subcell[1] += c;
    }

    int global_x = blockStart_x + threadIdx.x * 2 + neighs;
    int global_y = blockStart_y + threadIdx.y + neighs;

    if (global_x < size_j + neighs && global_y < size_i + neighs)
    {
        size_t my_id = global_y * (size_t)(size_i + halo) + global_x;
        int c = sh_lattice[(threadIdx.y + neighs) * sh_stride + (threadIdx.x * 2 + neighs)];
        int c2 = sh_lattice[(threadIdx.y + neighs) * sh_stride + (threadIdx.x * 2 + neighs + 1)];
        subcell[0] -= c;
        subcell[1] -= c2;
        d_lattice_new[my_id] = c * h(subcell[0], SMIN, SMAX) + (1 - c) * h(subcell[0], BMIN, BMAX);
        d_lattice_new[my_id + 1] = c2 * h(subcell[1], SMIN, SMAX) + (1 - c2) * h(subcell[1], BMIN, BMAX);
    }
}

#define NEIGHS1
__forceinline__ __device__ int count_neighs(unsigned char c, int my_id, int size_i, unsigned char *lattice, int neighs,
                                            int halo)
{
    size_t size = size_i + halo;
    int count = 0;

    for (int i = -neighs; i <= neighs; i++)
    {
#pragma unroll
        for (int j = -neighs; j <= neighs; j++)
        {
            count += lattice[my_id + i * size + j];
        }
    }
    count -= c;
    return count;
}

__global__ void copy_Rows(int size_i, unsigned char *d_lattice, int neighs, int halo)
{
    size_t my_id = (size_t)blockDim.x * blockIdx.x + threadIdx.x + neighs;
    int i = 0;
    size_t size = size_i + halo;

    if (my_id < (size_i + neighs))
    {
        for (i = 0; i < neighs; i++)
        {
            d_lattice[size * (size_i + (i + neighs)) + my_id] =
                d_lattice[(i + neighs) * size + my_id];                           // copia primeras filas en ultimas
            d_lattice[i * size + my_id] = d_lattice[size * (size_i + i) + my_id]; // copia ultimas filas en primeras
        }
    }
}

__global__ void copy_Cols(int size_i, unsigned char *d_lattice, int neighs, int halo)
{
    size_t my_id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    int i = 0;
    // Al haber copiado la primer fila en la ultima columna, se puede directamente copiar la primer columna completa,
    // incluidas las ghost cells, en la ultima columna ghost, y las esquinas van a tener el valor apropiado, la esquina
    // diagonal opuesta.
    size_t size = size_i + halo;

    if (my_id < size)
    {
        for (i = 0; i < neighs; i++)
        {
            d_lattice[my_id * size + (size_i + (i + neighs))] =
                d_lattice[my_id * size + (i + neighs)];                           // copia primeras columnas en ultimas
            d_lattice[my_id * size + i] = d_lattice[my_id * size + (size_i + i)]; // copia ultimas columnas en primeras
        }
    }
}

#define my_id_topa ((size_t)y * (size_i + halo) + x)
#define col_topa (threadIdx.x + neighs)
#define row_topa (threadIdx.y + neighs)
#define my_sh_id_topa ((size_t)(row_topa) * (blockDim.x + halo) + (col_topa))
#define row_topa2 (warpId + neighs)

__global__ void SHARED_KERNEL(unsigned char *d_lattice, unsigned char *d_lattice_new, int size_i, int size_j,
                              int neighs, int halo)
{
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    int count = 0;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x + neighs;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y + neighs;
    int v = 0;

    extern __shared__ unsigned char sh_lattice[];

    // interior
    if (y < size_i + neighs && x < size_j + neighs)
    {
        sh_lattice[my_sh_id_topa] = d_lattice[my_id_topa];
    }
    // halo
    size_t y2 = blockDim.y * blockIdx.y + warpId + neighs;
    // y= blockDim.y * blockIdx.y + warpId + neighs;
    if (warpId == 0 || warpId == 1)
    {
        for (v = 0; v < neighs; v++)
        {
            int gy = y2 - ((row_topa2)-neighs);
            size_t up_or_down = ((blockDim.x + neighs) * ((row_topa2)-neighs)) + v;

            sh_lattice[(up_or_down) * (blockDim.x + halo) + col_topa] =
                d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + x];
            // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row,col,   up_or_down,col, d_lattice[(gy - neighs
            // + (up_or_down)) * (size_i + halo) + x]);

            // Corner Halos: left-up and left-down
            if ((col_topa - neighs) < neighs)
            {
                sh_lattice[(up_or_down) * (blockDim.x + halo) + (col_topa - neighs)] =
                    d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + (x - neighs)];
                // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row, col,  up_or_down, col-neighs,
                // d_lattice[(gy - neighs + (up_or_down)) * (size_i + halo) + (x-neighs)]);
            }

            // Corner Halos: right-up and right-down
            if ((col_topa + neighs) >= blockDim.y + neighs)
            {
                sh_lattice[(up_or_down) * (blockDim.x + halo) + (col_topa + neighs)] =
                    d_lattice[(gy - neighs + up_or_down) * (size_i + halo) + (x + neighs)];
                // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, row, col,  up_or_down, col+neighs,
                // sh_lattice[(up_or_down) * (blockDim.x+halo) + (col+neighs)] );
            }
        }
    }
    else if (warpId == 2 || warpId == 3)
    {
        for (v = 0; v < neighs; v++)
        {
            int gy = y2 - ((row_topa2)-neighs);
            int gx = x - (col_topa - neighs);
            int lr = ((blockDim.y + neighs) * ((row_topa2) & 1)) + v;

            // printf("row=%d v=%d -- (%d,%d)-> (%d,%d)=%d\n",row, v, col, row,  col, lr, d_lattice[(gx - neighs + lr) +
            // (gy + (col-neighs)) * (size_i + halo)]);
            sh_lattice[col_topa * (blockDim.x + halo) + lr] =
                d_lattice[(gx - neighs + lr) + (gy + (col_topa - neighs)) * (size_i + halo)];
        }
    }

    __syncthreads();

    if (x < size_i + neighs && y < size_j + neighs)
    {
        // if (i <= size_i && j <= size_j && (ii-1) != 0 && (ii-1) != blockDim.x && (jj-1) != 0 && (jj-1) != blockDim.y)
        // {
        unsigned char c = sh_lattice[my_sh_id_topa];

        count = count_neighs(
            c, my_sh_id_topa, blockDim.x, sh_lattice, neighs,
            halo); // decrease sh_size_x by 2 to use the same count_neighs function than the rest of the implementations
        d_lattice_new[my_id_topa] = c * h(count, SMIN, SMAX) + (1 - c) * h(count, BMIN, BMAX);
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_lookup_table(int *GPU_lookup_table, int radius)
{
    int stride = (radius * 2 + 1);
    int *lookup_table = GPU_lookup_table;

    if (threadIdx.y < 2 && blockIdx.x < (stride * stride))
    {
        if (threadIdx.y == 0)
            if (blockIdx.x >= BMIN && blockIdx.x <= BMAX)
                lookup_table[threadIdx.y * stride + blockIdx.x] = 1;
            else
                lookup_table[threadIdx.y * stride + blockIdx.x] = 0;

        if (threadIdx.y == 1)
            if (blockIdx.x >= SMIN && blockIdx.x <= SMAX)
                lookup_table[threadIdx.y * stride + blockIdx.x] = 1;
            else
                lookup_table[threadIdx.y * stride + blockIdx.x] = 0;
    }
}

__global__ void ghostRows(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize)
{
    // We want id ∈ [1,GRID_SIZE]
    size_t my_id = blockDim.x * blockIdx.x + threadIdx.x + horizontalHaloWidth;
    int fullHorizontalSize = ROW_SIZE + 2 * horizontalHaloWidth;

    if (my_id < (ROW_SIZE + horizontalHaloWidth))
    {
        for (int currentHalo = 0; currentHalo < verticalHaloSize; currentHalo++)
        {
            // fill bottom halo
            grid[(currentHalo + verticalHaloSize + GRID_SIZE) * fullHorizontalSize + my_id] =
                grid[(currentHalo + verticalHaloSize) * fullHorizontalSize + my_id];

            // fill top halo
            grid[currentHalo * fullHorizontalSize + my_id] =
                grid[(currentHalo + GRID_SIZE) * fullHorizontalSize + my_id];
        }
    }
}

__global__ void ghostCols(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize)
{
    // We want id ∈ [0,SIZE+1]
    size_t my_id = blockDim.x * blockIdx.x + threadIdx.x;
    int fullHorizontalSize = ROW_SIZE + 2 * horizontalHaloWidth;
    int fullVerticalSize = GRID_SIZE + 2 * verticalHaloSize;

    if (my_id < fullVerticalSize)
    {
        for (int currentHalo = 0; currentHalo < horizontalHaloWidth; currentHalo++)
        {
            // Copy first real column to right most ghost column
            grid[(my_id) * (fullHorizontalSize) + horizontalHaloWidth + currentHalo + ROW_SIZE] =
                grid[(my_id) * (fullHorizontalSize) + horizontalHaloWidth + currentHalo];
            // Copy last real column to left most ghost column
            grid[my_id * (fullHorizontalSize) + currentHalo] =
                grid[my_id * (fullHorizontalSize) + currentHalo + ROW_SIZE];
        }
    }
}

__device__ inline int dist(int x0, int x1)
{
    return abs(x0 - x1);
}

// Original CAGIGAS code for r=1
__global__ void CAGIGAS_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE)
{
    // We want id ∈ [1,SIZE]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (ROW_SIZE + 2) + ix;
    uint64_t cell, new_cell = 0;
    uint64_t up_cell, down_cell, right_cell, left_cell;                // Up,down,right,left cells
    uint64_t upleft_cell, downleft_cell, upright_cell, downright_cell; // Diagonal cells
    unsigned char subcell;

    int k, numNeighbors;
    int(*lookup_table)[CELL_NEIGHBOURS + 1] = (int(*)[CELL_NEIGHBOURS + 1]) GPU_lookup_table;

    if (iy > 0 && iy <= GRID_SIZE && ix > 0 && ix <= ROW_SIZE)
    {
        cell = grid[id];

        // First (0) subcell:
        up_cell = grid[id - (ROW_SIZE + 2)];
        down_cell = grid[id + (ROW_SIZE + 2)];
        left_cell = grid[id - 1];
        upleft_cell = grid[id - (ROW_SIZE + 3)];
        downleft_cell = grid[id + (ROW_SIZE + 1)];

        numNeighbors = getSubCellD(up_cell, 0) + getSubCellD(down_cell, 0);                  // upper lower
        numNeighbors += getSubCellD(left_cell, 8 - 1) + getSubCellD(cell, 1);                // left right
        numNeighbors += getSubCellD(upleft_cell, 8 - 1) + getSubCellD(downleft_cell, 8 - 1); // diagonals left
        numNeighbors += getSubCellD(up_cell, 1) + getSubCellD(down_cell, 1);                 // diagonals right
        subcell = getSubCellD(cell, 0);
        setSubCellD(&new_cell, 0, lookup_table[subcell][numNeighbors]);

        // Middle subcells:
        for (k = 1; k < CELL_NEIGHBOURS - 1; k++)
        {
            numNeighbors = getSubCellD(up_cell, k) + getSubCellD(down_cell, k);          // upper lower
            numNeighbors += getSubCellD(cell, k - 1) + getSubCellD(cell, k + 1);         // left right
            numNeighbors += getSubCellD(up_cell, k - 1) + getSubCellD(down_cell, k - 1); // diagonals left
            numNeighbors += getSubCellD(up_cell, k + 1) + getSubCellD(down_cell, k + 1); // diagonals right
            subcell = getSubCellD(cell, k);
            setSubCellD(&new_cell, k, lookup_table[subcell][numNeighbors]);
        }

        // Last (CELL_NEIGHBOURS-1) subcell:
        right_cell = grid[id + 1];
        upright_cell = grid[id - (ROW_SIZE + 1)];
        downright_cell = grid[id + (ROW_SIZE + 3)];

        numNeighbors = getSubCellD(up_cell, 8 - 1) + getSubCellD(down_cell, 8 - 1);    // upper lower
        numNeighbors += getSubCellD(cell, 8 - 2) + getSubCellD(right_cell, 0);         // left right
        numNeighbors += getSubCellD(up_cell, 8 - 2) + getSubCellD(down_cell, 8 - 2);   // diagonals left
        numNeighbors += getSubCellD(upright_cell, 0) + getSubCellD(downright_cell, 0); // diagonals right
        subcell = getSubCellD(cell, 8 - 1);
        setSubCellD(&new_cell, 8 - 1, lookup_table[subcell][numNeighbors]);

        // Copy new_cell to newGrid:
        newGrid[id] = new_cell;
    }
}

__global__ void PACK_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE,
                            int horizontalHaloWidth, int verticalHaloSize, int radius)
{
    // int cagigas_cell_neigh = ((radius * 2 + 1) * (radius * 2 + 1) - 1);
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
    uint64_t new_cell = 0;
    // unsigned char subcell;
    int *lookup_table = GPU_lookup_table;
    int stride = (radius * 2 + 1);
    extern __shared__ uint64_t sh_grid[];
    int blockStart_x = blockIdx.x * blockDim.x;
    int blockStart_y = blockIdx.y * blockDim.y;

    for (int i = threadIdx.y; i < blockDim.y + 2 * verticalHaloSize; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < blockDim.x + 2 * horizontalHaloWidth; j += blockDim.x)
        {
            if ((blockStart_y + i) < fullVerticalSize && blockStart_x + j < fullHorizontalSize)
            {
                sh_grid[i * (blockDim.x + 2 * horizontalHaloWidth) + j] =
                    grid[(blockStart_y + i) * fullHorizontalSize + blockStart_x + j];
            }
        }
    }
    __syncthreads();

    uint32_t subcells[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // unsigned char subcells[8] = {0,0,0,0,0,0,0,0};
    uint64_t threadWord = sh_grid[sh_id];
    uint64_t left[2] = {0, 0};
    uint64_t right[2] = {0, 0};
    if (iy >= verticalHaloSize && iy < GRID_SIZE + verticalHaloSize && ix >= horizontalHaloWidth &&
        ix < ROW_SIZE + horizontalHaloWidth)
    {
        for (int i = -radius; i <= radius; i++)
        {
            int currentNeighPosition_y = threadIdx.y + verticalHaloSize + i;
            int currentNeighPosition_x = (threadIdx.x + horizontalHaloWidth);
            int currentNeighCellIndex = currentNeighPosition_y * fullSharedWidth + currentNeighPosition_x;
            // read the corresponding 64-bit words from x-neighborhood, once per i-row
            uint64_t centerWord = sh_grid[currentNeighCellIndex];
            left[0] = sh_grid[currentNeighCellIndex - 1];
            right[0] = sh_grid[currentNeighCellIndex + 1];
#if RADIUS > 8
            left[1] = sh_grid[currentNeighCellIndex - 2];
            right[1] = sh_grid[currentNeighCellIndex + 2];
#endif

// LEFT LOOP
#pragma unroll
            for (int j = -radius; j < 0; j++)
            {
                int currentNeighSubcellIndex = (j) & (8 - 1);
                uint64_t currentNeighCell = left[((-j) - 1) >> 3];
                unsigned char subcell = getSubCellD(currentNeighCell, currentNeighSubcellIndex);
                int from = max(0, j - radius);
                int to = min(7, j + radius);
                for (int k = from; k <= to; k++)
                {
                    subcells[k] += subcell;
                }
            }

// CENTER LOOP
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                unsigned char subcell = getSubCellD(centerWord, j);
                if (i == 0)
                {
                    int from = max(0, j - radius);
                    int to = j - 1;
                    for (int k = from; k <= to; k++)
                    {
                        subcells[k] += subcell;
                    }
                    from = j + 1;
                    to = min(7, j + radius);
                    for (int k = from; k <= to; k++)
                    {
                        subcells[k] += subcell;
                    }
                }
                else
                {
                    int from = max(0, j - radius);
                    int to = min(7, j + radius);
                    for (int k = from; k <= to; k++)
                    {
                        subcells[k] += subcell;
                    }
                }
            }
// RIGHT LOOP
#pragma unroll
            for (int j = 8; j < 8 + radius; j++)
            {
                int currentNeighSubcellIndex = (j) & (8 - 1);
                uint64_t currentNeighCell = right[(j - 8) >> 3];
                unsigned char subcell = getSubCellD(currentNeighCell, currentNeighSubcellIndex);
                int from = max(0, j - radius);
                int to = min(7, j + radius);
                for (int k = from; k <= to; k++)
                {
                    subcells[k] += subcell;
                }
            }
        }
// TRANSITION STATES
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            setSubCellD(&new_cell, i, lookup_table[getSubCellD(threadWord, i) * stride + subcells[i]]);
        }
        // WRITE NEW 64-bit WORD
        newGrid[id] = new_cell;
    }
}

__forceinline__ unsigned char getSubCellH(uint64_t cell, unsigned char pos)
{
    return (cell >> (8 - 1 - pos) * 8);
}

__forceinline__ void setSubCellH(uint64_t *cell, unsigned char pos, unsigned char subcell)
{
    uint64_t mask = 0xFF;
    uint64_t maskNewCell = subcell;

    // Erase pos content in cell:
    mask = mask << (8 - 1 - pos) * 8;
    mask = ~mask;
    *cell = *cell & mask;

    // Add subcell content to cell in pos:
    *cell = *cell | (maskNewCell << (8 - 1 - pos) * 8);
}

__device__ unsigned char getSubCellD(uint64_t cell, unsigned char pos)
{
    return (cell >> (8 - 1 - pos) * 8);
}

__device__ void setSubCellD(uint64_t *cell, unsigned char pos, unsigned char subcell)
{
    uint64_t mask = 0xFF;
    uint64_t maskNewCell = subcell;

    // Erase pos content in cell:
    mask = mask << (8 - 1 - pos) * 8;
    mask = ~mask;
    *cell = *cell & mask;

    // Add subcell content to cell in pos:
    *cell = *cell | (maskNewCell << (8 - 1 - pos) * 8);
}

__global__ void unpackStateKernel(uint64_t *from, unsigned char *to, int ROW_SIZE, int GRID_SIZE,
                                  int horizontalHaloWidth, int verticalHaloSize)
{
    // We want id ∈ [1,SIZE]
    size_t unpacked_x = (blockDim.x * blockIdx.x + threadIdx.x) * 8 + verticalHaloSize;
    size_t unpacked_y = blockDim.y * blockIdx.y + threadIdx.y + verticalHaloSize;

    size_t packed_x = (blockDim.x * blockIdx.x + threadIdx.x) + horizontalHaloWidth;

    size_t unpackedIndex = unpacked_y * (GRID_SIZE + 2 * verticalHaloSize) + unpacked_x;
    size_t packedIndex = unpacked_y * (ROW_SIZE + 2 * horizontalHaloWidth) + packed_x;
    // print all i in one line

    uint64_t cellValue;
    unsigned char subcell;

    if (unpacked_y < GRID_SIZE + verticalHaloSize && unpacked_x < GRID_SIZE + verticalHaloSize)
    {
        cellValue = from[packedIndex];
        for (int i = 0; i < 8; i++)
        {
            subcell = getSubCellD(cellValue, i);
            to[unpackedIndex + i] = subcell;
        }
    }
}

__global__ void packStateKernel(unsigned char *from, uint64_t *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                                int verticalHaloSize)
{

    // We want id ∈ [1,SIZE]
    size_t unpacked_x = (blockDim.x * blockIdx.x + threadIdx.x) * 8 + verticalHaloSize;
    size_t unpacked_y = blockDim.y * blockIdx.y + threadIdx.y + verticalHaloSize;

    size_t packed_x = (blockDim.x * blockIdx.x + threadIdx.x) + horizontalHaloWidth;

    size_t unpackedIndex = unpacked_y * (GRID_SIZE + 2 * verticalHaloSize) + unpacked_x;
    size_t packedIndex = unpacked_y * (ROW_SIZE + 2 * horizontalHaloWidth) + packed_x;
    // print all i in one line
    uint64_t cellValue = 0;
    // if (threadIdx.x + threadIdx.y == 0) {
    //     for (int i = 0; i < GRID_SIZE + 2 * verticalHaloSize; i++) {
    //         for (int j = 0; j < GRID_SIZE + 2 * verticalHaloSize; j++) {
    //             printf("%d ", from[i * (GRID_SIZE + 2 * verticalHaloSize) + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // printf("(%d, %d) = %d\n", unpacked_x, unpacked_y, unpackedIndex);
    // printf("t(%i, %i) -> unpack(%i, %i) -> GRID_SIZE:%i,  verticalHalo:%i, total_width:%i\n", threadIdx.x,
    // threadIdx.y, unpacked_x, unpacked_y, GRID_SIZE, verticalHaloSize, GRID_SIZE + 2 * verticalHaloSize);
    if (unpacked_y < GRID_SIZE + verticalHaloSize && unpacked_x < GRID_SIZE + verticalHaloSize)
    {

        for (int i = 0; i < 8; i++)
        {
            // unsigned char subcell = getSubCellD(cellValue, i);
            // printf("i,j = %llu, %llu = %i\n", unpacked_y, unpacked_x+i, subcell);
            setSubCellD(&cellValue, i, from[unpackedIndex + i]);
        }
        to[packedIndex] = cellValue;
        // printf("%d, %d -> cellValue=%lx\n", unpacked_y, unpacked_x, cellValue);
    }
}

#endif
