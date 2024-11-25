#pragma once

#include "Defines.h"
#include <cuda.h>
#include <mma.h>

#define HINDEX(x, y, nWithHalo, radius) ((y + radius) * ((size_t)nWithHalo) + (x + radius))
#define GINDEX(x, y, nshmem) ((y) * (nshmem) + (x))

__device__ inline int h(int k, int a, int b);

__global__ void BASE_KERNEL(unsigned char *pDataIn[], unsigned char *pDataOut[], size_t n, int halo, int radius, int SMIN, int SMAX, int BMIN, int BMAX);

__global__ void COARSE_KERNEL(unsigned char *pDataIn[], unsigned char *pDataOut[], size_t n, int halo, int radius, int SMIN, int SMAX, int BMIN, int BMAX);

__global__ void CAT_KERNEL(half *pDataIn[], half *pDataOut[], size_t n, int halo, int radius, int nRegionsH,
                           int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX);

__global__ void convertFp32ToFp16AndDoChangeLayout(half *out[], unsigned char *in[], size_t n, int halo);
__global__ void convertFp16ToFp32AndUndoChangeLayout(unsigned char *out[], half *in[], size_t n, int halo);

__global__ void UndoChangeLayout(unsigned char *out, unsigned char *in, size_t nWithHalo);

__global__ void copyHorizontalHalo(unsigned char *data, size_t n, size_t nWithHalo);
__global__ void copyVerticalHalo(unsigned char *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloCoalescedVersion(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloCoalescedVersion(half *data, size_t n, size_t nWithHalo);
__global__ void copyHorizontalHaloHalf(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloHalf(half *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloTensor(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloTensor(half *data, size_t n, size_t nWithHalo);

__global__ void copyFromMTYPEAndCast(unsigned char *from, int *to, size_t nWithHalo);
__global__ void copyToMTYPEAndCast(int *from, unsigned char *to, size_t nWithHalo);

__forceinline__ __device__ int count_neighs(int my_id, int size_i, unsigned char *lattice, int neighs, int halo);
__global__ void MCELL_KERNEL(unsigned char *d_lattice[], unsigned char *d_lattice_new[], int size_i, int size_j,
                             int cellsPerThread, int neighs, int halo, int SMIN, int SMAX, int BMIN, int BMAX);
__global__ void copy_Rows(int size_i, unsigned char *d_lattice, int neighs, int halo);
__global__ void copy_Cols(int size_i, unsigned char *d_lattice, int neighs, int halo);

__global__ void SHARED_KERNEL(unsigned char *d_lattice[], unsigned char *d_lattice_new[], int size_i, int size_j,
                              int neighs, int halo, int SMIN, int SMAX, int BMIN, int BMAX);

__global__ void kernel_init_lookup_table(int *GPU_lookup_table, int radius, int SMIN, int SMAX, int BMIN, int BMAX);

//////////////////////////
//////////////////////////
//////////////////////////

__global__ void ghostRows(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void ghostCols(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void CAGIGAS_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE);
__global__ void PACK_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE,
                            int horizontalHaloWidth, int verticalHaloSize, int radius);

__forceinline__ unsigned char getSubCellH(uint64_t cell, unsigned char pos);

__forceinline__ void setSubCellH(uint64_t *cell, unsigned char pos, unsigned char subcell);

__device__ unsigned char getSubCellD(uint64_t cell, unsigned char pos);

__device__ void setSubCellD(uint64_t *cell, unsigned char pos, unsigned char subcell);

__global__ void unpackStateKernel(uint64_t *from, unsigned char *to, int ROW_SIZE, int GRID_SIZE,
                                  int horizontalHaloWidth, int verticalHaloSize);
__global__ void packStateKernel(unsigned char *from, uint64_t *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                                int verticalHaloSize);
