#pragma once

#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define HINDEX(x, y, radius, nWithHalo) ((y + radius) * ((size_t)nWithHalo) + (x + radius))
#define GINDEX(x, y, nshmem) ((y) * (nshmem) + (x))

__device__ inline int h(int k, int a, int b);

__forceinline__ __device__ void workWithShmem(char *pDataOut, char *shmem, uint2 dataCoord, uint32_t nWithHalo,
                                              uint32_t nShmem);

__forceinline__ __device__ void workWithGbmem(char *pDataIn, char *pDataOut, uint2 dataCoord, uint32_t nWithHalo);

__global__ void BASE_KERNEL(char *pDataIn, char *pDataOut, size_t n, size_t nWithHalo);

__global__ void COARSE_KERNEL(char *pDataIn, char *pDataOut, size_t n, size_t nWithHalo);

__global__ void CAT_KERNEL(half *pDataIn, half *pDataOut, size_t n, size_t nWithHalo);

__global__ void convertFp32ToFp16(half *out, int *in, int nWithHalo);
__global__ void convertFp16ToFp32(int *out, half *in, int nWithHalo);

__global__ void convertFp32ToFp16AndDoChangeLayout(half *out, int *in, size_t nWithHalo);
__global__ void convertFp16ToFp32AndUndoChangeLayout(int *out, half *in, size_t nWithHalo);

__global__ void convertUInt32ToUInt4AndDoChangeLayout(int *out, char *in, size_t nWithHalo);
__global__ void convertUInt4ToUInt32AndUndoChangeLayout(char *out, int *in, size_t nWithHalo);
__global__ void UndoChangeLayout(char *out, char *in, size_t nWithHalo);

__global__ void onlyConvertUInt32ToUInt4(int *out, char *in, size_t nWithHalo);
__global__ void convertInt32ToInt8AndDoChangeLayout(unsigned char *out, int *in, size_t nWithHalo);
__global__ void convertInt8ToInt32AndUndoChangeLayout(int *out, unsigned char *in, size_t nWithHalo);

__global__ void copyHorizontalHalo(char *data, size_t n, size_t nWithHalo);
__global__ void copyVerticalHalo(char *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloCoalescedVersion(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloCoalescedVersion(half *data, size_t n, size_t nWithHalo);
__global__ void copyHorizontalHaloHalf(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloHalf(half *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloTensor(half *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloTensor(half *data, size_t n, size_t nWithHalo);

__global__ void copyFromMTYPEAndCast(char *from, int *to, size_t nWithHalo);
__global__ void copyToMTYPEAndCast(int *from, char *to, size_t nWithHalo);

__forceinline__ __device__ int count_neighs(int my_id, int size_i, char *lattice, int neighs, int halo);
__global__ void MCELL_KERNEL(char *d_lattice, char *d_lattice_new, int size_i, int size_j, int cellsPerThread,
                             int neighs, int halo);
__global__ void copy_Rows(int size_i, char *d_lattice, int neighs, int halo);
__global__ void copy_Cols(int size_i, char *d_lattice, int neighs, int halo);

__global__ void SHARED_KERNEL(char *d_lattice, char *d_lattice_new, int size_i, int size_j, int neighs, int halo);

__global__ void kernel_init_lookup_table(int *GPU_lookup_table);

//////////////////////////
//////////////////////////
//////////////////////////

__global__ void ghostRows(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void ghostCols(uint64_t *grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void CAGIGAS_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE);
__global__ void PACK_KERNEL(uint64_t *grid, uint64_t *newGrid, int *GPU_lookup_table, int ROW_SIZE, int GRID_SIZE,
                            int horizontalHaloWidth, int verticalHaloSize);

__forceinline__ unsigned char getSubCellH(uint64_t cell, char pos);

__forceinline__ void setSubCellH(uint64_t *cell, char pos, unsigned char subcell);

__device__ unsigned char getSubCellD(uint64_t cell, char pos);

__device__ void setSubCellD(uint64_t *cell, char pos, unsigned char subcell);

__global__ void unpackStateKernel(uint64_t *from, int *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                                  int verticalHaloSize);
__global__ void packStateKernel(int *from, uint64_t *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                                int verticalHaloSize);
