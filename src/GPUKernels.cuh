#pragma once

#include "Defines.h"

#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

#define SHMEM_N (BSIZE3DX + HALO_SIZE)
#define BMAXLLSHMEM_N (80 + HALO_SIZE)

#define HINDEX(x, y, nWithHalo) ((y + RADIUS) * ((size_t)nWithHalo) + (x + RADIUS))
#define GINDEX(x, y, nshmem) ((y) * (nshmem) + (x))

__device__ inline int h(int k, int a, int b);

__forceinline__ __device__ void workWithShmem(MTYPE *pDataOut, MTYPE *shmem, uint2 dataCoord, uint32_t nWithHalo,
                                              uint32_t nShmem);

__forceinline__ __device__ void workWithGbmem(MTYPE *pDataIn, MTYPE *pDataOut, uint2 dataCoord, uint32_t nWithHalo);

__global__ void BASE_KERNEL(MTYPE *pDataIn, MTYPE *pDataOut, size_t n, size_t nWithHalo);

__global__ void COARSE_KERNEL(MTYPE *pDataIn, MTYPE *pDataOut, size_t n, size_t nWithHalo);

__global__ void CAT_KERNEL(FTYPE *pDataIn, FTYPE *pDataOut, size_t n, size_t nWithHalo);

__global__ void convertFp32ToFp16(FTYPE *out, int *in, int nWithHalo);
__global__ void convertFp16ToFp32(int *out, FTYPE *in, int nWithHalo);

__global__ void convertFp32ToFp16AndDoChangeLayout(FTYPE *out, int *in, size_t nWithHalo);
__global__ void convertFp16ToFp32AndUndoChangeLayout(int *out, FTYPE *in, size_t nWithHalo);

__global__ void convertUInt32ToUInt4AndDoChangeLayout(int *out, MTYPE *in, size_t nWithHalo);
__global__ void convertUInt4ToUInt32AndUndoChangeLayout(MTYPE *out, int *in, size_t nWithHalo);
__global__ void UndoChangeLayout(MTYPE *out, MTYPE *in, size_t nWithHalo);

__global__ void onlyConvertUInt32ToUInt4(int *out, MTYPE *in, size_t nWithHalo);
__global__ void convertInt32ToInt8AndDoChangeLayout(unsigned char *out, int *in, size_t nWithHalo);
__global__ void convertInt8ToInt32AndUndoChangeLayout(int *out, unsigned char *in, size_t nWithHalo);

__global__ void copyHorizontalHalo(MTYPE *data, size_t n, size_t nWithHalo);
__global__ void copyVerticalHalo(MTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloCoalescedVersion(FTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloCoalescedVersion(FTYPE *data, size_t n, size_t nWithHalo);
__global__ void copyHorizontalHaloHalf(FTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloHalf(FTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloTensor(FTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloTensor(FTYPE *data, size_t n, size_t nWithHalo);

__global__ void copyFromMTYPEAndCast(MTYPE *from, int *to, size_t nWithHalo);
__global__ void copyToMTYPEAndCast(int *from, MTYPE *to, size_t nWithHalo);

__forceinline__ __device__ int count_neighs(int my_id, int size_i, MTYPE *lattice, int neighs, int halo);
__global__ void MCELL_KERNEL(MTYPE *d_lattice, MTYPE *d_lattice_new, int size_i, int size_j, int cellsPerThread,
                             int neighs, int halo);
__global__ void copy_Rows(int size_i, MTYPE *d_lattice, int neighs, int halo);
__global__ void copy_Cols(int size_i, MTYPE *d_lattice, int neighs, int halo);

__global__ void SHARED_KERNEL(MTYPE *d_lattice, MTYPE *d_lattice_new, int size_i, int size_j, int neighs, int halo);

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

__global__ void unpackState(uint64_t *from, int *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                            int verticalHaloSize);
__global__ void packState(int *from, uint64_t *to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth,
                          int verticalHaloSize);
