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

__forceinline__ __device__ void workWithShmem(MTYPE* pDataOut, MTYPE* shmem, uint2 dataCoord, uint32_t nWithHalo, uint32_t nShmem);

__forceinline__ __device__ void workWithGbmem(MTYPE* pDataIn, MTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo);

__global__ void ClassicGlobalMemGoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo);

__forceinline__ __device__ void workWithGbmemHALF(FTYPE* pDataIn, FTYPE* pDataOut, uint2 dataCoord, uint32_t nWithHalo);

__global__ void ClassicGlobalMemHALFGoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);

__global__ void ClassicV1GoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo);

__global__ void ClassicV2GoLStep(MTYPE* pDataIn, MTYPE* pDataOut, size_t n, size_t nWithHalo);

// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorV1GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);
// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorCoalescedV1GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);
// This kernel assumes that pDataIn has a halo of width=16 (fragment size)
__global__ void TensorCoalescedV2GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);
static inline bool is_aligned(const void* pointer, size_t byte_count);

__global__ void TensorCoalescedV3GoLStep(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);

__global__ void TensorCoalescedV4GoLStep_Step1(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);

__global__ void TensorCoalescedV4GoLStep_Step2(FTYPE* pDataIn, FTYPE* pDataOut, size_t n, size_t nWithHalo);

__device__ __inline__ uint32_t addInt4(int i, char int4index, int* shmem);

__device__ __inline__ uint32_t addInt4left(int i, char int4index, int* shmem);
__device__ __inline__ uint32_t addInt4right(int i, char int4index, int* shmem);
__global__ void TensorCoalescedSubTypeGoLStep(int* pDataIn, size_t n, size_t nWithHalo, MTYPE* buffer);
__global__ void convertFp32ToFp16(FTYPE* out, int* in, int nWithHalo);
__global__ void convertFp16ToFp32(int* out, FTYPE* in, int nWithHalo);

__global__ void convertFp32ToFp16AndDoChangeLayout(FTYPE* out, int* in, size_t nWithHalo);
__global__ void convertFp16ToFp32AndUndoChangeLayout(int* out, FTYPE* in, size_t nWithHalo);

__global__ void convertUInt32ToUInt4AndDoChangeLayout(int* out, MTYPE* in, size_t nWithHalo);
__global__ void convertUInt4ToUInt32AndUndoChangeLayout(MTYPE* out, int* in, size_t nWithHalo);
__global__ void UndoChangeLayout(MTYPE* out, MTYPE* in, size_t nWithHalo);

__global__ void onlyConvertUInt32ToUInt4(int* out, MTYPE* in, size_t nWithHalo);
__global__ void convertInt32ToInt8AndDoChangeLayout(unsigned char* out, int* in, size_t nWithHalo);
__global__ void convertInt8ToInt32AndUndoChangeLayout(int* out, unsigned char* in, size_t nWithHalo);

__global__ void TensorCoalescedInt8(unsigned char* pDataIn, unsigned char* pDataOut, size_t n, size_t nWithHalo);
__global__ void copyHorizontalHalo(MTYPE* data, size_t n, size_t nWithHalo);
__global__ void copyVerticalHalo(MTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloCoalescedVersion(FTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloCoalescedVersion(FTYPE* data, size_t n, size_t nWithHalo);
__global__ void copyHorizontalHaloHalf(FTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloHalf(FTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyHorizontalHaloTensor(FTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyVerticalHaloTensor(FTYPE* data, size_t n, size_t nWithHalo);

__global__ void copyFromMTYPEAndCast(MTYPE* from, int* to, size_t nWithHalo);
__global__ void copyToMTYPEAndCast(int* from, MTYPE* to, size_t nWithHalo);

__forceinline__ __device__ int count_neighs(int my_id, int size_i, MTYPE* lattice, int neighs, int halo);
__global__ void moveKernel(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int cellsPerThread, int neighs, int halo);
__global__ void moveKernel2(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int cellsPerThread, int neighs, int halo);
__global__ void moveKernel3(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int cellsPerThread, int neighs, int halo);

__global__ void copy_Rows(int size_i, MTYPE* d_lattice, int neighs, int halo);
__global__ void copy_Cols(int size_i, MTYPE* d_lattice, int neighs, int halo);

__global__ void moveKernelTopa(MTYPE* d_lattice, MTYPE* d_lattice_new, int size_i, int size_j, int neighs, int halo);

__global__ void kernel_init_lookup_table(int* GPU_lookup_table);

//////////////////////////
//////////////////////////
//////////////////////////

__global__ void ghostRows(uint64_t* grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void ghostCols(uint64_t* grid, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__global__ void GOL(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);
__global__ void GOLr1(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE);
__global__ void GOL33_gm(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);
__global__ void GOL33_sm(uint64_t* grid, uint64_t* newGrid, int* GPU_lookup_table, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);

__forceinline__ unsigned char getSubCellH(uint64_t cell, char pos);

__forceinline__ void setSubCellH(uint64_t* cell, char pos, unsigned char subcell);

__device__ unsigned char getSubCellD(uint64_t cell, char pos);

__device__ void setSubCellD(uint64_t* cell, char pos, unsigned char subcell);

__global__ void unpackState(uint64_t* from, int* to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);
__global__ void packState(int* from, uint64_t* to, int ROW_SIZE, int GRID_SIZE, int horizontalHaloWidth, int verticalHaloSize);
