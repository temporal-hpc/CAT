#ifndef _CLASSIC_GOL_KERNELS_H_
#define _CLASSIC_GOL_KERNELS_H_

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

#endif