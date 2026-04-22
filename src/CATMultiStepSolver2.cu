#include "CATMultiStepSolver2.cuh"
#include "GPUKernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

using namespace Temporal;

CATMultiStepSolver2::CATMultiStepSolver2(int nRegionsH, int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX, size_t n, size_t halo) 
    : Solver(SMIN, SMAX, BMIN, BMAX),
      m_l2CacheSize(0),
      m_maxPersistingL2(0),
      m_l2PersistenceOn(false),
      m_stream(nullptr)
{
    this->m_nRegionsH = nRegionsH;
    this->m_nRegionsV = nRegionsV;
    this->m_sharedMemoryBytes =
        ((nRegionsH + 2) * (nRegionsV + 2) * 16 * 24 + 384 * 2) * sizeof(half);

    // Allow large dynamic shared memory
    cudaFuncSetAttribute(CAT_KERNEL_CG3,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         this->m_sharedMemoryBytes);

    if (this->m_sharedMemoryBytes > 100000)
    {
        int carveout = int(60 + ((this->m_sharedMemoryBytes - 100000) / 64000.0) * 40.0);
        carveout = carveout > 100 ? 100 : carveout;
        cudaFuncSetAttribute(CAT_KERNEL_CG3,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             carveout);
    }

    cudaStreamCreate(&m_stream);

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceGetAttribute(&m_l2CacheSize,     cudaDevAttrL2CacheSize, device);
#if CUDART_VERSION >= 11000
    cudaDeviceGetAttribute(&m_maxPersistingL2,
                           cudaDevAttrMaxPersistingL2CacheSize, device);
#else
    m_maxPersistingL2 = 0;
#endif
    _configureL2ForData((n + 2 * halo) * (n + 2 * halo) * sizeof(half) * 2); // Assuming 2 tiles for ping-pong buffering
}

CATMultiStepSolver2::~CATMultiStepSolver2()
{
    resetL2Persistence();
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

void CATMultiStepSolver2::_configureL2ForData(size_t bytes)
{
#if CUDART_VERSION >= 11000
    if (bytes > m_l2CacheSize)
    {
        throw std::runtime_error("Requested L2 persistence size exceeds total L2 cache size.");
    }
    cudaMalloc(&d_scratchBuffer, bytes);


    size_t l2Budget = (size_t)bytes < (size_t)m_maxPersistingL2 ? bytes : (size_t)m_maxPersistingL2;
    if (m_maxPersistingL2 > 0)
    {
        // Reserve the full allowed persisting region
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2Budget);
        m_l2PersistenceOn = true;

        // printf("[CATMultiStepSolver2] L2 persistence enabled\n"
        //        "  Total L2        : %d bytes\n"
        //        "  Max persisting  : %d bytes\n",
        //        m_l2CacheSize, m_maxPersistingL2);
    }
    // else
    // {
    //     printf("[CATMultiStepSolver2] L2 persistence NOT available "
    //            "(pre-Ampere or CUDA < 11.0)\n");
    // }

    float hitRatio = 1.0;
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = d_scratchBuffer;
    attr.accessPolicyWindow.num_bytes = l2Budget;
    attr.accessPolicyWindow.hitRatio  = hitRatio;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    cudaError_t err = cudaStreamSetAttribute(
        m_stream, cudaStreamAttributeAccessPolicyWindow, &attr);

    if (err != cudaSuccess)
    {
        printf("[CATMultiStepSolver2] Warning: failed to set L2 access policy: %s\n",
               cudaGetErrorString(err));
    }

#endif
}

void CATMultiStepSolver2::resetL2Persistence()
{
#if CUDART_VERSION >= 11000
    if (!m_stream)
        return;

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;

    cudaStreamSetAttribute(m_stream,
                           cudaStreamAttributeAccessPolicyWindow, &attr);
#endif
    if (d_scratchBuffer)
    {
        cudaFree(d_scratchBuffer);
        d_scratchBuffer = nullptr;
    }
}

void CATMultiStepSolver2::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;

    this->castingKernelsBlockSize[0] = 16;
    this->castingKernelsBlockSize[1] = 16;
}

void CATMultiStepSolver2::prepareGrid(int n, int halo)
{
    // ---- Cooperative-launch occupancy query ------------------------------
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    int blocksPerSM = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        CAT_KERNEL_CG3,
        this->m_mainKernelsBlockSize[0] * this->m_mainKernelsBlockSize[1],
        m_sharedMemoryBytes);

    if (err != cudaSuccess)
    {
        printf("Occupancy query failed: %s\n", cudaGetErrorString(err));
        return;
    }

    int maxCoopBlocks = blocksPerSM * prop.multiProcessorCount;
    this->m_mainKernelsGridSize[0] = maxCoopBlocks;
    this->m_mainKernelsGridSize[1] = 1;

    printf("[CATMultiStepSolver2 cooperative launch check]\n"
           "  max blocks   : %d\n"
           "  block        : (%d, %d, 1)\n"
           "  shared mem   : %zu bytes\n"
           "  SM count     : %d\n"
           "  blocks / SM  : %d\n",
           maxCoopBlocks,
           m_mainKernelsBlockSize[0], m_mainKernelsBlockSize[1],
           m_sharedMemoryBytes,
           prop.multiProcessorCount,
           blocksPerSM);

    size_t nWithHalo = n + 2 * halo;
    this->castingKernelsGridSize[0] =
        (nWithHalo + this->castingKernelsBlockSize[0] - 1) / this->castingKernelsBlockSize[0];
    this->castingKernelsGridSize[1] =
        (nWithHalo + this->castingKernelsBlockSize[1] - 1) / this->castingKernelsBlockSize[1];
}

void CATMultiStepSolver2::changeLayout(uint8_t *inData[], void *outData[],
                                        int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0],
                     this->castingKernelsGridSize[1], nTiles);
    dim3 block = dim3(this->castingKernelsBlockSize[0],
                      this->castingKernelsBlockSize[1], 1);
    convertFp32ToFp16AndDoChangeLayout<<<grid, block>>>(
        (half **)outData, inData, n, halo);
    cudaDeviceSynchronize();
}

void CATMultiStepSolver2::unchangeLayout(void *inData[], uint8_t *outData[],
                                          int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0],
                     this->castingKernelsGridSize[1], nTiles);
    dim3 block = dim3(this->castingKernelsBlockSize[0],
                      this->castingKernelsBlockSize[1], 1);
    convertFp16ToFp32AndUndoChangeLayout<<<grid, block>>>(
        outData, (half **)inData, n, halo);
    cudaDeviceSynchronize();
}

void CATMultiStepSolver2::prepareData(uint8_t *inData[], void *outData[],
                                       int n, int halo, int radius, int nTiles)
{
    this->changeLayout(inData, outData, n, halo, nTiles);
}

void CATMultiStepSolver2::unprepareData(void *inData[], uint8_t *outData[],
                                          int n, int halo, int radius, int nTiles)
{
    this->unchangeLayout(inData, outData, n, halo, nTiles);
}

void CATMultiStepSolver2::StepSimulation(void *inData[], void *outData[],
                                          int n, int halo, int radius, int nTiles)
{
    StepSimulationMulti(inData, outData, n, halo, radius, nTiles, 1);
}

void CATMultiStepSolver2::StepSimulationMulti(
    void *inData[], void *outData[],
    int n, int halo, int radius, int nTiles, int innerSteps)
{
    if (innerSteps <= 0)
        return;

    dim3 grid(this->m_mainKernelsGridSize[0], 1, 1);
    dim3 block(this->m_mainKernelsBlockSize[0],
               this->m_mainKernelsBlockSize[1], 1);

    // ----- Kernel arguments -----------------------------------------------
    half     **inDataHalf  = (half **)inData;
    half     **outDataHalf = (half **)outData;
    uint32_t   total_regions_x = (n + (m_nRegionsH * 16) - 1) / (m_nRegionsH * 16);
    uint32_t   total_regions_y = (n + (m_nRegionsV * 16) - 1) / (m_nRegionsV * 16);
    size_t     n_size = (size_t)n;  // kernel expects size_t; passing &n (int) would cause 8-byte read from a 4-byte var

    void *kernelArgs[] = {
        (void *)&inDataHalf,
        (void *)&outDataHalf,
        (void *)&d_scratchBuffer,
        (void *)&n_size,
        (void *)&halo,
        (void *)&radius,
        (void *)&m_nRegionsH,
        (void *)&m_nRegionsV,
        (void *)&total_regions_x,
        (void *)&total_regions_y,
        (void *)&SMIN,
        (void *)&SMAX,
        (void *)&BMIN,
        (void *)&BMAX,
        (void *)&nTiles,
        (void *)&innerSteps,
    };

    // printf("[CATMultiStepSolver2] Launching CG3 kernel with %d inner steps\n", innerSteps);
    // printf("  grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    // printf("  block: (%d, %d, %d)\n", block.x, block.y, block.z);

    // ----- Cooperative kernel launch (on the L2-persistence stream) ------
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)CAT_KERNEL_CG3,
        grid, block,
        kernelArgs,
        m_sharedMemoryBytes,
        m_stream        
    );

    if (err != cudaSuccess)
    {
        printf("cudaLaunchCooperativeKernel (CG3) failed: %s\n",
               cudaGetErrorString(err));
        return;
    }

    err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess)
    {
        printf("CG3 kernel execution failed: %s\n",
               cudaGetErrorString(err));
        return;
    }

}

void CATMultiStepSolver2::fillPeriodicBoundaryConditions(void *data[], int n,
                                                          int halo, int nTiles)
{
    size_t nWithHalo = n + 2 * halo;
    dim3 horizontalGrid =
        dim3(2 * (int)ceil(n / (float)this->castingKernelsBlockSize[0]));
    dim3 verticalGrid =
        dim3(2 * (int)ceil(nWithHalo / (float)this->castingKernelsBlockSize[0]));
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);

    copyHorizontalHaloCoalescedVersion<<<horizontalGrid, block>>>(
        (half **)data, n, nWithHalo);
    cudaDeviceSynchronize();
    copyVerticalHaloCoalescedVersion<<<verticalGrid, block>>>(
        (half **)data, n, nWithHalo);
    cudaDeviceSynchronize();
}
