#include "CATMultiStepSolver.cuh"
#include "GPUKernels.cuh"
#include <cuda_runtime.h>
#include <assert.h>
using namespace Temporal;

CATMultiStepSolver::CATMultiStepSolver(int nRegionsH, int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX)
    : Solver(SMIN, SMAX, BMIN, BMAX)
{
    this->m_nRegionsH = nRegionsH;
    this->m_nRegionsV = nRegionsV;
    this->m_sharedMemoryBytes = ((nRegionsH + 2) * (nRegionsV + 2) * 16 * 16 + 256 * 2) * sizeof(half);

    cudaFuncSetAttribute(CAT_KERNEL_CG, cudaFuncAttributeMaxDynamicSharedMemorySize, this->m_sharedMemoryBytes);
    if (this->m_sharedMemoryBytes > 100000)
    {
        int carveout = int(60 + ((this->m_sharedMemoryBytes - 100000) / 64000.0) * 40.0);
        carveout = carveout > 100 ? 100 : carveout;
        cudaFuncSetAttribute(CAT_KERNEL_CG, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    }
}

void CATMultiStepSolver::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;

    this->castingKernelsBlockSize[0] = 16;
    this->castingKernelsBlockSize[1] = 16;
}

void CATMultiStepSolver::prepareGrid(int n, int halo)
{
    size_t nWithHalo = n + 2 * halo;
    this->m_mainKernelsGridSize[0] = (n + (m_nRegionsH * 16) - 1) / (m_nRegionsH * 16);
    this->m_mainKernelsGridSize[1] = (n + (m_nRegionsV * 16) - 1) / (m_nRegionsV * 16);

    this->castingKernelsGridSize[0] =
        (nWithHalo + this->castingKernelsBlockSize[0] - 1) / this->castingKernelsBlockSize[0];
    this->castingKernelsGridSize[1] =
        (nWithHalo + this->castingKernelsBlockSize[1] - 1) / this->castingKernelsBlockSize[1];
}

void CATMultiStepSolver::changeLayout(uint8_t *inData[], void *outData[], int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], nTiles);
    dim3 block = dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);
    convertFp32ToFp16AndDoChangeLayout<<<grid, block>>>((half **)outData, inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATMultiStepSolver::unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], nTiles);
    dim3 block = dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);
    convertFp16ToFp32AndUndoChangeLayout<<<grid, block>>>(outData, (half **)inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATMultiStepSolver::prepareData(uint8_t *inData[], void *outData[], int n, int halo, int radius, int nTiles)
{
    this->changeLayout(inData, outData, n, halo, nTiles);
}

void CATMultiStepSolver::unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    this->unchangeLayout(inData, outData, n, halo, nTiles);
}

void CATMultiStepSolver::StepSimulation(void *inData[], void *outData[], int n, int halo, int radius, int nTiles)
{
    StepSimulationMulti(inData, outData, n, halo, radius, nTiles, 1);
}

void CATMultiStepSolver::StepSimulationMulti(
    void *inData[],
    void *outData[],
    int n,
    int halo,
    int radius,
    int nTiles,
    int innerSteps
) {
    if (innerSteps <= 0) {
        return;
    }

    // Grid / block configuration
    dim3 grid(
        this->m_mainKernelsGridSize[0],
        this->m_mainKernelsGridSize[1],
        nTiles
    );

    dim3 block(
        this->m_mainKernelsBlockSize[0],
        this->m_mainKernelsBlockSize[1],
        1
    );

    // Total number of blocks in the cooperative grid
    int totalBlocks = grid.x * grid.y * grid.z;

    // ---------------------------------------------------------------------
    // Compute cooperative launch limit
    // ---------------------------------------------------------------------
    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    int blocksPerSM = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        CAT_KERNEL_CG,
        block.x * block.y * block.z,
        m_sharedMemoryBytes
    );

    if (err != cudaSuccess) {
        printf("Occupancy query failed: %s\n", cudaGetErrorString(err));
        return;
    }

    int maxCoopBlocks = blocksPerSM * prop.multiProcessorCount;

    printf(
        "[Cooperative launch check]\n"
        "  total blocks : %d\n"
        "  max blocks   : %d\n"
        "  block        : (%d, %d, %d)\n"
        "  shared mem   : %zu bytes\n"
        "  SM count     : %d\n"
        "  blocks / SM  : %d\n",
        totalBlocks,
        maxCoopBlocks,
        block.x, block.y, block.z,
        m_sharedMemoryBytes,
        prop.multiProcessorCount,
        blocksPerSM
    );

    assert(
        totalBlocks <= maxCoopBlocks &&
        "ERROR: cooperative kernel grid exceeds device limit"
    );

    // ---------------------------------------------------------------------
    // Kernel arguments
    // ---------------------------------------------------------------------
    half **inDataHalf  = (half **)inData;
    half **outDataHalf = (half **)outData;

    void *kernelArgs[] = {
        (void *)&inDataHalf,
        (void *)&outDataHalf,
        (void *)&n,
        (void *)&halo,
        (void *)&radius,
        (void *)&m_nRegionsH,
        (void *)&m_nRegionsV,
        (void *)&SMIN,
        (void *)&SMAX,
        (void *)&BMIN,
        (void *)&BMAX,
        (void *)&innerSteps,
    };

    // ---------------------------------------------------------------------
    // Cooperative kernel launch
    // ---------------------------------------------------------------------
    err = cudaLaunchCooperativeKernel(
        (void *)CAT_KERNEL_CG,
        grid,
        block,
        kernelArgs,
        m_sharedMemoryBytes,
        0
    );

    if (err != cudaSuccess) {
        printf(
            "cudaLaunchCooperativeKernel failed: %s\n",
            cudaGetErrorString(err)
        );
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf(
            "Kernel execution failed: %s\n",
            cudaGetErrorString(err)
        );
        return;
    }

    // ---------------------------------------------------------------------
    // Pointer swap logic (unchanged)
    // ---------------------------------------------------------------------
    if ((innerSteps & 1) == 0) {
        void **temp = inData;
        inData = outData;
        outData = temp;
    }
}


void CATMultiStepSolver::fillPeriodicBoundaryConditions(void *data[], int n, int halo, int nTiles){
    size_t nWithHalo = n + 2 * halo;
    dim3 horizontalGrid = dim3(2 * (int)ceil(n / (float)this->castingKernelsBlockSize[0]));
    dim3 verticalGrid = dim3(2 * (int)ceil(nWithHalo / (float)this->castingKernelsBlockSize[0]));
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);

    copyHorizontalHaloCoalescedVersion<<<horizontalGrid, block>>>((half **)data, n, n + 2 * halo);
    cudaDeviceSynchronize();
    copyVerticalHaloCoalescedVersion<<<verticalGrid, block>>>((half **)data, n, n + 2 * halo);
    cudaDeviceSynchronize();
}
