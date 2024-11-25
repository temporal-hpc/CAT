#include "CATSolver.cuh"
#include "GPUKernels.cuh"
using namespace Temporal;

CATSolver::CATSolver(int nRegionsH, int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX ) : Solver(SMIN, SMAX, BMIN, BMAX)
{
    this->m_nRegionsH = nRegionsH;
    this->m_nRegionsV = nRegionsV;
    this->m_sharedMemoryBytes = ((nRegionsH + 2) * (nRegionsV + 2) * 16 * 16 + 256 * 2) * sizeof(half);
    cudaFuncSetAttribute(CAT_KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, this->m_sharedMemoryBytes);
    if (this->m_sharedMemoryBytes > 100000)
    {
        int carveout = int(60 + ((this->m_sharedMemoryBytes - 100000) / 64000.0) * 40.0);
        carveout = carveout > 100 ? 100 : carveout;
        cudaFuncSetAttribute(CAT_KERNEL, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    }
}

void CATSolver::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;

    this->castingKernelsBlockSize[0] = 16;
    this->castingKernelsBlockSize[1] = 16;
}

void CATSolver::prepareGrid(int n, int halo)
{
    size_t nWithHalo = n + 2 * halo;
    this->m_mainKernelsGridSize[0] = (n + (m_nRegionsH * 16) - 1) / (m_nRegionsH * 16);
    this->m_mainKernelsGridSize[1] = (n + (m_nRegionsV * 16) - 1) / (m_nRegionsV * 16);

    this->castingKernelsGridSize[0] =
        (nWithHalo + this->castingKernelsBlockSize[0] - 1) / this->castingKernelsBlockSize[0];
    this->castingKernelsGridSize[1] =
        (nWithHalo + this->castingKernelsBlockSize[1] - 1) / this->castingKernelsBlockSize[1];
}

void CATSolver::changeLayout(uint8_t *inData[], void *outData[], int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);
    convertFp32ToFp16AndDoChangeLayout<<<grid, block>>>((half **)outData, inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATSolver::unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo, int nTiles)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], 1);
    convertFp16ToFp32AndUndoChangeLayout<<<grid, block>>>(outData, (half **)inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATSolver::prepareData(uint8_t *inData[], void *outData[], int n, int halo, int radius, int nTiles)
{
    this->changeLayout(inData, outData, n, halo, nTiles);
}

void CATSolver::unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    this->unchangeLayout(inData, outData, n, halo, nTiles);
}

void CATSolver::StepSimulation(void *inData[], void *outData[], int n, int halo, int radius, int nTiles)
{
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);
    CAT_KERNEL<<<grid, block, m_sharedMemoryBytes>>>((half **)inData, (half **)outData, n, halo, radius, m_nRegionsH,
                                                     m_nRegionsV, SMIN, SMAX, BMIN, BMAX);
    (cudaDeviceSynchronize());
}
