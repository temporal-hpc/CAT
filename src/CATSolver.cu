#include "CATSolver.cuh"
#include "GPUKernels.cuh"
using namespace Temporal;

CATSolver::CATSolver(int nRegionsH, int nRegionsV)
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
    this->m_mainKernelsBlockSize[2] = 1;

    this->castingKernelsBlockSize[0] = 16;
    this->castingKernelsBlockSize[1] = 16;
    this->castingKernelsBlockSize[2] = 1;
}

void CATSolver::setGridSize(int n, int halo, int grid_z)
{
    size_t nWithHalo = n + 2 * halo;
    this->m_mainKernelsGridSize[0] = (n + (m_nRegionsH * 16) - 1) / (m_nRegionsH * 16);
    this->m_mainKernelsGridSize[1] = (n + (m_nRegionsV * 16) - 1) / (m_nRegionsV * 16);
    this->m_mainKernelsGridSize[2] = grid_z;

    this->castingKernelsGridSize[0] =
        (nWithHalo + this->castingKernelsBlockSize[0] - 1) / this->castingKernelsBlockSize[0];
    this->castingKernelsGridSize[1] =
        (nWithHalo + this->castingKernelsBlockSize[1] - 1) / this->castingKernelsBlockSize[1];
    this->castingKernelsGridSize[2] = grid_z;
}

void CATSolver::changeLayout(uint8_t *inData[], void *outData[], int n, int halo)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], this->castingKernelsGridSize[2]);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], this->castingKernelsBlockSize[2]);
    convertFp32ToFp16AndDoChangeLayout<<<grid, block>>>((half **)outData, inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATSolver::unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], this->castingKernelsGridSize[2]);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], this->castingKernelsBlockSize[2]);
    convertFp16ToFp32AndUndoChangeLayout<<<grid, block>>>(outData, (half **)inData, n, halo);
    (cudaDeviceSynchronize());
}

void CATSolver::prepareData(uint8_t *inData[], void *outData[], int n, int halo, int radius)
{
    this->changeLayout(inData, outData, n, halo);
}

void CATSolver::unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius)
{
    this->unchangeLayout(inData, outData, n, halo);
}

void CATSolver::StepSimulation(void *inData[], void *outData[], int n, int halo, int radius)
{
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], this->m_mainKernelsGridSize[2]);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], this->m_mainKernelsBlockSize[2]);
    CAT_KERNEL<<<grid, block, m_sharedMemoryBytes>>>((half **)inData, (half **)outData, n, halo, radius, m_nRegionsH,
                                                     m_nRegionsV);
    (cudaDeviceSynchronize());
}
