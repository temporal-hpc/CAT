#include "BASESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void BASESolver::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;
}

void BASESolver::prepareGrid(int n, int halo)
{
    this->m_mainKernelsGridSize[0] = (n + this->m_mainKernelsBlockSize[0] - 1) / this->m_mainKernelsBlockSize[0];
    this->m_mainKernelsGridSize[1] = (n + this->m_mainKernelsBlockSize[1] - 1) / this->m_mainKernelsBlockSize[1];
}

void BASESolver::prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    // Do nothing
}

void BASESolver::unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    // Do nothing
}

void BASESolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    dim3 grid = dim3(m_mainKernelsGridSize[0], m_mainKernelsGridSize[1], nTiles);
    dim3 block = dim3(m_mainKernelsBlockSize[0], m_mainKernelsBlockSize[1], 1);

    BASE_KERNEL<<<grid, block>>>(inData, outData, n, radius, radius);
    (cudaDeviceSynchronize());
}