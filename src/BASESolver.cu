#include "BASESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void BASESolver::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;
    this->m_mainKernelsBlockSize[2] = 1;
}

void BASESolver::setGridSize(int n, int halo, int grid_z)
{
    this->m_mainKernelsGridSize[0] = (n + this->m_mainKernelsBlockSize[0] - 1) / this->m_mainKernelsBlockSize[0];
    this->m_mainKernelsGridSize[1] = (n + this->m_mainKernelsBlockSize[1] - 1) / this->m_mainKernelsBlockSize[1];
    this->m_mainKernelsGridSize[2] = grid_z;
}

void BASESolver::prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius)
{
    // Do nothing
}

void BASESolver::unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius)
{
    // Do nothing
}

void BASESolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius)
{
    dim3 grid = dim3(m_mainKernelsGridSize[0], m_mainKernelsGridSize[1], m_mainKernelsGridSize[2]);
    dim3 block = dim3(m_mainKernelsBlockSize[0], m_mainKernelsBlockSize[1], m_mainKernelsBlockSize[2]);

    BASE_KERNEL<<<grid, block>>>(inData, outData, n, radius, radius);
    (cudaDeviceSynchronize());
}