#include "GPUKernels.cuh"
#include "SHAREDSolver.cuh"

using namespace Temporal;

void SHAREDSolver::setBlockSize(int block_x, int block_y)
{
    // this->m_mainKernelsBlockSize = dim3(block_x, block_y);

    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;
}

void SHAREDSolver::prepareGrid(int n, int halo)
{
    this->m_mainKernelsGridSize[0] = (n + this->m_mainKernelsBlockSize[0] - 1) / this->m_mainKernelsBlockSize[0];
    this->m_mainKernelsGridSize[1] = (n + this->m_mainKernelsBlockSize[1] - 1) / this->m_mainKernelsBlockSize[1];
}

void SHAREDSolver::prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
}

void SHAREDSolver::unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
}

void SHAREDSolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    size_t sharedMemorySize = sizeof(uint8_t) * (this->m_mainKernelsBlockSize[0] + 2 * radius) *
                              (this->m_mainKernelsBlockSize[1] + 2 * radius);

    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);

    SHARED_KERNEL<<<grid, block, sharedMemorySize>>>(inData, outData, n, n, radius, 2 * radius);
    (cudaDeviceSynchronize());
}
