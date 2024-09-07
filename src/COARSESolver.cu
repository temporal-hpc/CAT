#include "COARSESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void COARSESolver::setBlockSize(int block_x, int block_y)
{
    // this->m_mainKernelsBlockSize = dim3(block_x, block_y);

    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;
}

void COARSESolver::prepareGrid(int n, int halo)
{
    this->m_mainKernelsGridSize[0] = (n + 80 - 1) / 80;
    this->m_mainKernelsGridSize[1] = (n + 80 - 1) / 80;
}

void COARSESolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    size_t sharedMemorySize = (80 + 2 * halo) * (80 + 2 * halo) * sizeof(uint8_t);
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);
    COARSE_KERNEL<<<grid, block, sharedMemorySize>>>(inData, outData, n, halo, radius);
    (cudaDeviceSynchronize());
}