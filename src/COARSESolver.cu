#include "COARSESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void COARSESolver::setBlockSize(int block_x, int block_y)
{
    // this->m_mainKernelsBlockSize = dim3(block_x, block_y);

    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;
    this->m_mainKernelsBlockSize[2] = 1;
}

void COARSESolver::setGridSize(int n, int halo, int grid_z)
{
    this->m_mainKernelsGridSize[0] = (n + 80 - 1) / 80;
    this->m_mainKernelsGridSize[1] = (n + 80 - 1) / 80;
    this->m_mainKernelsGridSize[2] = grid_z;
}

void COARSESolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius)
{
    size_t sharedMemorySize = (80 + 2 * halo) * (80 + 2 * halo) * sizeof(uint8_t);
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], this->m_mainKernelsGridSize[2]);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], this->m_mainKernelsBlockSize[2]);
    COARSE_KERNEL<<<grid, block, sharedMemorySize>>>(inData, outData, n, halo, radius);
    (cudaDeviceSynchronize());
}