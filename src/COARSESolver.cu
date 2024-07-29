#include "COARSESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void COARSESolver::setBlockSize(int block_x, int block_y)
{
    // this->mainKernelsBlockSize = dim3(block_x, block_y);

    this->mainKernelsBlockSize[0] = block_x;
    this->mainKernelsBlockSize[1] = block_y;
    this->mainKernelsBlockSize[2] = 1;
}

void COARSESolver::setGridSize(int n, int grid_z)
{
    this->mainKernelsGridSize[0] = (n + this->mainKernelsBlockSize[0] - 1) / this->mainKernelsBlockSize[0];
    this->mainKernelsGridSize[1] = (n + this->mainKernelsBlockSize[1] - 1) / this->mainKernelsBlockSize[1];
    this->mainKernelsGridSize[2] = grid_z;
}

void COARSESolver::StepSimulation(uint8_t *inData, uint8_t *outData, int n, int radius)
{
    dim3 grid = dim3(this->mainKernelsGridSize[0], this->mainKernelsGridSize[1], this->mainKernelsGridSize[2]);
    dim3 block = dim3(this->mainKernelsBlockSize[0], this->mainKernelsBlockSize[1], this->mainKernelsBlockSize[2]);
    COARSE_KERNEL<<<grid, block>>>(inData, outData, n, n + 2 * radius, radius);
    // (cudaDeviceSynchronize());
}