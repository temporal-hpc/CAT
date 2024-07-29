#include "BASESolver.cuh"
#include "GPUKernels.cuh"

using namespace Temporal;

void BASESolver::setBlockSize(int block_x, int block_y)
{
    this->mainKernelsBlockSize[0] = block_x;
    this->mainKernelsBlockSize[1] = block_y;
    this->mainKernelsBlockSize[2] = 1;
}

void BASESolver::setGridSize(int n, int grid_z)
{
    // this->mainKernelsGridSize = dim3((n + this->mainKernelsBlockSize.x - 1) / this->mainKernelsBlockSize.x,
    //                                  (n + this->mainKernelsBlockSize.y - 1) / this->mainKernelsBlockSize.y, grid_z);
    this->mainKernelsGridSize[0] = (n + this->mainKernelsBlockSize[0] - 1) / this->mainKernelsBlockSize[0];
    this->mainKernelsGridSize[1] = (n + this->mainKernelsBlockSize[1] - 1) / this->mainKernelsBlockSize[1];
    this->mainKernelsGridSize[2] = grid_z;
}

void BASESolver::prepareData(uint8_t *inData, uint8_t *outData, int n, int radius)
{
    // Do nothing
}

void BASESolver::StepSimulation(uint8_t *inData, uint8_t *outData, int n, int radius)
{
    dim3 grid = dim3(mainKernelsGridSize[0], mainKernelsGridSize[1], mainKernelsGridSize[2]);
    dim3 block = dim3(mainKernelsBlockSize[0], mainKernelsBlockSize[1], mainKernelsBlockSize[2]);

    BASE_KERNEL<<<grid, block>>>(inData, outData, n, n + 2 * radius, radius);
    // (cudaDeviceSynchronize());
}