#include "include/BASESolver.cuh"

void BASESolver::setBlockSize(int block_x = 32, int block_y = 32)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
}

void BASESolver::setGridSize(int n, int grid_z = 1)
{
    this->mainKernelsGridSize = dim3((n + this->mainKernelsBlockSize.x - 1) / this->mainKernelsBlockSize.x,
                                     (n + this->mainKernelsBlockSize.y - 1) / this->mainKernelsBlockSize.y, grid_z);
}

void BASESolver::CAStepAlgorithm(char *inData, char *outData, int n, int radius)
{
    BASE_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize>>>(inData, outData, n, n + 2 * radius);
    // (cudaDeviceSynchronize());
}