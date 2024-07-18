#include "include/COARSESolver.cuh"

void COARSESolver::setBlockSize(int block_x = 16, int block_y = 16)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
}

void COARSESolver::setGridSize(int n, int grid_z = 1)
{
    this->mainKernelsGridSize = dim3((n + this->mainKernelsBlockSize.x - 1) / this->mainKernelsBlockSize.x,
                                     (n + this->mainKernelsBlockSize.y - 1) / this->mainKernelsBlockSize.y, grid_z);
}

void COARSESolver::CAStepAlgorithm(char *inData, char *outData, int n, int radius)
{
    COARSE_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize>>>(inData, outData, n, n + 2 * radius);
    // (cudaDeviceSynchronize());
}