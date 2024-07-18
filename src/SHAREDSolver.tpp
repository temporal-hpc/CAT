#include "include/SHAREDSolver.cuh"

void SHAREDSolver::setBlockSize(int block_x = 16, int block_y = 16)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
}

void SHAREDSolver::setGridSize(int n, int grid_z = 1)
{
    this->mainKernelsGridSize = dim3((n + this->mainKernelsBlockSize.x - 1) / this->mainKernelsBlockSize.x,
                                     (n + this->mainKernelsBlockSize.y - 1) / this->mainKernelsBlockSize.y, grid_z);
}

void SHAREDSolver::CAStepAlgorithm(char *inData, char *outData, int n, int radius)
{
    size_t sharedMemorySize =
        sizeof(char) * (this->mainKernelsBlockSize.x + 2 * radius) * (this->mainKernelsBlockSize.y + 2 * radius);

    SHARED_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize, sharedMemorySize>>>(inData, outData, n, n,
                                                                                               radius, 2 * radius);
}

void SHAREDSolver::fillHorizontalBoundaryConditions(char *inData, int n, int radius)
{
    copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
}
void SHAREDSolver::fillVerticalBoundaryConditions(char *inData, int n, int radius)
{
    copy_Cols<<<this->verticalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
}
