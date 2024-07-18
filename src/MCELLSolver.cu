#include "include/MCELLSolver.cuh"

void MCELLSolver::fillHorizontalBoundaryConditions(char *inData, int n, int radius)
{
    copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
}
void MCELLSolver::fillVerticalBoundaryConditions(char *inData, int n, int radius)
{
    copy_Cols<<<this->verticalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
}

void MCELLSolver::setBlockSize(int block_x = 32, int block_y = 32)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
    this->boundaryKernelsBlockSize = dim3(256, 1);
}
void MCELLSolver::setGridSize(int n, int radius, int grid_z = 1)
{
    int block_x = (this->mainKernelsBlockSize.x * this->cellsPerThread);
    int block_y = this->mainKernelsBlockSize.y;
    this->mainKernelsGridSize = dim3((n + block_x - 1) / block_x, (n + block_y - 1) / block_y, grid_z);

    this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryKernelsBlockSize.x));
    this->verticalBoundaryGrid = dim3((int)ceil((n + 2 * radius) / (float)this->boundaryKernelsBlockSize.x));
}

void MCELLSolver::CAStepAlgorithm(char *inData, char *outData, int n, int radius)
{
    int sharedMemorySize =
        (this->mainKernelsBlockSize.x * 2 + 2 * radius) * (this->mainKernelsBlockSize.y + 2 * radius) * sizeof(char);
    MCELL_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize, sharedMemorySize>>>(inData, outData, n,
                                                                                              n + 2 * radius);
}
