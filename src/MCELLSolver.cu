#include "GPUKernels.cuh"
#include "MCELLSolver.cuh"

using namespace Temporal;

MCELLSolver::MCELLSolver(int radius)
{
    this->radius = radius;
}

// void MCELLSolver::fillHorizontalBoundaryConditions(char *inData, int n, int radius)
// {
//     copy_Rows<<<this->horizontalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
// }
// void MCELLSolver::fillVerticalBoundaryConditions(char *inData, int n, int radius)
// {
//     copy_Cols<<<this->verticalBoundaryGrid, this->boundaryKernelsBlockSize>>>(n, inData, radius, 2 * radius);
// }

void MCELLSolver::setBlockSize(int block_x, int block_y)
{
    this->mainKernelsBlockSize[0] = block_x;
    this->mainKernelsBlockSize[1] = block_y;
    this->mainKernelsBlockSize[2] = 1;

    this->boundaryKernelsBlockSize[0] = 256;
    this->boundaryKernelsBlockSize[1] = 1;
    this->boundaryKernelsBlockSize[2] = 1;
}
void MCELLSolver::setGridSize(int n, int grid_z)
{
    int block_x = (this->mainKernelsBlockSize[0] * this->cellsPerThread);
    int block_y = this->mainKernelsBlockSize[1];

    this->mainKernelsGridSize[0] = (n + block_x - 1) / block_x;
    this->mainKernelsGridSize[1] = (n + block_y - 1) / block_y;
    this->mainKernelsGridSize[2] = grid_z;
    // this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryKernelsBlockSize.x));
    // this->verticalBoundaryGrid = dim3((int)ceil((n + 2 * radius) / (float)this->boundaryKernelsBlockSize.x));
}

void MCELLSolver::StepSimulation(uint8_t *inData, uint8_t *outData, int n, int radius)
{
    dim3 grid = dim3(this->mainKernelsGridSize[0], this->mainKernelsGridSize[1], this->mainKernelsGridSize[2]);
    dim3 block = dim3(this->mainKernelsBlockSize[0], this->mainKernelsBlockSize[1], this->mainKernelsBlockSize[2]);

    int sharedMemorySize =
        (this->mainKernelsBlockSize[0] * 2 + 2 * radius) * (this->mainKernelsBlockSize[1] + 2 * radius) * sizeof(char);
    MCELL_KERNEL<<<grid, block, sharedMemorySize>>>(inData, outData, n, n, cellsPerThread, radius, 2 * radius);
}
