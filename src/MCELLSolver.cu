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
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;

    this->boundaryKernelsBlockSize[0] = 256;
    this->boundaryKernelsBlockSize[1] = 1;
}

void MCELLSolver::prepareGrid(int n, int halo)
{
    int block_x = (this->m_mainKernelsBlockSize[0] * this->cellsPerThread);
    int block_y = this->m_mainKernelsBlockSize[1];

    this->m_mainKernelsGridSize[0] = (n + block_x - 1) / block_x;
    this->m_mainKernelsGridSize[1] = (n + block_y - 1) / block_y;
    // this->horizontalBoundaryGrid = dim3((int)ceil(n / (float)this->boundaryKernelsBlockSize.x));
    // this->verticalBoundaryGrid = dim3((int)ceil((n + 2 * radius) / (float)this->boundaryKernelsBlockSize.x));
}

void MCELLSolver::StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
{
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);

    int sharedMemorySize =
        (this->m_mainKernelsBlockSize[0] * 2 + 2 * halo) * (this->m_mainKernelsBlockSize[1] + 2 * halo) * sizeof(char);
    MCELL_KERNEL<<<grid, block, sharedMemorySize>>>(inData, outData, n, n, cellsPerThread, radius, 2 * halo);
    (cudaDeviceSynchronize());
}
