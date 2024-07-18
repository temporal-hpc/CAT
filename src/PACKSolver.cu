#include "CellularAutomata/Solvers/PACKSolver.cuh"

void PACKSolver::setBlockSize(int block_x, int block_y)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
    this->boundaryKernelsBlockSize = dim3(256);
}

void PACKSolver::setGridSize(int n, int radius, int grid_z)
{
    int innerHorizontalSize = n;
    int innerVerticalSize = n;
    int fullVerticalSize = n + 2 * radius;
    this->mainKernelsGridSize = dim3((int)ceil(innerHorizontalSize / (float)this->mainKernelsBlockSize.x),
                                     (int)ceil(innerVerticalSize / (float)this->mainKernelsBlockSize.y));

    this->horizontalBoundaryGrid = dim3((int)ceil(innerHorizontalSize / (float)this->boundaryKernelsBlockSize.x));
    this->verticalBoundaryGrid = dim3((int)ceil(fullVerticalSize / (float)this->boundaryKernelsBlockSize.x));
}

void PACKSolver::setCALookUpTable(uint64_t *CALookUpTable)
{
    cudaMalloc(&CALookUpTable, sizeof(int) * 2 * (elementsPerCel + 1));
    dim3 block = dim3(1, 2, 1);
    dim3 grid = dim3((elementsPerCel + 1), 1, 1);
    kernel_init_lookup_table<<<grid, block>>>(CALookUpTable);
}

// TODO: FIX /8 and stuff the width is not the same
// TODO: make it from 64 to 64 bits?
void PACKSolver::unpackState(uint64_t *inData, char *outData, int n, int radius)
{
    int innerHorizontalSize = n;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;

    unpackStateKernel<<<this->mainKernelsGridSize, this->mainKernelsBlockSize>>>(
        inData, outData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
}

void PACKSolver::packState(char *inData, u_int64_t *outData, int n, int radius);
{
    int innerHorizontalSize = n;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;
    packStateKernel<<<this->mainKernelsGridSize, this->mainKernelsBlockSize>>>(
        outData, inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
}

void PACKSolver::fillHorizontalBoundaryConditions(uint64_t *inData, int n, int radius)
{
    int innerHorizontalSize = n;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;

    ghostRows<<<this->horizontalBoundaryGrid, this->boundaryKernelsBlockSize>>>(
        inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
}

void PACKSolver::fillVerticalBoundaryConditions(uint64_t *inData, int n, int radius)
{
    int innerHorizontalSize = n;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;

    ghostCols<<<this->verticalBoundaryGrid, this->boundaryKernelsBlockSize>>>(
        inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
}

void PACKSolver::CAStepAlgorithm(uint64_t *inData, uint64_t *outData, int n, int radius)
{
    int sharedMemoryBytes = (mainKernelsBlockSize.x + 2 * n) * (mainKernelsBlockSize.y + 2 * n) * sizeof(uint64_t);
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;
    PACK_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize, sharedMemoryBytes>>>(
        inData, outData, this->CALookUpTable, n, n, horizontalHaloSize, verticalHaloSize);
}
