#include "GPUKernels.cuh"
#include "PACKSolver.cuh"

using namespace Temporal;

PACKSolver::PACKSolver(int n, int radius)
{
    this->n = n;
    this->packedN = n / elementsPerCel;
    this->packedRadius = ceil(radius / (float)elementsPerCel);
    setupLookupTable(radius);
}

void PACKSolver::setBlockSize(int block_x, int block_y)
{
    this->mainKernelsBlockSize[0] = block_x;
    this->mainKernelsBlockSize[1] = block_y;
    this->mainKernelsBlockSize[2] = 1;

    this->boundaryKernelsBlockSize[0] = 256;
    this->boundaryKernelsBlockSize[1] = 1;
    this->boundaryKernelsBlockSize[2] = 1;
}

void PACKSolver::setGridSize(int n, int grid_z)
{
    int innerHorizontalSize = packedN;
    int innerVerticalSize = n;
    // int fullVerticalSize = n + 2 * radius;

    this->mainKernelsGridSize[0] = (int)ceil(innerHorizontalSize / (float)this->mainKernelsBlockSize[0]);
    this->mainKernelsGridSize[1] = (int)ceil(innerVerticalSize / (float)this->mainKernelsBlockSize[1]);
    this->mainKernelsGridSize[2] = grid_z;
}

void PACKSolver::setupLookupTable(int radius)
{
    int cagigas_cell_neigh = ((radius * 2 + 1) * (radius * 2 + 1) - 1);

    cudaMalloc(&CALookUpTable, sizeof(int) * 2 * (cagigas_cell_neigh + 1));
    dim3 block = dim3(1, 2, 1);
    dim3 grid = dim3((cagigas_cell_neigh + 1), 1, 1);
    kernel_init_lookup_table<<<grid, block>>>(CALookUpTable, radius);
}

void PACKSolver::unpackState(uint64_t *inData, uint8_t *outData, int n, int radius)
{
    int innerHorizontalSize = packedN;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = packedRadius;

    dim3 grid = dim3(this->mainKernelsGridSize[0], this->mainKernelsGridSize[1], this->mainKernelsGridSize[2]);
    dim3 block = dim3(this->mainKernelsBlockSize[0], this->mainKernelsBlockSize[1], this->mainKernelsBlockSize[2]);
    unpackStateKernel<<<grid, block>>>(inData, outData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                       verticalHaloSize);
}

void PACKSolver::packState(uint8_t *inData, uint64_t *outData, int n, int radius)
{
    int innerHorizontalSize = packedN;
    int innerVerticalSize = n;
    int horizontalHaloSize = radius;
    int verticalHaloSize = packedRadius;
    dim3 grid = dim3(this->mainKernelsGridSize[0], this->mainKernelsGridSize[1], this->mainKernelsGridSize[2]);
    dim3 block = dim3(this->mainKernelsBlockSize[0], this->mainKernelsBlockSize[1], this->mainKernelsBlockSize[2]);

    packStateKernel<<<grid, block>>>(inData, outData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                     verticalHaloSize);
}

// void PACKSolver::fillHorizontalBoundaryConditions(uint64_t *inData, int n, int radius)
// {
//     int innerHorizontalSize = n;
//     int innerVerticalSize = n;
//     int horizontalHaloSize = radius;
//     int verticalHaloSize = radius;

//     ghostRows<<<this->horizontalBoundaryGrid, this->boundaryKernelsBlockSize>>>(
//         inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
// }

// void PACKSolver::fillVerticalBoundaryConditions(uint64_t *inData, int n, int radius)
// {
//     int innerHorizontalSize = n;
//     int innerVerticalSize = n;
//     int horizontalHaloSize = radius;
//     int verticalHaloSize = radius;

//     ghostCols<<<this->verticalBoundaryGrid, this->boundaryKernelsBlockSize>>>(
//         inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize, verticalHaloSize);
// }

void PACKSolver::StepSimulation(uint64_t *inData, uint64_t *outData, int n, int radius)
{
    int sharedMemoryBytes = (mainKernelsBlockSize[0] + 2 * n) * (mainKernelsBlockSize[1] + 2 * n) * sizeof(uint64_t);
    int horizontalHaloSize = radius;
    int verticalHaloSize = radius;

    dim3 grid = dim3(mainKernelsGridSize[0], mainKernelsGridSize[1], mainKernelsGridSize[2]);
    dim3 block = dim3(mainKernelsBlockSize[0], mainKernelsBlockSize[1], mainKernelsBlockSize[2]);
    PACK_KERNEL<<<grid, block, sharedMemoryBytes>>>(inData, outData, this->CALookUpTable, n, n, horizontalHaloSize,
                                                    verticalHaloSize, radius);
}
