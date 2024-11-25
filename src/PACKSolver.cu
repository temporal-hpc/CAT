#include "GPUKernels.cuh"
#include "PACKSolver.cuh"

using namespace Temporal;

PACKSolver::PACKSolver(int halo, int SMIN, int SMAX, int BMIN, int BMAX) : Solver(SMIN, SMAX, BMIN, BMAX)
{
    this->m_halo = halo;
    setupLookupTable(halo);
}

void PACKSolver::setBlockSize(int block_x, int block_y)
{
    this->m_mainKernelsBlockSize[0] = block_x;
    this->m_mainKernelsBlockSize[1] = block_y;

    this->boundaryKernelsBlockSize[0] = 256;
    this->boundaryKernelsBlockSize[1] = 1;

    this->m_sharedMemoryBytes = (block_x + 2 * (int)ceil(this->m_halo / this->elementsPerCel)) *
                                (block_y + 2 * this->m_halo) * sizeof(uint64_t);
}

void PACKSolver::prepareGrid(int n, int halo)
{
    int innerHorizontalSize = n / this->elementsPerCel;
    int innerVerticalSize = n;
    // int fullVerticalSize = n + 2 * radius;

    this->m_mainKernelsGridSize[0] = (int)ceil(innerHorizontalSize / (float)this->m_mainKernelsBlockSize[0]);
    this->m_mainKernelsGridSize[1] = (int)ceil(innerVerticalSize / (float)this->m_mainKernelsBlockSize[1]);
}

void PACKSolver::setupLookupTable(int radius)
{
    int cagigas_cell_neigh = ((radius * 2 + 1) * (radius * 2 + 1) - 1);

    cudaMalloc(&CALookUpTable, sizeof(int) * 2 * (cagigas_cell_neigh + 1));
    dim3 block = dim3(1, 2, 1);
    dim3 grid = dim3((cagigas_cell_neigh + 1), 1, 1);
    kernel_init_lookup_table<<<grid, block>>>(CALookUpTable, radius, SMIN, SMAX, BMIN, BMAX);
}

void PACKSolver::unpackState(uint64_t *inData, uint8_t *outData, int n, int halo, int radius, int nTiles)
{
    int innerHorizontalSize = n / this->elementsPerCel;
    int innerVerticalSize = n;
    int verticalHaloSize = radius;
    int horizontalHaloSize = ceil(halo / this->elementsPerCel);

    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);
    unpackStateKernel<<<grid, block>>>(inData, outData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                       verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACKSolver::packState(uint8_t *inData, uint64_t *outData, int n, int halo, int radius, int nTiles)
{
    int innerHorizontalSize = n / this->elementsPerCel;
    int innerVerticalSize = n;
    int verticalHaloSize = radius;
    int horizontalHaloSize = ceil(halo / this->elementsPerCel);
    dim3 grid = dim3(this->m_mainKernelsGridSize[0], this->m_mainKernelsGridSize[1], nTiles);
    dim3 block =
        dim3(this->m_mainKernelsBlockSize[0], this->m_mainKernelsBlockSize[1], 1);

    packStateKernel<<<grid, block>>>(inData, outData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                     verticalHaloSize);
    gpuErrchk(cudaDeviceSynchronize());
}

void PACKSolver::fillHorizontalBoundaryConditions(uint64_t *inData, int n, int radius)
{
    int innerHorizontalSize = n / this->elementsPerCel;
    int innerVerticalSize = n;
    int verticalHaloSize = radius;
    int horizontalHaloSize = ceil(radius / this->elementsPerCel);
    dim3 horizontalBoundaryGrid = dim3((int)ceil(innerHorizontalSize / (float)this->boundaryKernelsBlockSize[0]));

    dim3 block =
        dim3(this->boundaryKernelsBlockSize[0], this->boundaryKernelsBlockSize[1], 1);
    ghostRows<<<horizontalBoundaryGrid, block>>>(inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                                 verticalHaloSize);
}

void PACKSolver::fillVerticalBoundaryConditions(uint64_t *inData, int n, int radius)
{
    int innerHorizontalSize = n / this->elementsPerCel;
    int innerVerticalSize = n;
    int verticalHaloSize = radius;
    int horizontalHaloSize = ceil(radius / this->elementsPerCel);

    int fullVerticalSize = n + 2 * radius;
    dim3 verticalBoundaryGrid = dim3((int)ceil(fullVerticalSize / (float)this->boundaryKernelsBlockSize[0]));
    dim3 block =
        dim3(this->boundaryKernelsBlockSize[0], this->boundaryKernelsBlockSize[1], 1);

    ghostCols<<<verticalBoundaryGrid, block>>>(inData, innerHorizontalSize, innerVerticalSize, horizontalHaloSize,
                                               verticalHaloSize);
}

void PACKSolver::StepSimulation(uint64_t *inData[], uint64_t *outData[], int n, int halo, int radius, int nTiles)
{
    int verticalHaloSize = halo;
    int horizontalHaloSize = ceil(halo / this->elementsPerCel);

    dim3 grid = dim3(m_mainKernelsGridSize[0], m_mainKernelsGridSize[1], nTiles);
    dim3 block = dim3(m_mainKernelsBlockSize[0], m_mainKernelsBlockSize[1], 1);
    // PACK_KERNEL<<<grid, block, m_sharedMemoryBytes>>>(inData, outData, this->CALookUpTable, n / 8, n,
    //                                                   horizontalHaloSize, verticalHaloSize, radius);
    (cudaDeviceSynchronize());
}
