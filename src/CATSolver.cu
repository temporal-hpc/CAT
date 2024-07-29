#include "CATSolver.cuh"
#include "GPUKernels.cuh"
using namespace Temporal;

CATSolver::CATSolver(int nRegionsH, int nRegionsV)
{
    this->m_nRegionsH = nRegionsH;
    this->m_nRegionsV = nRegionsV;
}

void CATSolver::setBlockSize(int block_x, int block_y)
{
    this->mainKernelsBlockSize[0] = block_x;
    this->mainKernelsBlockSize[1] = block_y;
    this->mainKernelsBlockSize[2] = 1;

    this->castingKernelsBlockSize[0] = block_x;
    this->castingKernelsBlockSize[1] = block_y;
    this->castingKernelsBlockSize[2] = 1;
}
void CATSolver::setGridSize(int n, int grid_z)
{
    this->mainKernelsGridSize[0] = (n + (m_nRegionsH * 16) - 1) / (m_nRegionsH * 16);
    this->mainKernelsGridSize[1] = (n + (m_nRegionsV * 16) - 1) / (m_nRegionsV * 16);
    this->mainKernelsGridSize[2] = grid_z;

    this->castingKernelsGridSize[0] = (n + this->castingKernelsBlockSize[0] - 1) / this->castingKernelsBlockSize[0];
    this->castingKernelsGridSize[1] = (n + this->castingKernelsBlockSize[1] - 1) / this->castingKernelsBlockSize[1];
    this->castingKernelsGridSize[2] = grid_z;
}

void CATSolver::changeLayout(uint8_t *inData, void *outData, int n, int radius)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], this->castingKernelsGridSize[2]);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], this->castingKernelsBlockSize[2]);
    convertFp16ToFp32AndUndoChangeLayout<<<grid, block>>>(inData, (half *)outData, n);
}

void CATSolver::unchangeLayout(void *inData, uint8_t *outData, int n, int radius)
{
    dim3 grid = dim3(this->castingKernelsGridSize[0], this->castingKernelsGridSize[1], this->castingKernelsGridSize[2]);
    dim3 block =
        dim3(this->castingKernelsBlockSize[0], this->castingKernelsBlockSize[1], this->castingKernelsBlockSize[2]);

    convertFp32ToFp16AndDoChangeLayout<<<grid, block>>>((half *)inData, outData, n);
}

void copyAndCast(uint8_t *inData, void *outData, int n, int radius)
{
}

void CATSolver::prepareData(uint8_t *inData, void *outData, int n, int radius)
{
    this->changeLayout(inData, outData, n, radius);
}

void CATSolver::unprepareData(void *inData, uint8_t *outData, int n, int radius)
{
    this->unchangeLayout(inData, outData, n, radius);
}

void CATSolver::StepSimulation(void *inData, void *outData, int n, int radius)
{
    dim3 grid = dim3(this->mainKernelsGridSize[0], this->mainKernelsGridSize[1], this->mainKernelsGridSize[2]);
    dim3 block = dim3(this->mainKernelsBlockSize[0], this->mainKernelsBlockSize[1], this->mainKernelsBlockSize[2]);
    CAT_KERNEL<<<grid, block>>>((half *)inData, (half *)outData, n, n + 2 * radius, radius, m_nRegionsH, m_nRegionsV);
    (cudaDeviceSynchronize());
}
