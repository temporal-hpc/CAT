#include "include/CATSolver.cuh"

void CATSolver::setBlockSize(int block_x = 16, int block_y = 16)
{
    this->mainKernelsBlockSize = dim3(block_x, block_y);
    this->castingKernelsBlockSize = dim3(block_x, block_y);
}
void CATSolver::setGridSize(int n, int nRegionsH = 1, int nRegionsV = 1, int grid_z = 1)
{
    this->mainKernelsGridSize =
        dim3((n + (nRegionsH * 16) - 1) / (nRegionsH * 16), (n + (nRegionsV * 16) - 1) / (nRegionsV * 16));
    this->castingKernelsGridSize =
        dim3((n + this->castingKernelsBlockSize.x - 1) / this->castingKernelsBlockSize.x,
             (n + this->castingKernelsBlockSize.y - 1) / this->castingKernelsBlockSize.y, grid_z);
}

void CATSolver::changeLayout(half *inData, half *outData, int n, int radius)
{
    convertFp16ToFp32AndUndoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(inData,
                                                                                                          outData, n);
}

void CATSolver::unchangeLayout(half *inData, half *outData, int n, int radius)
{
    convertFp32ToFp16AndDoChangeLayout<<<this->castingKernelsGridSize, this->castingKernelsBlockSize>>>(inData, outData,
                                                                                                        n);
}

void CATSolver::CAStepAlgorithm(half *inData, half *outData, int n, int radius)
{
    CAT_KERNEL<<<this->mainKernelsGridSize, this->mainKernelsBlockSize>>>(inData, outData, n, n + 2 * radius);
    (cudaDeviceSynchronize());
}
