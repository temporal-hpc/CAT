#pragma once

#include "GPUKernels.cuh"

namespace Temporal
{
class CATSolver
{
  protected:
    int m_nRegionsH;
    int m_nRegionsV;

    dim3 mainKernelsBlockSize;
    dim3 mainKernelsGridSize;

    dim3 castingKernelsBlockSize;
    dim3 castingKernelsGridSize;

    void setBlockSize(int block_x = 16, int block_y = 16);
    void setGridSize(int n, int nRegionsH = 1, int nRegionsV = 1, int grid_z = 1);

  public:
    void changeLayout(half *inData, half *outData, int n, int radius);
    void unchangeLayout(half *inData, half *outData, int n, int radius);

    void CAStepAlgorithm(half *inData, half *outData, int n, int radius);
};

} // namespace Temporal
