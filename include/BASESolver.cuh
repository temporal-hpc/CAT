#pragma once

#include "GPUKernels.cuh"

namespace Temporal
{

class BASESolver
{
  private:
    dim3 mainKernelsBlockSize;
    dim3 mainKernelsGridSize;

  public:
    void setBlockSize(int block_x = 32, int block_y = 32);
    void setGridSize(int n, int grid_z = 1);

    void CAStepAlgorithm(char *inData, char *outData, int n, int radius);
};

} // namespace Temporal