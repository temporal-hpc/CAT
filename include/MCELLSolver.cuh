#pragma once

#include "GPUKernels.cuh"

namespace Temporal
{

class MCELLSolver
{
  private:
    dim3 mainKernelsBlockSize;
    dim3 mainKernelsGridSize;

    dim3 boundaryKernelsBlockSize;
    dim3 boundaryKernelsGridSize;

    int cellsPerThread = 2;

  public:
    void fillHorizontalBoundaryConditions(char *inData, int n, int radius);
    void fillVerticalBoundaryConditions(char *inData, int n, int radius);

    void setBlockSize(int block_x = 32, int block_y = 32);
    void setGridSize(int n, int radius, int grid_z = 1);

    void CAStepAlgorithm(char *inData, char *outData, int n, int radius);
};

} // namespace Temporal