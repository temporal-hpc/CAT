#pragma once

#include "GPUKernels.cuh"

namespace Temporal
{

class SHAREDSolver
{
    dim3 mainKernelsBlockSize;
    dim3 mainKernelsGridSize;

    dim3 boundaryKernelsBlockSize;
    dim3 boundaryKernelsGridSize;

  public:
    void fillHorizontalBoundaryConditions(char *inData, int n, int radius);
    void fillVerticalBoundaryConditions(char *inData, int n, int radius);

    void setBlockSize(int block_x = 16, int block_y = 16);
    void setGridSize(int n, int grid_z = 1);

    void CAStepAlgorithm(char *inData, char *outData, int n, int radius);
};

} // namespace Temporal