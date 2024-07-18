#pragma once

#include "GPUKernels.cuh"

namespace Temporal
{

class PACKSolver
{
  private:
    int *CALookUpTable;

    dim3 mainKernelsBlockSize;
    dim3 mainKernelsGridSize;

    dim3 boundaryKernelsBlockSize;
    dim3 boundaryKernelsGridSize;

    void setupLookupTable();
    void setBlockSize(int block_x = 16, int block_y = 16);
    void setGridSize(int n, int radius, int grid_z);

    void packState(char *inData, u_int64_t *outData, int n, int radius);
    void unpackState(uint64_t *inData, char *outData, int n, int radius);
    void fillHorizontalBoundaryConditions(uint64_t *inData, int n, int radius);
    void fillVerticalBoundaryConditions(uint64_t *inData, int n, int radius);

    void CAStepAlgorithm(uint64_t *inData, uint64_t *outData, int n, int radius);

  public:
    int elementsPerCel = 8;
};

} // namespace Temporal