#pragma once
#include <stdint.h>

namespace Temporal
{

template <typename T> class Solver
{
  protected:
    int mainKernelsBlockSize[3];
    int mainKernelsGridSize[3];

  public:
    virtual void setBlockSize(int block_x = 16, int block_y = 16) = 0;
    virtual void setGridSize(int n, int grid_z = 1) = 0;

    // virtual void fillHorizontalBoundaryConditions(char *inData, int n, int radius) = 0;
    // virtual void fillVerticalBoundaryConditions(char *inData, int n, int radius) = 0;

    virtual void prepareData(uint8_t *inData, T *outData, int n, int radius) = 0;

    virtual void StepSimulation(T *inData, T *outData, int n, int radius) = 0;
};
} // namespace Temporal