#pragma once
#include <stdint.h>
#include <vector>

namespace Temporal
{

// struct vec2
// {
//     int x, y;

//     vec2(int x = 0, int y = 0) : x(x), y(y)
//     {
//     }
// };

template <typename T> class Solver
{
  protected:
    int m_mainKernelsBlockSize[3];
    int m_mainKernelsGridSize[3];

    // vec2 m_halo[2];

  public:
    virtual void setBlockSize(int block_x = 16, int block_y = 16) = 0;
    virtual void setGridSize(int n, int halo, int grid_z = 1) = 0;

    // virtual void fillHorizontalBoundaryConditions(char *inData, int n, int radius) = 0;
    // virtual void fillVerticalBoundaryConditions(char *inData, int n, int radius) = 0;

    virtual void prepareData(uint8_t *inData[], T *outData[], int n, int halo, int radius) = 0;
    virtual void unprepareData(T *inData[], uint8_t *outData[], int n, int halo, int radius) = 0;

    virtual void StepSimulation(T *inData[], T *outData[], int n, int halo, int radius) = 0;
};
} // namespace Temporal