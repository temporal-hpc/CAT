#pragma once

#include "Solver.cuh"
#include <stdint.h>

namespace Temporal
{
class CATSolver : public Solver<void>
{
  protected:
    int m_nRegionsH;
    int m_nRegionsV;
    size_t m_sharedMemoryBytes;
    int castingKernelsBlockSize[3];
    int castingKernelsGridSize[3];

    void changeLayout(uint8_t *inData[], void *outData[], int n, int halo);
    void unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo);

  public:
    CATSolver(int nRegionsH, int nRegionsV);

    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void setGridSize(int n, int halo, int grid_z = 1) override;

    void prepareData(uint8_t *inData[], void *outData[], int n, int halo, int radius) override;
    void unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius) override;
    void StepSimulation(void *inData[], void *outData[], int n, int halo, int radius) override;
};

} // namespace Temporal
