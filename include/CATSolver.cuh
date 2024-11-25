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
    int castingKernelsBlockSize[2];
    int castingKernelsGridSize[2];

    void changeLayout(uint8_t *inData[], void *outData[], int n, int halo, int nTiles);
    void unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo, int nTiles);

  public:
    CATSolver(int nRegionsH, int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX);

    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void prepareGrid(int n, int halo) override;

    void prepareData(uint8_t *inData[], void *outData[], int n, int halo, int radius, int nTiles) override;
    void unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
    void StepSimulation(void *inData[], void *outData[], int n, int halo, int radius, int nTiles) override;
};

} // namespace Temporal
