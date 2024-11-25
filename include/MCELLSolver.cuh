#pragma once

#include "Solver.cuh"

namespace Temporal
{

class MCELLSolver : public Solver<uint8_t>
{
  private:
    int boundaryKernelsBlockSize[2];
    int boundaryKernelsGridSize[2];

    int cellsPerThread = 2;
    int radius;

  public:
    MCELLSolver(int radius, int SMIN, int SMAX, int BMIN, int BMAX);

    void setBlockSize(int block_x = 32, int block_y = 32) override;
    void prepareGrid(int n, int halo) override;

    void prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override {};
    void unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override {};

    void StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
};

} // namespace Temporal