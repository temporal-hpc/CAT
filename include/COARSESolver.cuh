#pragma once

#include "Solver.cuh"

namespace Temporal
{
class COARSESolver : public Solver<uint8_t>
{
  public:

    COARSESolver(int SMIN, int SMAX, int BMIN, int BMAX) : Solver(SMIN, SMAX, BMIN, BMAX) {}

    void setBlockSize(int block_x = 16, int block_y = 16);
    void prepareGrid(int n, int halo = 1);
    void prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
    {
    }
    void unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles)
    {
    }
    void StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles);
};

} // namespace Temporal