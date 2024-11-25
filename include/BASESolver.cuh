#pragma once

#include "Solver.cuh"

namespace Temporal
{

class BASESolver : public Solver<uint8_t>
{
private:
  public:
    BASESolver(int SMIN, int SMAX, int BMIN, int BMAX) : Solver(SMIN, SMAX, BMIN, BMAX) {}

    void setBlockSize(int block_x = 32, int block_y = 32) override;
    void prepareGrid(int n, int halo) override;

    void prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
    void unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;

    void StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
};

} // namespace Temporal