#pragma once

#include "Solver.cuh"

namespace Temporal
{

class SHAREDSolver : public Solver<uint8_t>
{

  public:
    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void prepareGrid(int n, int halo) override;

    void prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;

    void unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;

    void StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
};

} // namespace Temporal