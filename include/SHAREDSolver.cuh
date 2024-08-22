#pragma once

#include "Solver.cuh"

namespace Temporal
{

class SHAREDSolver : public Solver<uint8_t>
{

  public:
    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void setGridSize(int n, int halo, int grid_z = 1) override;

    void prepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius) override;

    void unprepareData(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius) override;

    void StepSimulation(uint8_t *inData[], uint8_t *outData[], int n, int halo, int radius) override;
};

} // namespace Temporal