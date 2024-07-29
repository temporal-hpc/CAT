#pragma once

#include "Solver.cuh"

namespace Temporal
{
class COARSESolver : public Solver<uint8_t>
{
  public:
    void setBlockSize(int block_x = 16, int block_y = 16);
    void setGridSize(int n, int grid_z = 1);
    void prepareData(uint8_t *inData, uint8_t *outData, int n, int radius)
    {
    }
    void StepSimulation(uint8_t *inData, uint8_t *outData, int n, int radius);
};

} // namespace Temporal