#pragma once

#include "Solver.cuh"

namespace Temporal
{

class MCELLSolver : public Solver<uint8_t>
{
  private:
    int boundaryKernelsBlockSize[3];
    int boundaryKernelsGridSize[3];

    int cellsPerThread = 2;
    int radius;

  public:
    MCELLSolver(int radius);

    void setBlockSize(int block_x = 32, int block_y = 32) override;
    void setGridSize(int n, int grid_z = 1) override;

    void prepareData(uint8_t *inData, uint8_t *outData, int n, int radius) {};

    void StepSimulation(uint8_t *inData, uint8_t *outData, int n, int radius) override;
};

} // namespace Temporal