#pragma once

#include "Solver.cuh"

namespace Temporal
{

class PACKSolver : public Solver<uint64_t>
{
  private:
    int *CALookUpTable;

    int n;
    int packedN;
    int packedRadius;

    int boundaryKernelsBlockSize[3];
    int boundaryKernelsGridSize[3];

    void setupLookupTable(int radius);

  public:
    int elementsPerCel = 8;
    int radius;

    PACKSolver(int n, int radius);

    void setBlockSize(int block_x = 16, int block_y = 16);
    void setGridSize(int n, int grid_z);

    void packState(uint8_t *inData, uint64_t *outData, int n, int radius);
    void unpackState(uint64_t *inData, uint8_t *outData, int n, int radius);

    void prepareData(uint8_t *inData, uint64_t *outData, int n, int radius)
    {
    }

    void StepSimulation(uint64_t *inData, uint64_t *outData, int n, int radius);
};

} // namespace Temporal