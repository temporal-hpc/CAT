#pragma once

#include "Solver.cuh"

namespace Temporal
{

class PACKSolver : public Solver<uint64_t>
{
  private:
    int *CALookUpTable;

    size_t m_sharedMemoryBytes;
    int m_halo;

    int boundaryKernelsBlockSize[3];
    int boundaryKernelsGridSize[3];

    void setupLookupTable(int radius);

  public:
    float elementsPerCel = 8.0f;

    PACKSolver(int halo);

    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void setGridSize(int n, int halo, int grid_z) override;

    void packState(uint8_t *inData, uint64_t *outData, int n, int halo, int radius);
    void unpackState(uint64_t *inData, uint8_t *outData, int n, int halo, int radius);
    void fillVerticalBoundaryConditions(uint64_t *inData, int n, int radius);
    void fillHorizontalBoundaryConditions(uint64_t *inData, int n, int radius);

    void prepareData(uint8_t *inData[], uint64_t *outData[], int n, int halo, int radius) override
    {
        // packState(inData, outData, n, halo, radius);
    }

    void unprepareData(uint64_t *inData[], uint8_t *outData[], int n, int halo, int radius) override
    {
        // unpackState(inData, outData, n, halo, radius);
    }

    void StepSimulation(uint64_t *inData[], uint64_t *outData[], int n, int halo, int radius) override;
};

} // namespace Temporal