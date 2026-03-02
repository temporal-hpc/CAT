#pragma once

#include "Solver.cuh"
#include <stdint.h>
#include <cuda_runtime.h>

namespace Temporal
{

class CATMultiStepSolver2 : public Solver<void>
{
  protected:
    int    m_nRegionsH;
    int    m_nRegionsV;
    size_t m_sharedMemoryBytes;

    int castingKernelsBlockSize[2];
    int castingKernelsGridSize[2];

    cudaStream_t m_stream;           
    int          m_l2CacheSize;      
    int          m_maxPersistingL2;  
    bool         m_l2PersistenceOn;

    uint8_t *d_scratchBuffer; 

    void changeLayout  (uint8_t *inData[], void *outData[], int n, int halo, int nTiles);
    void unchangeLayout(void *inData[], uint8_t *outData[], int n, int halo, int nTiles);

  public:
    CATMultiStepSolver2(int nRegionsH, int nRegionsV, int SMIN, int SMAX, int BMIN, int BMAX);
    ~CATMultiStepSolver2();

    void configureL2ForData(size_t bytes);
    void resetL2Persistence();

    void fillPeriodicBoundaryConditions(void *data[], int n, int halo, int nTiles);

    void setBlockSize(int block_x = 16, int block_y = 16) override;
    void prepareGrid (int n, int halo)                     override;

    void prepareData  (uint8_t *inData[], void *outData[], int n, int halo, int radius, int nTiles) override;
    void unprepareData(void *inData[], uint8_t *outData[], int n, int halo, int radius, int nTiles) override;
    void StepSimulation(void *inData[], void *outData[], int n, int halo, int radius, int nTiles)   override;

    void StepSimulationMulti(void *inData[], void *outData[], int n, int halo, int radius,
                             int nTiles, int innerSteps);

    cudaStream_t getStream() const { return m_stream; }
};

} // namespace Temporal
