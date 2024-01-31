#pragma once
#include <cuda_runtime.h>
#include <random>
#include "CellularAutomata/Solvers/CASolver.cuh"
#include "CudaTimer.cuh"
#include "Debug.h"
#include "Defines.h"
#include "StatsCollector.hpp"
#define PRINT_LIMIT (512)

class GPUBenchmark {
   private:
    int steps;
    int n;
    int repeats;
    int seed;
    float density;

    CASolver* solver;
    CudaTimer* timer;

    StatsCollector* stats;

    void doOneRun();
    void registerElapsedTime(float milliseconds);
    void reset();

   public:
    GPUBenchmark(CASolver* pSolver, int n, int pRepeats, int pSteps, int pSeed, float pDensity);

    void run();
    StatsCollector* getStats();
};
