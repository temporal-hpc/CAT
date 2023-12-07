#pragma once
#include <random>
#include "CellularAutomata/Solvers/CASolver.cuh"

class GPUBenchmark {
   private:
    int steps;
    CASolver* solver;

   public:
    GPUBenchmark(CASolver* pSolver, int pSteps);

    void reset(int seed, float density);
    void run();
};