#include "CellularAutomata/GPUBenchmark.cuh"

GPUBenchmark::GPUBenchmark(CASolver* pSolver, int pSteps) {
    solver = pSolver;
    steps = pSteps;
}

void GPUBenchmark::reset(int seed, float density) {
    solver->resetState(seed, density);
}

void GPUBenchmark::run() {
    solver->doSteps(steps);
}
