#include "GPUBenchmark.cuh"

#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

GPUBenchmark::GPUBenchmark(CASolver *pSolver, int n, int pSteps, int pSeed, float pDensity)
{
    solver = pSolver;
    this->n = n;
    steps = pSteps;
    seed = pSeed;
    density = pDensity;

    timer = new CudaTimer();
    stats = new StatsCollector();
}

void GPUBenchmark::reset()
{
    int s = rand() % 1000000;

    lDebug(1, "Resetting state:");
    solver->resetState(s, density);
}

void GPUBenchmark::run()
{
    srand(seed);
    reset();
    lDebug(1, "Benchmark started");
    // WARMUP for STEPS/4
    // solver->doSteps(steps >> 2);

    lDebug(1, "Initial state:");
    if (n <= PRINT_LIMIT)
    {
        fDebug(1, solver->printCurrentState());
    }

#ifdef MEASURE_POWER
    GPUPowerBegin("0", 100);
#endif

    doOneRun();

#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif

    lDebug(1, "Benchmark finished. Results:");

    if (n <= PRINT_LIMIT)
    {
        fDebug(1, solver->printCurrentState());
    }
}
void GPUBenchmark::doOneRun()
{
    timer->start();

    solver->doSteps(steps);

    timer->stop();
    registerElapsedTime(timer->getElapsedTimeMiliseconds() / steps);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void GPUBenchmark::registerElapsedTime(float milliseconds)
{
    stats->add(milliseconds);
}

StatsCollector *GPUBenchmark::getStats()
{
    return stats;
}
