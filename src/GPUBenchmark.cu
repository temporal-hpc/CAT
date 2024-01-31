#include "GPUBenchmark.cuh"

GPUBenchmark::GPUBenchmark(CASolver* pSolver, int n, int pRepeats, int pSteps, int pSeed, float pDensity) {
    solver = pSolver;
    this->n = n;
    steps = pSteps;
    repeats = pRepeats;
    seed = pSeed;
    density = pDensity;

    timer = new CudaTimer();
    stats = new StatsCollector();
}

void GPUBenchmark::reset() {
    solver->resetState(seed, density);
}

void GPUBenchmark::run() {
    for (int i = 0; i < repeats; i++) {
        reset();
        lDebug(1, "Benchmark started. Initial state:");
        if (n <= PRINT_LIMIT) {
            fDebug(1, solver->printCurrentState());
        }
        doOneRun();
        lDebug(1, "Benchmark finished. Results:");
        if (n <= PRINT_LIMIT) {
            fDebug(1, solver->printCurrentState());
        }
    }
}
void GPUBenchmark::doOneRun() {
    timer->start();
    solver->doSteps(steps);
    timer->stop();
    registerElapsedTime(timer->getElapsedTimeMiliseconds() / steps);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

void GPUBenchmark::registerElapsedTime(float milliseconds) {
    stats->add(milliseconds);
}

StatsCollector* GPUBenchmark::getStats() {
    return stats;
}
