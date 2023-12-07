#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <cinttypes>
#include "StatsCollector.hpp"

#include "CellularAutomata/CADataDomainComparator.cuh"
#include "CellularAutomata/CASolverFactory.cuh"
#include "CellularAutomata/GPUBenchmark.cuh"
#include "TensorCA2D.cuh"

#define PRINT_LIMIT 7
#ifndef R
#define R 1
#endif
// change to runtime parameter
const uint32_t STEPS = 3;

int main(int argc, char** argv) {
    // srand ( time(NULL) );
    if (argc != 7) {
        printf("run as ./prog <deviceId> <n> <mode> <repeats> <density> <seed>\n");
        exit(1);
    }
    //     debugInit(5, "log.txt");
    uint32_t deviceId = atoi(argv[1]);
    uint32_t n = atoi(argv[2]);
    uint32_t mode = atoi(argv[3]);
    uint32_t repeats = atoi(argv[4]);
    float density = atof(argv[5]);
    uint32_t seed = atoi(argv[6]);

    CASolver* solver = CASolverFactory::createSolver(mode, n, R);
    if (solver == nullptr) {
        printf("main(): solver is NULL\n");
        exit(1);
    }
    GPUBenchmark* benchmark = new GPUBenchmark(solver, STEPS);
    benchmark->reset(seed, density);
    benchmark->run();

    CASolver* referenceSolver = CASolverFactory::createSolver(0, n, R);
    if (referenceSolver == nullptr) {
        printf("main(): solver is NULL\n");
        exit(1);
    }
    GPUBenchmark* referenceBenchmark = new GPUBenchmark(referenceSolver, STEPS);
    referenceBenchmark->reset(seed, density);
    referenceBenchmark->run();

    CADataDomainComparator* comparator = new CADataDomainComparator();

    //      StatsCollector stats;
    //      TensorCA2D* benchmark;

    //     for (int i = 0; i < repeats; i++) {
    //         benchmark = new TensorCA2D(deviceId, n, mode, density);
    //         if (!benchmark->init(seed)) {
    //             exit(1);
    //         }
    //         float iterationTime = benchmark->doBenchmarkAction(STEPS);
    //         // benchmark->transferDeviceToHost();
    //         stats.add(iterationTime);
    //         if (i != repeats - 1) {
    //             delete benchmark;
    //         }
    //     }

    //     benchmark->transferDeviceToHost();
    //     fDebug(1, benchmark->printHostData());

    // #ifdef VERIFY
    //     TensorCA2D* reference = new TensorCA2D(deviceId, n, 0, density);
    //     if (!reference->init(seed)) {
    //         exit(1);
    //     }
    //     reference->doBenchmarkAction(STEPS);
    //     reference->transferDeviceToHost();
    //     fDebug(1, reference->printHostData());

    //     printf("main(): avg kernel time: %f ms\n", stats.getAverage());
    //     printf("\x1b[0m");
    //     fflush(stdout);
    //     if (!reference->compare(benchmark)) {
    //         printf("\n[VERIFY] verification FAILED!.\n\n");

    //         exit(1);
    //     }

    //     printf("\n[VERIFY] verification successful.\n\n");

    // #endif

    // #ifdef DEBUG
    //     printf("maxlong %lu\n", LONG_MAX);
    //     printf("\x1b[1m");
    //     fflush(stdout);
    //     printf("main(): avg kernel time: %f ms\n", stats.getAverage());
    //     printf("\x1b[0m");
    //     fflush(stdout);
    // #else
    //     printf("%f, %f, %f, %f\n", stats.getAverage(), stats.getStandardDeviation(), stats.getStandardError(), stats.getVariance());
    // #endif
}
