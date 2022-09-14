#include "StatsCollector.hpp"
#include <cinttypes>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "TensorCA.cuh"

#define PRINT_LIMIT 6
const uint32_t STEPS = 5000;

int main(int argc, char** argv) {
    // srand ( time(NULL) );
    if (argc != 7) {
        printf("run as ./prog <deviceId> <n> <dimensions> <repeats> <density> <seed>\n");
        exit(1);
    }
    debugInit(5, "log.txt");
    uint32_t deviceId = atoi(argv[1]);
    uint32_t n = atoi(argv[2]);
    uint32_t dimensions = atoi(argv[3]);
    uint32_t repeats = atoi(argv[4]);
    float density = atof(argv[5]);
    uint32_t seed = atoi(argv[6]);

    StatsCollector stats;
    TensorCA* benchmark;

    for (int i = 0; i < repeats; i++) {
        benchmark = new TensorCA(deviceId, n, dimensions, density, seed);
        if (!benchmark->init()) {
            exit(1);
        }
        float iterationTime = benchmark->doBenchmarkAction(STEPS);
        benchmark->transferDeviceToHost();
        stats.add(iterationTime);
        if (i != repeats - 1) {
            delete benchmark;
        }
    }

    fDebug(1, benchmark->printDeviceData());

    // #ifdef VERIFY
    //     TensorCA* reference = new TensorCA(deviceId, n, 0);
    //     if (!reference->init()) {
    //         exit(1);
    //     }
    //     reference->doBenchmarkAction(STEPS);
    //     reference->transferDeviceToHost();

    //     if (!TensorCA::compare(benchmark, reference)) {
    //         printf("\n[VERIFY] verification FAILED!.\n\n");

    //         exit(1);
    //     }

    //     printf("\n[VERIFY] verification successful.\n\n");

    // #endif

#ifdef DEBUG
    printf("maxlong %lu\n", LONG_MAX);
    printf("\x1b[1m");
    fflush(stdout);
    printf("main(): avg kernel time: %f ms\n", stats.getAverage());
    printf("\x1b[0m");
    fflush(stdout);
#else
    printf("%f, %f, %f, %f\n", stats.getAverage(), stats.getStandardDeviation(), stats.getStandardError(), stats.getVariance());
#endif
}
