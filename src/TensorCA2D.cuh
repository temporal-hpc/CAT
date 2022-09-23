#pragma once

#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// Lazy Fix
#define MTYPE uint32_t
#define HALO_SIZE 2

#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

#include "Debug.h"

enum class Mode {
    CLASSICV1,
    CLASSICV2,
    TENSORCA,
    NOT_IMPLEMENTED
};

class TensorCA2D {
public:
    uint32_t n;
    uint32_t nWithHalo;
    size_t nElements;

    uint32_t deviceId;
    float density;
    uint32_t seed;

    Mode mode;

    dim3 GPUBlock;
    dim3 GPUGrid;

    bool hasBeenAllocated;

    MTYPE* hostData;
    MTYPE* devDataPing;
    MTYPE* devDataPong;

    // auto stepKernel;

    TensorCA2D(uint32_t deviceId, uint32_t n, uint32_t modeCode, float density);
    ~TensorCA2D();

    static bool compare(TensorCA2D* a, TensorCA2D* b);
    bool init(uint32_t seed);
    void allocateMemory();
    void reset();
    bool isInHalo(size_t i);
    void freeMemory();
    void transferHostToDevice();
    void transferDeviceToHost();

    void printHostData();
    void printDeviceData();

    float doBenchmarkAction(uint32_t nTimes);
};
