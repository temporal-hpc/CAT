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
    CA1D,
    CA2D,
    CA3D,
    NOT_IMPLEMENTED
};

class TensorCA {
public:
    uint32_t n;
    uint32_t nWithHalo;
    size_t nElements;

    uint32_t deviceId;
    uint32_t dimensions;
    float density;
    uint32_t seed;

    Mode mode;

    dim3 GPUBlock;
    dim3 GPUGrid;

    bool hasBeenAllocated;

    MTYPE* hostData;
    MTYPE* devData;

    TensorCA(uint32_t deviceId, uint32_t n, uint32_t dimensions, float density, uint32_t seed);
    ~TensorCA();

    static bool compare(TensorCA* a, TensorCA* b);
    bool init();
    void allocateMemory();
    void reset();
    void freeMemory();
    void transferHostToDevice();
    void transferDeviceToHost();

    void printHostData();
    void printDeviceData();

    float doBenchmarkAction(uint32_t nTimes);
};
