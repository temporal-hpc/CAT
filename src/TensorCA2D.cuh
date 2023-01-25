#pragma once

#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// Lazy Fix
#define MTYPE uint32_t // ⚠️ changing this also requires to change the convertXtoY kernels
#define FTYPE half
#define FTYPE_ACC FTYPE
#define HALO_SIZE 2

// These control how many regions of 16x16 (fragsize) each block processes.
// if NREGIONS_H*16>n or NREGIONS_V*16>n then it will be fixed to meet the condition
#ifndef NREGIONS_H
#define NREGIONS_H 2 // ⚠️ Stored in an uint8_t
#endif
#ifndef NREGIONS_V
#define NREGIONS_V 4
#endif
#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

#include "Debug.h"

enum class Mode {
    CLASSICGBMEM,
    CLASSICV1,
    CLASSICV2,
    TENSORCA,
    TENSORCACOALESCED,
    CLASSICGBMEMHALF,
    TENSORCACOALESCEDMORETHREADS,
    TENSORCACOALESCEDLESSSHMEM,
    TENSORCACOALESCEDNOSHMEM,
    NOT_IMPLEMENTED
};

class TensorCA2D {
public:
    uint32_t n;
    uint32_t nWithHalo;
    size_t nElements;
    uint32_t haloWidth;

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

    FTYPE* devDataPingTensor;
    FTYPE* devDataPongTensor;
    MTYPE* devDataBufferTensor;

    // auto stepKernel;

    TensorCA2D(uint32_t deviceId, uint32_t n, uint32_t modeCode, float density);
    ~TensorCA2D();

    bool compare(TensorCA2D* a);
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
