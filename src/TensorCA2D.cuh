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
#define MTYPE char // ⚠️ changing this also requires to change the convertXtoY kernels
#define FTYPE half
#define FTYPE half
#define CASTM2F(M) __uint2half_rn(M)
#define CASTF2M(F) __half2uint_rn(F)
// #define FTYPE unsigned char
// #define CASTM2F(M) (M)
// #define CASTF2M(F) (F)

#define HINDEX(x, y, nWithHalo) ((y + R) * ((size_t)nWithHalo) + (x + R))
#define FTYPE_ACC FTYPE
#define HALO_SIZE (2 * R)

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
    CLASSICGBMEM, //0
    CLASSICV1,  //1
    CLASSICV2,
    TENSORCA, //3
    TENSORCACOALESCED,
    CLASSICGBMEMHALF, //5
    TENSORCACOALESCEDMORETHREADS,
    TENSORCACOALESCEDLESSSHMEM, //7
    TENSORCACOALESCEDNOSHMEM,
    TENSORCACOALESCEDLESSSHMEMINT4,
    TENSORCACOALESCEDLESSSHMEMINT4V2,
    TENSORCACOALESCEDLESSSHMEMINT8,
    NOT_IMPLEMENTED
};

class TensorCA2D {
public:
    uint32_t n;
    uint32_t haloWidth;
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

    FTYPE* devDataPingTensor;
    FTYPE* devDataPongTensor;

    unsigned char* devDataPingTensorInt8;
    unsigned char* devDataPongTensorInt8;

    //uint32_t* devDataPingTensor;
   	//uint32_t* devDataPongTensor;
    int* devDataPingTensorInt4;
    MTYPE* devDataBufferTensor;
    int* devDataBufferTensorInt8;
    MTYPE* devDataBufferTensor2;

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
