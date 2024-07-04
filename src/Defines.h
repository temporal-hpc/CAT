#pragma once

#define MTYPE uint8_t
#define FTYPE half
#define CASTM2F(M) __uint2half_rn(M)
#define CASTF2M(F) __half2uint_rn(F)

#define HINDEX(x, y, nWithHalo) ((y + RADIUS) * ((size_t)nWithHalo) + (x + RADIUS))
#define FTYPE_ACC FTYPE
#define HALO_SIZE (2 * RADIUS)

// These control how many regions of 16x16 (fragsize) each block processes.
// if NREGIONS_H*16>n or NREGIONS_V*16>n then it will be fixed to meet the condition
#ifndef NREGIONS_H
#define NREGIONS_H 2
#endif
#ifndef NREGIONS_V
#define NREGIONS_V 4
#endif
#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

#define BEST_CELLS_PER_THREAD (2)

#define ELEMENTS_PER_CELL 8
#define CAGIGAS_CELL_NEIGHBOURS ((RADIUS * 2 + 1) * (RADIUS * 2 + 1) - 1)

#include <iostream>

#define gpuErrchk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
