#define MTYPE char  // ⚠️ changing this also requires to change the convertXtoY kernels
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
#define NREGIONS_H 2  // ⚠️ Stored in an uint8_t
#endif
#ifndef NREGIONS_V
#define NREGIONS_V 4
#endif
#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif
