#ifndef DEFINES_H
#define DEFINES_H

#include <algorithm>

#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

#ifdef _WIN32
typedef __uint64_t uint;
#endif

#define FETCHTIME(STR,END) \
(double)((END.tv_sec  - STR.tv_sec) * 1000000u + END.tv_usec - STR.tv_usec) / 1.e6;

#ifndef USE_DOUBLE
  #ifndef USE_FLOAT
    typedef double  PrecisionType;
    const size_t BW        = 2;        //  Boundary width
  #endif
#endif

#ifdef USE_DOUBLE
  typedef double  PrecisionType;
  const size_t BW          = 8;        //  Boundary width
#endif

#ifdef USE_FLOAT
  typedef float  PrecisionType;
  const size_t BW          = 16;       //  Boundary width
#endif

const size_t MAX_DIM       = 3;
const size_t BWP           = BW / 2;   //  Boundary padding
const PrecisionType ONESIX = 1.0/6.0;

#define DIMENSION(A) sizeof(A)/sizeof(PrecisionType)

enum Flag {
    FIXED_VELOCITY_X = 0x000001,
    FIXED_VELOCITY_Y = 0x000010,
    FIXED_VELOCITY_Z = 0x000100,
    FIXED_PRESSURE   = 0x001000
};

#endif
