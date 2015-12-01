#ifndef SIMD_H
#define SIMD_H

#include "defines.hpp"

#ifdef USE_NOVEC
    #define VEC_ERROR \
        printf("Error: Explicit vectorization not enabled, please compile with USE_SSE2 or USE_AVX2 flags.\n");

    #define ALIGN 1
    #define VP    1
    #define VADD(A,B)             (A) + (B)
    #define VMUL(A,B)             (A) * (B)
    #define VLOAD( A)             (A[0])
    #define VSTORE(A,B)           (A[0]) = B
    #define VFMA(A,B,C)           (((A) * (B)) + (C))

    typedef double VectorType;
    const VectorType mmONESIX = 1.0f/6.0f;
#endif
#ifdef USE_SSE2
  #ifdef USE_DOUBLE
    #define ALIGN 16
    #define VP    2
    #define VADD(A,B)             _mm_add_pd((A),(B))
    #define VMUL(A,B)             _mm_mul_pd((A),(B))
    #define VLOAD( A)             _mm_load_pd((A))
    #define VSTORE(A,B)           _mm_store_pd((A),(B))
    #define VFMA(A,B,C)           _mm128_fmadd_pd((A),(B),(C))

    typedef __m128d VectorType;
    const VectorType mmONESIX =   _mm_set_pd(ONESIX,ONESIX);
  #endif

  #ifdef USE_FLOAT
    #define ALIGN 16
    #define VP    4
    #define VADD(A,B)             _mm_add_ps((A),(B))
    #define VMUL(A,B)             _mm_mul_ps((A),(B))
    #define VLOAD( A)             _mm_load_ps((A))
    #define VSTORE(A,B)           _mm_store_ps((A),(B))
    #define VFMA(A,B,C)           _mm128_fmadd_ps((A),(B),(C))

    typedef __m128 VectorType;
    const VectorType mmONESIX =   _mm_set_ps(ONESIX,ONESIX,ONESIX,ONESIX);
  #endif
#endif
#ifdef USE_AVX2
  #ifdef USE_DOUBLE
    #define ALIGN 32
    #define VP    4
    #define VADD(A,B)             _mm256_add_pd((A),(B))
    #define VMUL(A,B)             _mm256_mul_pd((A),(B))
    #define VLOAD( A)             _mm256_load_pd((A))
    #define VSTORE(A,B)           _mm256_store_pd((A),(B))
    #define VFMA(A,B,C)           _mm256_fmadd_pd((A),(B),(C))

    typedef __m256d VectorType;
    const VectorType mmONESIX =   _mm256_set_pd(ONESIX,ONESIX,ONESIX,ONESIX);
  #endif

  #ifdef USE_FLOAT
    #define ALIGN 32
    #define VP    8
    #define VADD(A,B)             _mm256_add_ps((A),(B))
    #define VMUL(A,B)             _mm256_mul_ps((A),(B))
    #define VLOAD( A)             _mm256_load_ps((A))
    #define VSTORE(A,B)           _mm256_store_ps((A),(B))
    #define VFMA(A,B,C)           _mm256_fmadd_ps((A),(B),(C))

    typedef __m256 VectorType;
    const VectorType mmONESIX =   _mm256_set_ps(ONESIX,ONESIX,ONESIX,ONESIX,ONESIX,ONESIX,ONESIX,ONESIX);
  #endif
#endif


#define VSTENSMP(L,R,T,B,F,K) VMUL(VADD(VADD(VADD((L),(R)),VADD((T),(B))),VADD((F),(K))),mmONESIX);
#define VSTENFMA(L,R,T,B,F,K) VFMA(VADD(VADD((L),(R)),VADD((T),(B))),VADD((F),(K)),mmONESIX);

#define VSTENCIL VSTENSMP

#endif
