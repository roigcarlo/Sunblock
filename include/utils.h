#ifndef UTILS_H
#define UTILS_H

// Memory allocation & de-allocation
#include <malloc.h>

#include "defines.h"
#include "hacks.h"

class MemManager {
public:

  MemManager(bool pinned_option = false){
    use_cuda_pinned_mem = pinned_option;
  }

  ~MemManager(){}

  template <typename T>
  void AllocateGrid(
      T ** grid,
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const size_t &dim,
      const size_t align) {

    size_t elements     = (X+BW) * (Y+BW) * (Z+BW);
    size_t element_size = sizeof(T) * dim;

    size_t size         = elements * element_size;

    if(use_cuda_pinned_mem) {
#ifdef USE_CUDA
      cudaMallocHost((void**)grid, size);
#else
      printf("Error: CUDA support is not enabled, please compile with -DUSE_CUDA.\n");
      exit(1);
#endif
    } else {
      if(align < 1) {
        printf("Error: Trying to align memory to negative values.\n");
        exit(1);
      } else if( align == 1) {
        *grid = (T *)malloc(size);
      } else {
        while((sizeof(T) * size) % align) size++;
#ifdef _WIN32
          *grid = (T *)_aligned_malloc(size,align);
#else
          *grid = (T *)memalign(align,size);
#endif
      }
    }
  }

  template <typename T>
  void ReleaseGrid(T * grid, const size_t align) {

    if(use_cuda_pinned_mem) {
#ifdef USE_CUDA
      cudaFreeHost(*grid);
#else
      printf("Error: CUDA support is not enabled, please compile with -DUSE_CUDA.");
      exit(1);
#endif
    } else {
      if(align < 1) {
        printf("Error: Trying to align memory to negative values.");
        exit(1);
      } else if( align == 1) {
        free(*grid);
      } else {
#ifdef _WIN32
        _aligned_free(*grid);
#else
        free(*grid);
#endif
      }
    }
  }

  union fui{
    int32_t i;
    float f;
  };

  union dui{
    int32_t i[2];
    double d;
  };

  inline void getExponent(double &value, int exponent) {

    dui a;
    a.d = value;

    exponent = ((a.i[1] & 0x7FF) >> 20 ) - 1023;
  }

  inline void getExponent(float &value, int exponent) {

    fui a;
    a.f = value;

    exponent = ((a.i & 0x7F8) >> 23 ) - 127;
  }


  // Doubles have:
  // 01 Bits -> Sign
  // 11 Bits -> Exp
  // 52 Bits -> Fraction
  inline void FlipDouble(double &value, size_t index) {

    if(index > 63) return;

    int lohi = index > 31;
    index = index % 31;

    dui a;
    a.d = value;

    a.i[lohi] = a.i[lohi] ^ (0x1<<index);
  }

  inline void FlipDoubleSign(double &value) {
    dui a;
    a.d = value;

    a.i[1] = a.i[1] ^ (0x1<<31);
  }

  inline void FlipDoubleExponent(double &value, size_t index) {

    if(index > 11) return;

    dui a;
    a.d = value;

    a.i[1] = a.i[1] ^ (0x1<<(index+20));
  }

  inline void FlipDoubleFraction(double &value, size_t index) {

    if(index > 51) return;

    int lohi = index > 31;
    index = index % 31;

    dui a;
    a.d = value;

    a.i[lohi] = a.i[lohi] ^ (0x1<<index);
  }

  // Floats have:
  // 01 Bits -> Sign
  // 08 Bits -> Exp
  // 23 Bits -> Fraction
  inline void FlipFloat(float &value, size_t index) {

    if(index > 31) return;

    fui a;
    a.f = value;

    a.i = a.i ^ (0x1<<index);
  }

  inline void FlipFloatSign(float &value) {

    fui a;
    a.f = value;

    a.i = a.i ^ (0x1<<31);
  }

  inline void FlipFloatExponent(float &value, size_t index) {

    if(index > 7) return;

    fui a;
    a.f = value;

    a.i = a.i ^ (0x1<<(index + 23));
  }

  inline void FlipFloatFraction(float &value, size_t index) {

    if(index > 22) return;

    fui a;
    a.f = value;

    a.i = a.i ^ (0x1<<index);
  }

private:
  bool use_cuda_pinned_mem;
};

// Index calculation
class Indexer {
public:
  /**
   * Calculates the standard index
   * @BW: BorderWidth
   **/
  static size_t GetIndex(
      const size_t &i,
      const size_t &j,
      const size_t &k,
      const size_t &sizeY,
      const size_t &sizeZ) {

    return k*sizeZ+j*sizeY+i;
  }

  static void PreCalculateIndexTable(
      const size_t &N
    ) {
    // Nedded by some indexers to calculate faster
  }

  static void ReleaseIndexTable(const size_t &N) {
    // Nedded by some indexers to calculate faster
  }
};

class MortonIndexer : Indexer {
public:

  static size_t * pIndexTable;

  /**
   * Calculates the morton index
   * @BW: BorderWidth
   **/
  template<typename BlockType>
  static uint GetIndex(
      const size_t &i,
      const size_t &j,
      const size_t &k,
      const size_t &sizeY,
      const size_t &sizeZ) {

    //return interleave64(i,j,k);
    return pIndexTable[k*sizeZ+j*sizeY+i];
  }

  static void PreCalculateIndexTable(const size_t &N) {

    if(pIndexTable == NULL) {
      pIndexTable = (size_t *)malloc(sizeof(size_t) * N * N * N);

      for(size_t k = 0; k < N; k++)
        for(size_t j = 0; j < N; j++)
          for(size_t i = 0; i < N; i++)
            pIndexTable[k*N*N+j*N+i] = interleave64(i,j,k);
    }
  }

  static void ReleaseIndexTable(const size_t &N) {

    if(pIndexTable == NULL) {
      free(pIndexTable);
    }
  }
};

size_t * MortonIndexer::pIndexTable;

class Utils {
private:
  Utils() {}
  ~Utils() {}

public:

  static void GlobalToLocal(
      PrecisionType * coord,
      PrecisionType f,
      const size_t &dim) {

    for(size_t d = 0; d < dim; d++)
      coord[d] *= f;
  }

  static void LocalToGlobal(
      PrecisionType * coord,
      PrecisionType f,
      const size_t &dim
    ) {

    for(size_t d = 0; d < dim; d++)
      coord[d] /= f;
  }
};

#endif
