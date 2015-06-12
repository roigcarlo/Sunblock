#ifndef UTILS_H
#define UTILS_H

// Memory allocation & de-allocation
#include <malloc.h>

#include "defines.h"
#include "block.h"
#include "hacks.h"

class MemManager {
public:

  MemManager(bool pinned_option = false){
    use_cuda_pinned_mem = pinned_option;
  }

  ~MemManager(){}

  template <typename T>
  void AllocateGrid(T ** grid, const uint &X, const uint &Y, const uint &Z, const uint align) {

    uint elements     = (X+BW) * (Y+BW) * (Z+BW);
    uint element_size = sizeof(T);

    uint size         = elements * element_size;

    if(use_cuda_pinned_mem) {
#ifdef USE_CUDA
      cudaMallocHost((void**)grid, size);
#else
      printf("Error: CUDA support is not enabled, please compile with -DUSE_CUDA.");
      exit(1);
#endif
    } else {
      if(align < 1) {
        printf("Error: Trying to align memory to negative values.");
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
  void ReleaseGrid(T * grid, const uint align) {

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
  template<typename BlockType>
  static uint GetIndex(BlockType * b, const uint &i, const uint &j, const uint &k) {
    return k*b->mPaddZ+j*b->mPaddY+i;
  }

  static void PreCalculateIndexTable(const uint &N) {
    // Nedded by some indexers to calculate faster
  }

  static void ReleaseIndexTable(const uint &N) {
    // Nedded by some indexers to calculate faster
  }
};

class MortonIndexer : Indexer {
public:

  static uint * pIndexTable;

  /** 
   * Calculates the morton index
   * @BW: BorderWidth
   **/
  template<typename BlockType>
  static uint GetIndex(BlockType * b, const uint &i, const uint &j, const uint &k) {
    //return interleave64(i,j,k);
    return pIndexTable[k*b->mPaddZ+j*b->mPaddY+i];
  }

  static void PreCalculateIndexTable(const uint &N) {
    if(pIndexTable == NULL) {
      pIndexTable = (uint *)malloc(sizeof(uint) * N * N * N);

      for(uint k = 0; k < N; k++)
        for(uint j = 0; j < N; j++)
          for(uint i = 0; i < N; i++)
            pIndexTable[k*N*N+j*N+i] = interleave64(i,j,k);
    }
  }

  static void ReleaseIndexTable(const uint &N) {
    if(pIndexTable == NULL) {
      free(pIndexTable);
    }
  }
};

uint * MortonIndexer::pIndexTable;

class PeanoIndexer : Indexer {
public:
  /** 
   * Calculates the Peano index
   * @BW: BorderWidth
   **/
  template<typename BlockType>
  static uint GetIndex(BlockType * b, const uint &i, const uint &j, const uint &k) {
    // To be implemented
    return 0;
  }
};

class Utils {
private:
  Utils() {}
  ~Utils() {}

public:

  static void GlobalToLocal(Variable3DType coord, double f) { 
    for(int d = 0; d < 3; d++)
      coord[d] *= f; 
  }

  static void LocalToGlobal(Variable3DType coord, double f) { 
    for(int d = 0; d < 3; d++)
      coord[d] /= f;
  }
};

#endif