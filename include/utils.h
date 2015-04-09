#ifndef UTILS_H
#define UTILS_H

// Memory allocation & de-allocation
#include <malloc.h>

#include "defines.h"
#include "block.h"
#include "hacks.h"

template <typename T>
void AllocateGrid(T ** grid,
    const uint &X, const uint &Y, const uint &Z) {

  *grid = (T *)malloc(sizeof(T) * 
    (X+BW) * (Y+BW) * (Z+BW));
}

template <typename T>
void AllocateGrid(T * grid,
    const uint &X, const uint &Y, const uint &Z, const uint &align) {

  uint size = (X+BW) * (Y+BW) * (Z+BW);

  while((sizeof(T) * size) % align) size++;

#ifdef _WIN32
  *grid = (T *)_aligned_malloc(sizeof(T) * size,align);
#else
  *grid = (T *)memalign(align,sizeof(T) * size);
#endif
}

template <typename T>
void ReleaseGrid(T * grid) {

  free(*grid);
}

template <typename T>
void ReleaseGrid(T * grid, const uint &align) {
  
// Memory allocated with _aligned_malloc cannot be 
//  released using free. 
#ifdef _WIN32
  _aligned_free(*grid);
#else
  free(*grid);
#endif
}

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
    if(f==1.0) 
      return; 
    for(int d = 0; d < 3; d++) {
      coord[d] *= f; 
    }
  }

  static void LocalToGlobal(Variable3DType coord, double f) { 
    if(f==1.0) 
      return;
    for(int d = 0; d < 3; d++) { 
      coord[d] *= f;
    }
  }
};

#endif