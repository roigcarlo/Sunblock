#ifndef SOLVER_H
#define SOLVER_H

#include <sys/types.h>

#include "defines.h"
#include "utils.h"
#include "block.h"
#include "kernels.h"
#include "interpolator.h"

class Solver {
public:

  typedef Block::IndexType        IndexType;
  typedef TrilinealInterpolator   InterpolateType;

  Solver(Block * block, const PrecisionType& Dt, const PrecisionType& Pdt) :
      pBlock(block),
      pPhiA(block->pPhiA),
      pPhiB(block->pPhiB),
      pPhiC(block->pPhiC),
      pPhiD(block->pPhiD),
      pPressA(block->pPressA),
      pPressB(block->pPressB),
      pVelocity(block->pVelocity),
      rDx(block->rDx),
      rIdx(1.0f/block->rDx),
      rDt(Dt),
      rPdt(Pdt),
      rRo(block->rRo),
      rMu(block->rMu),
      rKa(block->rKa),
      rCC2(block->rCC2),
      rBW(block->rBW),
      rBWP(block->rBW/2),
      rX(block->rX),
      rY(block->rY),
      rZ(block->rZ),
      rNB(block->rNB),
      rNE(block->rNE),
      rDim(block->rDim){
  }

  ~Solver() {
  }

  void copyAll(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint a = 0; a < rY + rBW; a++) {
      for(uint b = 0; b < rX + rBW; b++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(0,a,b)*dim+d] = buff[INDEX(1,a,b)*dim+d];
          buff[INDEX(a,0,b)*dim+d] = buff[INDEX(a,1,b)*dim+d];
          buff[INDEX(a,b,0)*dim+d] = buff[INDEX(a,b,1)*dim+d];

          buff[INDEX(rX + rBW - 1,a,b)*dim+d] = buff[INDEX(rX + rBW - 2,a,b)*dim+d];
          buff[INDEX(a,rY + rBW - 1,b)*dim+d] = buff[INDEX(a,rY + rBW - 2,b)*dim+d];
          buff[INDEX(a,b,rZ + rBW - 1)*dim+d] = buff[INDEX(a,b,rZ + rBW - 2)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void copyLeft(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(0,j,k)*dim+d] = buff[INDEX(1,j,k)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void copyRight(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(rX + rBW - 1,j,k)*dim+d] = buff[INDEX(rX + rBW - 2,j,k)*dim+d];
        }
      }
    }

    #undef INDEX

  }

    void copyDown(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint k = 0; k < rZ + rBW; k++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(i,0,k)*dim+d] = buff[INDEX(i,1,k)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void copyUp(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint k = 0; k < rZ + rBW; k++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(i,rY + rBW - 1,k)*dim+d] = buff[INDEX(i,rY + rBW - 2,k)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void copyBack(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(i,j,0)*dim+d] = buff[INDEX(i,j,1)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void copyFront(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(i,j,rZ + rBW - 1)*dim+d] = buff[INDEX(i,j,rZ + rBW - 2)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  //////////////////////////////////////////

  void copyLeftToRight(
      PrecisionType * buff,
      uint dim
    ) {

    #define INDEX(I,J,K) IndexType::GetIndex((I),(J),(K),pBlock->mPaddY,pBlock->mPaddZ)

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          buff[INDEX(0,j,k)*dim+d] = buff[INDEX(rX + rBW - 2,j,k)*dim+d];
        }
      }
    }

    #undef INDEX

  }

  void Prepare() {
  }

  void Finish() {
  }

  void Execute() {
  }

protected:

  Block * pBlock;

  PrecisionType * pPhiA;
  PrecisionType * pPhiB;
  PrecisionType * pPhiC;
  PrecisionType * pPhiD;

  PrecisionType * pPressA;
  PrecisionType * pPressB;

  PrecisionType * pVelocity;

  const PrecisionType & rDx;
  const PrecisionType rIdx;
  const PrecisionType & rDt;
  const PrecisionType & rPdt;

  const PrecisionType & rRo;
  const PrecisionType & rMu;
  const PrecisionType & rKa;

  const PrecisionType & rCC2;

  const uint & rBW;
  const uint rBWP;

  const uint & rX;
  const uint & rY;
  const uint & rZ;

  const uint & rNB;
  const uint & rNE;

  const uint & rDim;
};

#endif
