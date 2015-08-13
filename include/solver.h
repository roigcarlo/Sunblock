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
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint a = 0; a < rY + rBW; a++) {
      for(uint b = 0; b < rX + rBW; b++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(0,a,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(1,a,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
          Phi[IndexType::GetIndex(a,0,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(a,1,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
          Phi[IndexType::GetIndex(a,b,0,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(a,b,1,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];

          Phi[IndexType::GetIndex(rX + rBW - 1,a,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(rX + rBW - 2,a,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
          Phi[IndexType::GetIndex(a,rY + rBW - 1,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(a,rY + rBW - 2,b,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
          Phi[IndexType::GetIndex(a,b,rZ + rBW - 1,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(a,b,rZ + rBW - 2,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }
  
  }

  void copyLeft(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(0,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(1,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }
  
  }

  void copyRight(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(rX + rBW - 1,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(rX + rBW - 2,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }

  }

    void copyDown(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint k = 0; k < rZ + rBW; k++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(i,0,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(i,1,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }

  }

  void copyUp(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint k = 0; k < rZ + rBW; k++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(i,rY + rBW - 1,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(i,rY + rBW - 2,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }
  
  }

  void copyBack(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(i,j,0,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(i,j,1,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }
  }

  void copyFront(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint i = 0; i < rX + rBW; i++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(i,j,rZ + rBW - 1,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(i,j,rZ + rBW - 2,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }
  
  }

  //////////////////////////////////////////

  void copyLeftToRight(
      PrecisionType * Phi,
      uint dim
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < dim; d++) {
          Phi[IndexType::GetIndex(0,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d] = Phi[IndexType::GetIndex(rX + rBW - 2,j,k,pBlock->mPaddY,pBlock->mPaddZ)*dim+d];
        }
      }
    }

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
