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

  void copyLeft(
      PrecisionType * Phi
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < rDim; d++) {
          Phi[IndexType::GetIndex(0,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d] = Phi[IndexType::GetIndex(1,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d];
        }
      }
    }
  
  }

  void copyRight(
      PrecisionType * Phi
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < rDim; d++) {
          Phi[IndexType::GetIndex(rX + rBW - 1,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d] = Phi[IndexType::GetIndex(rX + rBW - 2,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d];
        }
      }
    }

  }

  void copyLeftToRight(
      PrecisionType * Phi
    ) {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint d = 0; d < rDim; d++) {
          Phi[IndexType::GetIndex(0,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d] = Phi[IndexType::GetIndex(rX + rBW - 2,j,k,pBlock->mPaddY,pBlock->mPaddZ)*rDim+d];
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
