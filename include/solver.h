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

  Solver(Block * block, const double& Dt) :
      pBlock(block),
      pPhiA(block->pPhiA),
      pPhiB(block->pPhiB),
      pPhiC(block->pPhiC),
      pVelocity(block->pVelocity),
      rDx(block->rDx),
      rIdx(1.0f/block->rDx),
      rDt(Dt),
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

  PrecisionType * pVelocity;

  const double & rDx;
  const double rIdx;
  const double & rDt;

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
