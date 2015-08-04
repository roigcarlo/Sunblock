#ifndef BLOCK_H
#define BLOCK_H

#include "defines.h"
#include "utils.h"

class Block {
public:

  typedef Indexer IndexType;

  Block(
      PrecisionType * PhiA,
      PrecisionType * PhiB,
      PrecisionType * PhiC,
      PrecisionType * PressA,
      PrecisionType * PressB,
      PrecisionType * Field,
      const double &Dx, const double &Omega,
      const uint &BW,
      const uint &X, const uint &Y, const uint &Z,
      const uint &NB, const uint &NE, const uint &DIM) :
    pPhiA(PhiA),
    pPhiB(PhiB),
    pPhiC(PhiC),
    pPressA(PressA),
    pPressB(PressB),
    pVelocity(Field),
    rDx(Dx),
    rIdx(1.0/Dx),
    rOmega(Omega),
    rBW(BW),
    rBWP(BW/2),
    rX(X),
    rY(Y),
    rZ(Z),
    rNB(NB),
    rNE(NE),
    rDim(DIM) {

    mPaddZ = (rZ+rBW)*(rY+rBW);
    mPaddY = (rY+rBW);

    //      F ------ G
    //     /|       /|
    //    / |      / |
    //   B -+---- C  |
    //   |  D ----+- E
    //   | /      | /
    //   |/       |/
    //   0 ------ A

    mPaddA = 1;
    mPaddB = (rY+rBW);
    mPaddC = (rY+rBW)+1;
    mPaddD = (rZ+rBW)*(rY+rBW);
    mPaddE = (rZ+rBW)*(rY+rBW) + 1;
    mPaddF = (rZ+rBW)*(rY+rBW) + (rY+rBW);
    mPaddG = (rZ+rBW)*(rY+rBW) + (rY+rBW) + 1;
  }

  ~Block() {}

  void Zero() {
    for(uint k = rBWP; k < rZ - rBWP; k++) {
      for(uint j = rBWP; j < rY - rBWP; j++) {
        for(uint i = rBWP; i < rX - rBWP; i++ ) {
          for(uint d = 0; d < rDim; d++) {
            pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0;
            pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0;
            pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0;
          }
        }
      }
    }
  }

  void InitializePressure() {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++) {
          for(uint d = 0; d < 1; d++) {
            pPressA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*1+d] = (i+1.0)/(rX-1.0) - 1.0/rX;
            pPressB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*1+d] = (i+1.0)/(rX-1.0) - 1.0/rX;
          }
        }
      }
    }

  }

  void InitializeVelocity() {

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++) {
          for(uint d = 0; d < rDim; d++) {
            pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
          }
        }
      }
    }

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++ ) {
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 1.0f; //-rOmega * (double)(j-(rY+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (double)(i-(rX+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 1.0f; //-rOmega * (double)(j-(rY+1.0)/2.0) * rDx;
          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (double)(i-(rX+1.0)/2.0) * rDx;
          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 1.0f; //-rOmega * (double)(j-(rY+1.0)/2.0) * rDx;
          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (double)(i-(rX+1.0)/2.0) * rDx;
          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 1.0f; //-rOmega * (double)(j-(rY+1.0)/2.0) * rDx;
          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (double)(i-(rX+1.0)/2.0) * rDx;
          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;
        }
      }
    }
  }

  void calculateMaxVelocity(double &maxv) {

    maxv = 1.0f;

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++ ) {
          maxv = std::max((double)fabs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0]),maxv);
          maxv = std::max((double)fabs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1]),maxv);
        }
      }
    }

  }

  void WriteHeatFocus() {
      uint Xc, Yc, Zc;

    Xc = (uint)(2.0 / 5.0 * (rX));
  	Yc = (uint)(2.0 / 5.5 * (rY));
  	Zc = (uint)(1.0 / 2.0 * (rZ));

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++) {

          double d2 = pow((Xc - (double)(i)),2) + pow((Yc - (double)(j)),2) + pow((Zc - (double)(k)),2);
          double rr = pow(rX/6.0,2);

          if(d2 < rr) {
            for(uint d = 0; d < rDim; d++) {
              pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 1.0 - d2/rr;
            }
          }
        }
      }
    }
  }

  /**
   * @phi:        Result of the operation
   * @phi_auxA:   first  operator
   * @phi_auxB:   second operator
   * @field:      field
   * @dx:         diferential of space
   * @dt:         diferential of time
   **/

  PrecisionType * pPhiA;
  PrecisionType * pPhiB;
  PrecisionType * pPhiC;

  PrecisionType * pPressA;
  PrecisionType * pPressB;

  PrecisionType * pVelocity;

  const double & rDx;
  const double rIdx;
  const double & rOmega;

  const uint & rBW;
  const uint rBWP;

  const uint & rX;
  const uint & rY;
  const uint & rZ;

  const uint & rNB;
  const uint & rNE;

  const uint & rDim;

  uint mPaddZ;
  uint mPaddY;

  uint mPaddA;
  uint mPaddB;
  uint mPaddC;
  uint mPaddD;
  uint mPaddE;
  uint mPaddF;
  uint mPaddG;

private:

};

#endif
