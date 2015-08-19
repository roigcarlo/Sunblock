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
      PrecisionType * PhiD,
      PrecisionType * PressA,
      PrecisionType * PressB,
      PrecisionType * Field,
      uint * Flags,
      const PrecisionType &Dx,
      const PrecisionType &Omega,
      const PrecisionType &Ro,
      const PrecisionType &Mu,
      const PrecisionType &Ka,
      const PrecisionType &CC2,
      const size_t &BW,
      const size_t &X, const size_t &Y, const size_t &Z,
      const size_t &NB, const size_t &NE, const size_t &DIM) :
    pPhiA(PhiA),
    pPhiB(PhiB),
    pPhiC(PhiC),
    pPhiD(PhiD),
    pPressA(PressA),
    pPressB(PressB),
    pVelocity(Field),
    pFlags(Flags),
    rDx(Dx),
    rIdx(1.0/Dx),
    rOmega(Omega),
    rRo(Ro),
    rMu(Mu),
    rKa(Ka),
    rCC2(CC2),
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
    for(size_t k = rBWP; k < rZ - rBWP; k++) {
      for(size_t j = rBWP; j < rY - rBWP; j++) {
        for(size_t i = rBWP; i < rX - rBWP; i++ ) {
          for(size_t d = 0; d < rDim; d++) {
            pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0f;
            pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0f;
            pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 7.0f;
          }
        }
      }
    }
  }

  void InitializePressure() {

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++) {
          pPressA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0f;//(-9.8f/(PrecisionType)rZ) * (PrecisionType)k;
          pPressB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0f;//(-9.8f/(PrecisionType)rZ) * (PrecisionType)k;
        }
      }
    }

  }

  void InitializeVelocity() {

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++) {
          for(size_t d = 0; d < rDim; d++) {
            pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
          }
        }
      }
    }

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++ ) {
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 0.0f; //-rOmega * (PrecisionType)(j-(rY+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (PrecisionType)(i-(rX+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 0.0f; //-rOmega * (PrecisionType)(j-(rY+1.0)/2.0) * rDx;
          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (PrecisionType)(i-(rX+1.0)/2.0) * rDx;
          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 0.0f; //-rOmega * (PrecisionType)(j-(rY+1.0)/2.0) * rDx;
          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (PrecisionType)(i-(rX+1.0)/2.0) * rDx;
          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 0.0f; //-rOmega * (PrecisionType)(j-(rY+1.0)/2.0) * rDx;
          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (PrecisionType)(i-(rX+1.0)/2.0) * rDx;
          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;
        }
      }
    }

    for(size_t jk = 0; jk < rY + rBW; jk++) {
      for(size_t i = 0; i < rX + rBW; i++ ) {
        pVelocity[IndexType::GetIndex(i,0,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pVelocity[IndexType::GetIndex(i,jk,0,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiA[IndexType::GetIndex(i,0,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiA[IndexType::GetIndex(i,jk,0,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiB[IndexType::GetIndex(i,0,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiB[IndexType::GetIndex(i,jk,0,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiC[IndexType::GetIndex(i,0,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiC[IndexType::GetIndex(i,jk,0,mPaddY,mPaddZ)*rDim+0] = 0.0f;
      }
    }

    for(size_t jk = 0; jk < rY + rBW; jk++) {
      for(size_t i = 0; i < rX + rBW; i++ ) {
        pVelocity[IndexType::GetIndex(i,rY+rBW-1,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pVelocity[IndexType::GetIndex(i,jk,rY+rBW-1,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiA[IndexType::GetIndex(i,rY+rBW-1,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiA[IndexType::GetIndex(i,jk,rY+rBW-1,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiB[IndexType::GetIndex(i,rY+rBW-1,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiB[IndexType::GetIndex(i,jk,rY+rBW-1,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiC[IndexType::GetIndex(i,rY+rBW-1,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        pPhiC[IndexType::GetIndex(i,jk,rY+rBW-1,mPaddY,mPaddZ)*rDim+0] = 0.0f;
      }
    }
  }

  void calculateMaxVelocity(PrecisionType &maxv) {

    maxv = 1.0f;

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++ ) {
          maxv = std::max((PrecisionType)fabs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0]),maxv);
          maxv = std::max((PrecisionType)fabs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1]),maxv);
          maxv = std::max((PrecisionType)fabs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2]),maxv);
        }
      }
    }

  }

  void WriteHeatFocus() {

    size_t Xc, Yc, Zc;

    Xc = (size_t)(2.0f / 5.0f * (PrecisionType)(rX));
  	Yc = (size_t)(2.0f / 5.5f * (PrecisionType)(rY));
  	Zc = (size_t)(1.0f / 2.0f * (PrecisionType)(rZ));

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++) {

          PrecisionType d2 =
            pow(((PrecisionType)Xc - (PrecisionType)(i)),2.0f) +
            pow(((PrecisionType)Yc - (PrecisionType)(j)),2.0f) +
            pow(((PrecisionType)Zc - (PrecisionType)(k)),2.0f);

          PrecisionType rr =
            pow((PrecisionType)rX/6.0,2.0f);

          if(d2 < rr) {
            for(size_t d = 0; d < rDim; d++) {
              pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 1.0f - d2/rr;
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
  PrecisionType * pPhiD;

  PrecisionType * pPressA;
  PrecisionType * pPressB;

  PrecisionType * pVelocity;

  uint * pFlags;

  const PrecisionType & rDx;
  const PrecisionType rIdx;
  const PrecisionType & rOmega;
  const PrecisionType & rRo;
  const PrecisionType & rMu;
  const PrecisionType & rKa;
  const PrecisionType & rCC2;

  const size_t &rBW;
  const size_t  rBWP;

  const size_t &rX;
  const size_t &rY;
  const size_t &rZ;

  const size_t &rNB;
  const size_t &rNE;

  const size_t &rDim;

  size_t mPaddZ;
  size_t mPaddY;

  size_t mPaddA;
  size_t mPaddB;
  size_t mPaddC;
  size_t mPaddD;
  size_t mPaddE;
  size_t mPaddF;
  size_t mPaddG;
};

#endif
