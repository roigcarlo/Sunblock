#ifndef BLOCK_H
#define BLOCK_H

#include "defines.h"
#include "utils.h"

class Block {
public:

  typedef Indexer IndexType;

  Block(
      PrecisionType ** buffers,
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
    pBuffers(buffers),
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
    for(size_t k = 0; k < rZ - 0; k++) {
      for(size_t j = 0; j < rY - 0; j++) {
        for(size_t i = 0; i < rX - 0; i++ ) {
          for(size_t d = 0; d < rDim; d++) {
            pBuffers[AUX_3D_0][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
            pBuffers[AUX_3D_1][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
            pBuffers[AUX_3D_2][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
          }
        }
      }
    }
  }

  void InitializePressure() {

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++) {
          pBuffers[PRESSURE][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0f;//(-9.8f/(PrecisionType)rZ) * (PrecisionType)k;
        }
      }
    }

  }

  void InitializeVelocity() {

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++) {
          for(size_t d = 0; d < rDim; d++) {
            pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 0.0f;
          }
        }
      }
    }

    int toUpdate[4] = {VELOCITY,AUX_3D_0,AUX_3D_1,AUX_3D_2};

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++ ) {
          for(size_t b = 0; b < 4; b++ ) {
            pBuffers[toUpdate[b]][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0] = 0.0f; //-rOmega * (PrecisionType)(j-(rY+1.0)/2.0) * rDx;
            pBuffers[toUpdate[b]][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1] = 0.0f; // rOmega * (PrecisionType)(i-(rX+1.0)/2.0) * rDx;
            pBuffers[toUpdate[b]][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;
          }
        }
      }
    }

    // for(size_t k = 1; k < rZ + rBW - 1; k++)
    //   for(size_t j = 2; j < rY + rBW - 2; j++)
    //     for(size_t i = 2; i < rX + rBW - 2; i++ )
    //       pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2] = 0.0f;

    for(size_t a = 1; a < rY + rBW - 1; a++)
      for(size_t b = 2; b < rX + rBW - 1; b++)
        pBuffers[VELOCITY][IndexType::GetIndex(a,b,rZ,mPaddY,mPaddZ)*rDim+0] = 10.0f;

    for(size_t jk = 0; jk < rY + rBW; jk++) {
      for(size_t i = 0; i < rX + rBW; i++ ) {
        for(size_t b = 0; b < 4; b++ ) {
          pBuffers[toUpdate[b]][IndexType::GetIndex(i,0,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
          pBuffers[toUpdate[b]][IndexType::GetIndex(i,jk,0,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        }
      }
    }

    for(size_t jk = 0; jk < rY + rBW; jk++) {
      for(size_t i = 0; i < rX + rBW; i++ ) {
        for(size_t b = 0; b < 4; b++ ) {
          pBuffers[toUpdate[b]][IndexType::GetIndex(i,rY+rBW-1,jk,mPaddY,mPaddZ)*rDim+0] = 0.0f;
          pBuffers[toUpdate[b]][IndexType::GetIndex(i,jk,rY+rBW-1,mPaddY,mPaddZ)*rDim+0] = 0.0f;
        }
      }
    }

  }

  void calculateMaxVelocity(PrecisionType &maxv) {

    maxv = 1.0f;

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++ ) {
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0]),maxv);
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1]),maxv);
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2]),maxv);
        }
      }
    }

  }

  void calculateRealMaxVelocity(PrecisionType &maxv) {

    maxv = -1.0f;

    for(size_t k = 0; k < rZ + rBW; k++) {
      for(size_t j = 0; j < rY + rBW; j++) {
        for(size_t i = 0; i < rX + rBW; i++ ) {
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+0]),maxv);
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+1]),maxv);
          maxv = std::max((PrecisionType)fabs(pBuffers[VELOCITY][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+2]),maxv);
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
              pBuffers[AUX_3D_0][IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)*rDim+d] = 1.0f - d2/rr;
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

  PrecisionType ** pBuffers;

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
