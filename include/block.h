#ifndef BLOCK_H
#define BLOCK_H

#include "defines.h"
#include "utils.h"

class Block {
public:

  typedef VariableType  ResultType;
  typedef Indexer       IndexType;

  Block(ResultType * PhiA, ResultType * PhiB, ResultType * PhiC,
      Variable3DType * Field,
      const double &Dx, const double &Omega,
      const uint &BW,
      const uint &X, const uint &Y, const uint &Z,
      const uint &NB, const uint &NE) :
    pPhiA(PhiA),
    pPhiB(PhiB), 
    pPhiC(PhiC),
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
    rNE(NE) {

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

  void InitializeVariable() {
    for(uint k = rBWP; k < rZ - rBWP; k++) {
      for(uint j = rBWP; j < rY - rBWP; j++) {
        for(uint i = rBWP; i < rX - rBWP; i++ ) {
          pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0;
          pPhiB[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0;
          pPhiC[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 0.0;
        }
      }
    }
  }

  double InitializeVelocity(double &maxv) {

    maxv = -1;

    for(uint k = 0; k < rZ + rBW; k++) {
      for(uint j = 0; j < rY + rBW; j++) {
        for(uint i = 0; i < rX + rBW; i++) {
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][0] = 0.0;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][1] = 0.0;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][2] = 0.0;
        } 
      }
    }

    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        for(uint i = rBWP; i < rX + rBWP; i++ ) {
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][0] = -rOmega * (double)(j-(rY+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][1] =  rOmega * (double)(i-(rX+1.0)/2.0) * rDx;
          pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][2] =  0.0;

          maxv = std::max((double)abs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][0]),maxv);
          maxv = std::max((double)abs(pVelocity[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)][1]),maxv);
        }
      }
    }

    return maxv;
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
          
          if(d2 < rr)
            pPhiA[IndexType::GetIndex(i,j,k,mPaddY,mPaddZ)] = 1.0 - d2/rr;
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

  ResultType * pPhiA;
  ResultType * pPhiB;
  ResultType * pPhiC;

  Variable3DType * pVelocity;

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