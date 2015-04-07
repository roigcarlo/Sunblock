#include <sys/types.h>

#include "defines.h"

template <
    typename VariableType, 
    typename Variable3DType,
    typename IndexType
>
class Solver {
public:
  Solver(){};
  ~Solver(){};

  virtual void Execute(){};

};

template <
    typename VariableType, 
    typename Variable3DType,
    typename IndexType
>
class BfeccSolver : public Solver<VariableType,Variable3DType,IndexType> {
public:
  BfeccSolver(
      VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      Variable3DType * Field,
      const double &Dx, const double &Dt,
      const uint &BW,
      const uint &X, const uint &Y, const uint &Z,
      const uint &NB, const uint &NE) :
    Solver<VariableType,Variable3DType,IndexType>(),
    mpPhi(Phi),
    mpPhiAuxA(PhiAuxA), 
    mpPhiAuxB(PhiAuxB),
    mpField(Field),
    mDx(Dx),
    mIdx(1.0/Dx),
    mDt(Dt),
    mBW(BW),
    mBWP(BW/2),
    mX(X),
    mY(Y),
    mZ(Z),
    mNB(NB),
    mNE(NE) {};

  ~BfeccSolver() {};

  void GlobalToLocal(Variable3DType coord, double f) { 
    if(f==1.0) 
      return; 
    for(int d = 0; d < 3; d++) {
      coord[d] *= f; 
    }
  }

  double Interpolate(Variable3DType prevDelta, VariableType * Phi) {

    uint pi,pj,pk,ni,nj,nk;

    GlobalToLocal(prevDelta,mIdx);

    pi = floor(prevDelta[0]); ni = pi+1;
    pj = floor(prevDelta[1]); nj = pj+1;
    pk = floor(prevDelta[2]); nk = pk+1;

    double Nx, Ny, Nz;

    Nx = 1-(prevDelta[0] - floor(prevDelta[0]));
    Ny = 1-(prevDelta[1] - floor(prevDelta[1]));
    Nz = 1-(prevDelta[2] - floor(prevDelta[2]));

    return (
      Phi[IndexType::GetIndex(pi,pj,pk,mBW,mX,mY,mZ)] * (    Nx) * (    Ny) * (    Nz) +
      Phi[IndexType::GetIndex(ni,pj,pk,mBW,mX,mY,mZ)] * (1 - Nx) * (    Ny) * (    Nz) +
      Phi[IndexType::GetIndex(pi,nj,pk,mBW,mX,mY,mZ)] * (    Nx) * (1 - Ny) * (    Nz) +
      Phi[IndexType::GetIndex(ni,nj,pk,mBW,mX,mY,mZ)] * (1 - Nx) * (1 - Ny) * (    Nz) +
      Phi[IndexType::GetIndex(pi,pj,nk,mBW,mX,mY,mZ)] * (    Nx) * (    Ny) * (1 - Nz) +
      Phi[IndexType::GetIndex(ni,pj,nk,mBW,mX,mY,mZ)] * (1 - Nx) * (    Ny) * (1 - Nz) +
      Phi[IndexType::GetIndex(pi,nj,nk,mBW,mX,mY,mZ)] * (    Nx) * (1 - Ny) * (1 - Nz) +
      Phi[IndexType::GetIndex(ni,nj,nk,mBW,mX,mY,mZ)] * (1 - Nx) * (1 - Ny) * (1 - Nz)
    );
  }

  /**
   * Executes the solver 
   **/
  virtual void Execute() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    for(uint k = mBWP + tid; k < mZ + mBWP; k+= tsize) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhiAuxA,mpPhi,mpPhi,-1.0,0.0,1.0,i,j,k);
        }
      }
    }

    #pragma omp barrier

    for(uint k = mBWP + tid; k < mZ + mBWP; k+= tsize) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhiAuxB,mpPhi,mpPhiAuxA,1.0,1.5,-0.5,i,j,k);
        }
      } 
    }

    #pragma omp barrier

    for(uint k = mBWP + tid; k < mZ + mBWP; k+= tsize) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhi,mpPhi,mpPhiAuxB,-1.0,0.0,1.0,i,j,k);
        }
      }
    }

  }

  /**
   * Executes the solver using blocking
   **/
  virtual void ExecuteBlock() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    // Backward
    for(uint kk = tid; kk < mNB; kk+= tsize)
      for(uint jj = 0; jj < mNB; jj++)
        for(uint ii = 0; ii < mNB; ii++)
          for(uint k = mBWP + (kk * mNE); k < mBWP + ((kk+1) * mNE); k++)
            for(uint j = mBWP + (jj * mNE); j < mBWP + ((jj+1) * mNE); j++)
              for(uint i = mBWP + (ii * mNE); i < mBWP + ((ii+1) * mNE); i++)
                Apply(mpPhiAuxA,mpPhi,mpPhi,-1.0,0.0,1.0,i,j,k);

    #pragma omp barrier

    // Forward 
    for(uint kk = tid; kk < mNB; kk+= tsize)
      for(uint jj = 0; jj < mNB; jj++)
        for(uint ii = 0; ii < mNB; ii++)
          for(uint k = mBWP + (kk * mNE); k < mBWP + ((kk+1) * mNE); k++)
            for(uint j = mBWP + (jj * mNE); j < mBWP + ((jj+1) * mNE); j++)
              for(uint i = mBWP + (ii * mNE); i < mBWP + ((ii+1) * mNE); i++)
                Apply(mpPhiAuxB,mpPhi,mpPhiAuxA,1.0,1.5,-0.5,i,j,k);

    #pragma omp barrier
   
    // Backward
    for(uint kk = tid; kk < mNB; kk+= tsize)
      for(uint jj = 0; jj < mNB; jj++)
        for(uint ii = 0; ii < mNB; ii++)
          for(uint k = mBWP + (kk * mNE); k < mBWP + ((kk+1) * mNE); k++)
            for(uint j = mBWP + (jj * mNE); j < mBWP + ((jj+1) * mNE); j++)
              for(uint i = mBWP + (ii * mNE); i < mBWP + ((ii+1) * mNE); i++)
                Apply(mpPhi,mpPhi,mpPhiAuxB,-1.0,0.0,1.0,i,j,k);
  }

  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/ 
  void Apply(VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      const double &Sign, const double &WeightA, const double &WeightB,
      const uint &i, const uint &j, const uint &k) {

    uint cell = IndexType::GetIndex(i,j,k,mBW,mX,mY,mZ);
    
    Triple origin;
    Triple displacement;

    origin[0] = i * mDx;
    origin[1] = j * mDx;
    origin[2] = k * mDx;
   
    for(int d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * mpField[cell][d]*mDt;
    }

    VariableType iPhi = Interpolate(displacement,PhiAuxB);

    Phi[cell] = WeightA * PhiAuxA[cell] + WeightB * iPhi;
  }

private:

  /**
   * @phi:        Result of the operation
   * @phi_auxA:   first  operator
   * @phi_auxB:   second operator
   * @field:      field
   * @dx:         diferential of space
   * @dt:         diferential of time
   **/
   
  VariableType   * mpPhi;
  VariableType   * mpPhiAuxA;
  VariableType   * mpPhiAuxB;

  Variable3DType * mpField;

  const double mDx;
  const double mIdx;
  const double mDt;

  const uint mBW;
  const uint mBWP;

  const uint mX; 
  const uint mY; 
  const uint mZ;

  const uint mNB;
  const uint mNE;
};

// This does not belong here! put it in a class
template <typename T>
inline void stencilCross(T * gridA, T * gridB,
    const uint &cell, 
    const uint &X, const uint &Y, const uint &Z) {
  
  gridB[cell] = (
    gridA[cell - 1]   +                       // Left
    gridA[cell + 1]   +                       // Right
    gridA[cell - (X+BW)]   +                  // Up
    gridA[cell + (X+BW)]   +                  // Down
    gridA[cell - (Y+BW)*(X+BW)] +             // Front
    gridA[cell + (Y+BW)*(X+BW)] ) * ONESIX;   // Back
}