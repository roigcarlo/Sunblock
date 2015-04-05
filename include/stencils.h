#include <sys/types.h>

#include "defines.h"

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

template <
    typename VariableType, 
    typename Variable3DType,
    typename Index
    //typename Interpolate
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
    typename Index
    //typename Interpolate
>
class BfeccSolver : public Solver<VariableType,Variable3DType,Index> {
public:
  BfeccSolver(
      VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      Variable3DType * Field, 
      const double &Dx, const double &Dt,
      const uint &BW,
      const uint &X, const uint &Y, const uint &Z) :
    mpPhi(Phi),
    mpPhiAuxA(PhiAuxA), 
    mpPhiAuxB(PhiAuxB),
    mpField(Field), 
    mDx(Dx),
    mDt(Dt),
    mBW(BW),
    mBWP(BW/2),
    mX(X),
    mY(Y),
    mZ(Z) {};

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

    GlobalToLocal(prevDelta,1.0/mDx);

    pi = floor(prevDelta[0]); ni = pi+1;
    pj = floor(prevDelta[1]); nj = pj+1;
    pk = floor(prevDelta[2]); nk = pk+1;

    double Nx, Ny, Nz;

    Nx = 1-(prevDelta[0] - floor(prevDelta[0]));
    Ny = 1-(prevDelta[1] - floor(prevDelta[1]));
    Nz = 1-(prevDelta[2] - floor(prevDelta[2]));

    return (
      Phi[pk*(mZ+mBW)*(mY+mBW)+pj*(mY+mBW)+pi] * (    Nx) * (    Ny) * (    Nz) +
      Phi[pk*(mZ+mBW)*(mY+mBW)+pj*(mY+mBW)+ni] * (1 - Nx) * (    Ny) * (    Nz) +
      Phi[pk*(mZ+mBW)*(mY+mBW)+nj*(mY+mBW)+pi] * (    Nx) * (1 - Ny) * (    Nz) +
      Phi[pk*(mZ+mBW)*(mY+mBW)+nj*(mY+mBW)+ni] * (1 - Nx) * (1 - Ny) * (    Nz) +
      Phi[nk*(mZ+mBW)*(mY+mBW)+pj*(mY+mBW)+pi] * (    Nx) * (    Ny) * (1 - Nz) +
      Phi[nk*(mZ+mBW)*(mY+mBW)+pj*(mY+mBW)+ni] * (1 - Nx) * (    Ny) * (1 - Nz) +
      Phi[nk*(mZ+mBW)*(mY+mBW)+nj*(mY+mBW)+pi] * (    Nx) * (1 - Ny) * (1 - Nz) +
      Phi[nk*(mZ+mBW)*(mY+mBW)+nj*(mY+mBW)+ni] * (1 - Nx) * (1 - Ny) * (1 - Nz)
    );
  }

  virtual void Execute() {

    for(uint k = mBWP + omp_get_thread_num(); k < mZ + mBWP; k+=omp_get_num_threads()) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhiAuxA,mpPhi,mpPhi,-1.0,0.0,1.0,i,j,k);
        }
      }
    }

    #pragma omp barrier

    for(uint k = mBWP + omp_get_thread_num(); k < mZ + mBWP; k+=omp_get_num_threads()) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhiAuxB,mpPhi,mpPhiAuxA,1.0,1.5,-0.5,i,j,k);
        }
      } 
    }

    #pragma omp barrier

    for(uint k = mBWP + omp_get_thread_num(); k < mZ + mBWP; k+=omp_get_num_threads()) {
      for(uint j = mBWP; j < mY + mBWP; j++) {
        for(uint i = mBWP; i < mX + mBWP; i++) {
          Apply(mpPhi,mpPhi,mpPhiAuxB,-1.0,0.0,1.0,i,j,k);
        }
      }
    }

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

    uint cell = Index::GetIndex(i,j,k,mBW,mX,mY,mZ);
    
    Triple origin;
    Triple displacement;

    origin[0] = i * mDx;
    origin[1] = j * mDx;
    origin[2] = k * mDx;
   
    for(int d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * mpField[cell][d]*mDt;
    }

    Phi[cell] = WeightA * PhiAuxA[cell] + 
                WeightB * Interpolate(displacement,PhiAuxB);
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
   
  VariableType * mpPhi;
  VariableType * mpPhiAuxA;
  VariableType * mpPhiAuxB;

  Variable3DType * mpField;

  const double mDx;
  const double mDt;

  const uint mBW;
  const uint mBWP;

  const uint mX; 
  const uint mY; 
  const uint mZ;
};

  /*void advectionBlock(T * gridA, T * gridB, T * gridC, U * fieldA, U * fieldB,
      const uint &X, const uint &Y, const uint &Z) {

    // Backward
    for(uint kk = 0; kk < NB; kk++)
      for(uint jj = 0; jj < NB; jj++)
        for(uint ii = 0; ii < NB; ii++)
          for(uint k = BWP + (kk * NE) + omp_get_thread_num(); k < BWP + ((kk+1) * NE); k+=omp_get_num_threads()) 
            for(uint j = BWP + (jj * NE); j < BWP + ((jj+1) * NE); j++)
              for(uint i = BWP + (ii * NE); i < BWP + ((ii+1) * NE); i++)
                bfecc<T,U,Indexer>(gridB,gridA,gridA,fieldA,dx,dt,-1.0,0.0,1.0,BW,i,j,k,X,Y,Z);

    #pragma omp barrier

    // Forward 
    for(uint kk = 0; kk < NB; kk++)
      for(uint jj = 0; jj < NB; jj++)
        for(uint ii = 0; ii < NB; ii++)
          for(uint k = BWP + (kk * NE) + omp_get_thread_num(); k < BWP + ((kk+1) * NE); k+=omp_get_num_threads())
            for(uint j = BWP + (jj * NE); j < BWP + ((jj+1) * NE); j++)
              for(uint i = BWP + (ii * NE); i < BWP + ((ii+1) * NE); i++)
                bfecc<T,U,Indexer>(gridC,gridA,gridB,fieldA,dx,dt,1.0,1.5,-0.5,BW,i,j,k,X,Y,Z);

    #pragma omp barrier
   
    // Backward
    for(uint kk = 0; kk < NB; kk++)
      for(uint jj = 0; jj < NB; jj++)
        for(uint ii = 0; ii < NB; ii++)
          for(uint k = BWP + (kk * NE) + omp_get_thread_num(); k < BWP + ((kk+1) * NE); k+=omp_get_num_threads())
            for(uint j = BWP + (jj * NE); j < BWP + ((jj+1) * NE); j++)
              for(uint i = BWP + (ii * NE); i < BWP + ((ii+1) * NE); i++)
                bfecc<T,U,Indexer>(gridA,gridA,gridC,fieldA,dx,dt,-1.0,0.0,1.0,BW,i,j,k,X,Y,Z);
  }*/