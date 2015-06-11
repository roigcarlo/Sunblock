#include "defines.h"

// "The beast"

template <
  typename ResultType,
  typename IndexType,
  typename BlockType
  >
class Interpolator {
public:

  Interpolator() {}
  ~Interpolator() {}

  static void Interpolate(VariableType * OldPhi, VariableType * NewPhi, Variable3DType Coords) {}
  static void Interpolate(Variable3DType * OldPhi, Variable3DType * NewPhi, Variable3DType Coords) {}
};

template <
  typename ResultType,
  typename IndexType,
  typename BlockType
  >
class TrilinealInterpolator : public Interpolator<ResultType,IndexType,BlockType> {
public:

  TrilinealInterpolator() {}
  ~TrilinealInterpolator() {}

  static void CalculateFactors( 
      BlockType * block,
      Variable3DType Coords,
      double * Factors) {

    uint pi,pj,pk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = (uint)(Coords[0]);
    pj = (uint)(Coords[1]);
    pk = (uint)(Coords[2]);

    double Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    Factors[0] = (    Nx) * (    Ny) * (    Nz);
    Factors[1] = (1 - Nx) * (    Ny) * (    Nz);
    Factors[2] = (    Nx) * (1 - Ny) * (    Nz);
    Factors[3] = (1 - Nx) * (1 - Ny) * (    Nz);
    Factors[4] = (    Nx) * (    Ny) * (1 - Nz);
    Factors[5] = (1 - Nx) * (    Ny) * (1 - Nz);
    Factors[6] = (    Nx) * (1 - Ny) * (1 - Nz);
    Factors[7] = (1 - Nx) * (1 - Ny) * (1 - Nz); 
  }

  static void Interpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords,
      double * Factors) {

    uint pi,pj,pk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = (uint)(Coords[0]);
    pj = (uint)(Coords[1]);
    pk = (uint)(Coords[2]);

    uint cellIndex = IndexType::GetIndex(block,pi,pj,pk);

    double a = OldPhi[cellIndex                ] * Factors[0];
    double b = OldPhi[cellIndex + block->mPaddA] * Factors[1];
    double c = OldPhi[cellIndex + block->mPaddB] * Factors[2];
    double d = OldPhi[cellIndex + block->mPaddC] * Factors[3];
    double e = OldPhi[cellIndex + block->mPaddD] * Factors[4];
    double f = OldPhi[cellIndex + block->mPaddE] * Factors[5];
    double g = OldPhi[cellIndex + block->mPaddF] * Factors[6];
    double h = OldPhi[cellIndex + block->mPaddG] * Factors[7];

    (*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void ReverseInterpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords,
      double * Factors) {

    uint pi,pj,pk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = (uint)(Coords[0]);
    pj = (uint)(Coords[1]);
    pk = (uint)(Coords[2]);

    uint cellIndex = IndexType::GetIndex(block,pi,pj,pk);

    double a = OldPhi[cellIndex                ] * Factors[3];
    double b = OldPhi[cellIndex + block->mPaddA] * Factors[2];
    double c = OldPhi[cellIndex + block->mPaddB] * Factors[1];
    double d = OldPhi[cellIndex + block->mPaddC] * Factors[0];
    double e = OldPhi[cellIndex + block->mPaddD] * Factors[7];
    double f = OldPhi[cellIndex + block->mPaddE] * Factors[6];
    double g = OldPhi[cellIndex + block->mPaddF] * Factors[5];
    double h = OldPhi[cellIndex + block->mPaddG] * Factors[4];

    (*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void Interpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords) {

    uint pi,pj,pk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = (uint)floor(Coords[0]); ni = pi + 1;
  	pj = (uint)floor(Coords[1]); nj = pj + 1;
  	pk = (uint)floor(Coords[2]); nk = pk + 1;

    double Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    uint cellIndex = IndexType::GetIndex(block,pi,pj,pk);

    (*NewPhi) = 0.0;

    (*NewPhi) += OldPhi[cellIndex                ] * (    Nx) * (    Ny) * (    Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddA] * (1 - Nx) * (    Ny) * (    Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddB] * (    Nx) * (1 - Ny) * (    Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddC] * (1 - Nx) * (1 - Ny) * (    Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddD] * (    Nx) * (    Ny) * (1 - Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddE] * (1 - Nx) * (    Ny) * (1 - Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddF] * (    Nx) * (1 - Ny) * (1 - Nz);
    (*NewPhi) += OldPhi[cellIndex + block->mPaddG] * (1 - Nx) * (1 - Ny) * (1 - Nz);

    //(*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void Interpolate(
      BlockType * block,
      Variable3DType * OldPhi,
      Variable3DType * NewPhi,
      Variable3DType Coords) {
 
    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = (uint)(Coords[0]); ni = pi+1;
    pj = (uint)(Coords[1]); nj = pj+1;
    pk = (uint)(Coords[2]); nk = pk+1;

    VariableType Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    for(int d = 0; d < 3; d++) {
      (*NewPhi)[d] = (
        OldPhi[IndexType::GetIndex(block,pi,pj,pk)][d] * (    Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(block,ni,pj,pk)][d] * (1 - Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(block,pi,nj,pk)][d] * (    Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(block,ni,nj,pk)][d] * (1 - Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(block,pi,pj,nk)][d] * (    Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(block,ni,pj,nk)][d] * (1 - Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(block,pi,nj,nk)][d] * (    Nx) * (1 - Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(block,ni,nj,nk)][d] * (1 - Nx) * (1 - Ny) * (1 - Nz)
      );
    }
  }

private:

};