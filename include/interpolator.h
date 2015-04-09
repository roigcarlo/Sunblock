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

    // printf("** %p\n",(void * )Factors);

    uint pi,pj,pk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = floor(Coords[0]);
    pj = floor(Coords[1]);
    pk = floor(Coords[2]);

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

  static void InvertFactors(double * Factors) {

      double aux;

      aux = Factors[0]; Factors[0] = Factors[7]; Factors[7] = aux;
      aux = Factors[1]; Factors[1] = Factors[6]; Factors[6] = aux;
      aux = Factors[2]; Factors[2] = Factors[5]; Factors[5] = aux;
      aux = Factors[3]; Factors[3] = Factors[4]; Factors[4] = aux;
  }

  static void Interpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords,
      double * Factors) {

    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = floor(Coords[0]); ni = pi+1;
    pj = floor(Coords[1]); nj = pj+1;
    pk = floor(Coords[2]); nk = pk+1;

    double a = OldPhi[IndexType::GetIndex(block,pi,pj,pk)] * Factors[0];
    double b = OldPhi[IndexType::GetIndex(block,ni,pj,pk)] * Factors[1];
    double c = OldPhi[IndexType::GetIndex(block,pi,nj,pk)] * Factors[2];
    double d = OldPhi[IndexType::GetIndex(block,ni,nj,pk)] * Factors[3];
    double e = OldPhi[IndexType::GetIndex(block,pi,pj,nk)] * Factors[4];
    double f = OldPhi[IndexType::GetIndex(block,ni,pj,nk)] * Factors[5];
    double g = OldPhi[IndexType::GetIndex(block,pi,nj,nk)] * Factors[6];
    double h = OldPhi[IndexType::GetIndex(block,ni,nj,nk)] * Factors[7];

    (*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void ReverseInterpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords,
      double * Factors) {

    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = floor(Coords[0]); ni = pi+1;
    pj = floor(Coords[1]); nj = pj+1;
    pk = floor(Coords[2]); nk = pk+1;

    double a = OldPhi[IndexType::GetIndex(block,pi,pj,pk)] * Factors[3];
    double b = OldPhi[IndexType::GetIndex(block,ni,pj,pk)] * Factors[2];
    double c = OldPhi[IndexType::GetIndex(block,pi,nj,pk)] * Factors[1];
    double d = OldPhi[IndexType::GetIndex(block,ni,nj,pk)] * Factors[0];
    double e = OldPhi[IndexType::GetIndex(block,pi,pj,nk)] * Factors[7];
    double f = OldPhi[IndexType::GetIndex(block,ni,pj,nk)] * Factors[6];
    double g = OldPhi[IndexType::GetIndex(block,pi,nj,nk)] * Factors[5];
    double h = OldPhi[IndexType::GetIndex(block,ni,nj,nk)] * Factors[4];

    (*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void Interpolate(
      BlockType * block,
      VariableType * OldPhi,
      VariableType * NewPhi,
      Variable3DType Coords) {

    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = floor(Coords[0]); ni = pi+1;
    pj = floor(Coords[1]); nj = pj+1;
    pk = floor(Coords[2]); nk = pk+1;

    double Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    double a = OldPhi[IndexType::GetIndex(block,pi,pj,pk)] * (    Nx) * (    Ny) * (    Nz);
    double b = OldPhi[IndexType::GetIndex(block,ni,pj,pk)] * (1 - Nx) * (    Ny) * (    Nz);
    double c = OldPhi[IndexType::GetIndex(block,pi,nj,pk)] * (    Nx) * (1 - Ny) * (    Nz);
    double d = OldPhi[IndexType::GetIndex(block,ni,nj,pk)] * (1 - Nx) * (1 - Ny) * (    Nz);
    double e = OldPhi[IndexType::GetIndex(block,pi,pj,nk)] * (    Nx) * (    Ny) * (1 - Nz);
    double f = OldPhi[IndexType::GetIndex(block,ni,pj,nk)] * (1 - Nx) * (    Ny) * (1 - Nz);
    double g = OldPhi[IndexType::GetIndex(block,pi,nj,nk)] * (    Nx) * (1 - Ny) * (1 - Nz);
    double h = OldPhi[IndexType::GetIndex(block,ni,nj,nk)] * (1 - Nx) * (1 - Ny) * (1 - Nz);

    (*NewPhi) = (a+b+c+d+e+f+g+h);
  }

  static void Interpolate(
      BlockType * block,
      Variable3DType * OldPhi,
      Variable3DType * NewPhi,
      Variable3DType Coords) {
 
    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx);

    pi = floor(Coords[0]); ni = pi+1;
    pj = floor(Coords[1]); nj = pj+1;
    pk = floor(Coords[2]); nk = pk+1;

    VariableType Nx, Ny, Nz;

    Nx = 1-(Coords[0] - floor(Coords[0]));
    Ny = 1-(Coords[1] - floor(Coords[1]));
    Nz = 1-(Coords[2] - floor(Coords[2]));

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