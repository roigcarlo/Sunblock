#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "defines.h"

// "The beast"

class Interpolator {
public:

  typedef Indexer       IndexType;

  Interpolator() {}
  ~Interpolator() {}

  static void Interpolate(
    PrecisionType * OldPhi,
    PrecisionType * NewPhi,
    PrecisionType * Coords,
    const uint &Dim) {}
};


class TrilinealInterpolator : public Interpolator {
public:

  TrilinealInterpolator() {}
  ~TrilinealInterpolator() {}

  static void Interpolate(
      Block * block,
      PrecisionType * OldPhi,
      PrecisionType * NewPhi,
      PrecisionType * Coords,
      const uint &Dim ) {

    uint pi,pj,pk,ni,nj,nk;

    Utils::GlobalToLocal(Coords,block->rIdx,Dim);

    pi = (uint)(Coords[0]); ni = pi+1;
    pj = (uint)(Coords[1]); nj = pj+1;
    pk = (uint)(Coords[2]); nk = pk+1;

    PrecisionType Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    for(int d = 0; d < 3; d++) {
      *(NewPhi+d) = (
        OldPhi[IndexType::GetIndex(pi,pj,pk,block->mPaddY,block->mPaddZ)+d] * (    Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(ni,pj,pk,block->mPaddY,block->mPaddZ)+d] * (1 - Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(pi,nj,pk,block->mPaddY,block->mPaddZ)+d] * (    Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(ni,nj,pk,block->mPaddY,block->mPaddZ)+d] * (1 - Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(pi,pj,nk,block->mPaddY,block->mPaddZ)+d] * (    Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(ni,pj,nk,block->mPaddY,block->mPaddZ)+d] * (1 - Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(pi,nj,nk,block->mPaddY,block->mPaddZ)+d] * (    Nx) * (1 - Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(ni,nj,nk,block->mPaddY,block->mPaddZ)+d] * (1 - Nx) * (1 - Ny) * (1 - Nz)
      );
    }
  }

private:

};

#endif
