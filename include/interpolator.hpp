#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "defines.hpp"
#include "utils.hpp"

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
    const size_t &Dim) {}
};


class TrilinealInterpolator : public Interpolator {
public:

  TrilinealInterpolator() {}
  ~TrilinealInterpolator() {}

  static void Interpolate(
      const PrecisionType & X,
      const PrecisionType & Y,
      const PrecisionType & Z,
      const PrecisionType & Idx,
      PrecisionType * OldPhi,
      PrecisionType * NewPhi,
      PrecisionType * Coords,
      const size_t &Dim) {

    uint pi,pj,pk,ni,nj,nk;

    size_t PaddZ = (Z+BW)*(Y+BW);
    size_t PaddY = (Y+BW);

    for(size_t i = 0; i < Dim; i++) {
      Coords[i] = Coords[i] < 0.0f ? 0.0f : Coords[i] > 1.0f ? 1.0f : Coords[i];
    }

    Utils::GlobalToLocal(Coords,Idx,Dim);

    pi = (uint)(Coords[0]); ni = pi+1;
    pj = (uint)(Coords[1]); nj = pj+1;
    pk = (uint)(Coords[2]); nk = pk+1;

    PrecisionType Nx, Ny, Nz;

    Nx = 1-(Coords[0] - pi);
    Ny = 1-(Coords[1] - pj);
    Nz = 1-(Coords[2] - pk);

    for(size_t d = 0; d < Dim; d++) {
      *(NewPhi+d) = (
        OldPhi[IndexType::GetIndex(pi,pj,pk,PaddY,PaddZ)*Dim+d] * (    Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(ni,pj,pk,PaddY,PaddZ)*Dim+d] * (1 - Nx) * (    Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(pi,nj,pk,PaddY,PaddZ)*Dim+d] * (    Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(ni,nj,pk,PaddY,PaddZ)*Dim+d] * (1 - Nx) * (1 - Ny) * (    Nz) +
        OldPhi[IndexType::GetIndex(pi,pj,nk,PaddY,PaddZ)*Dim+d] * (    Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(ni,pj,nk,PaddY,PaddZ)*Dim+d] * (1 - Nx) * (    Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(pi,nj,nk,PaddY,PaddZ)*Dim+d] * (    Nx) * (1 - Ny) * (1 - Nz) +
        OldPhi[IndexType::GetIndex(ni,nj,nk,PaddY,PaddZ)*Dim+d] * (1 - Nx) * (1 - Ny) * (1 - Nz)
      );
    }
  }

private:

};

#endif
