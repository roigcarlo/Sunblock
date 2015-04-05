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

/**
 * Performs the bfecc operation over a given element
 * @phi:        Result of the operation
 * @phi_auxA:   first  operator
 * @phi_auxB:   second operator
 * @field:      field
 * @dx:         diferential of space
 * @dt:         diferential of time
 * sign:        direction of the interpolation ( -1.0 backward, 1.0 forward )
 * weightA:     weigth of the first operator (A)
 * weightB:     weigth of the second operator (B)
 **/ 
template <
    typename VariableType, 
    typename Variable3DType,
    typename IndexFunction,
    typename InterpolateFunction
>
void bfecc(VariableType * phi, VariableType * phi_auxA, VariableType * phi_auxB, 
    Variable3DType * field,
    const double &dx,   const double &dt,
    const double &sign, const double &weightA, const double &weightB,
    const uint &BW,
    const uint &i, const uint &j, const uint &k,
    const uint &X, const uint &Y, const uint &Z) {

  uint cell = IndexFunction(i,j,k,BW,X,Y,Z);

  Triple origin;
  Triple displacement;

  origin[0] = i * dx;
  origin[1] = j * dx;
  origin[2] = k * dx;
 
  for(int d = 0; d < 3; d++) {
    displacement[d] = origin[d] + sign * field[cell][d]*dt;
  }

  phi[cell] = weightA * phi_auxA[cell] + weightB * InterpolateFunction(displacement,phi_auxB,X,Y,Z);
}