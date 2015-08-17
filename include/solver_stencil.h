#include "solver.h"
#include "simd.h"

class StencilSolver : public Solver {
private:

  // This does not belong here! put it in a class
  inline void stencilCross(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z,
      const uint &Dim ) {

    for (uint d = 0; d < Dim; d++) {
      gridB[cell*Dim+d] = (
        gridA[(cell - 1)*Dim+d]   +                  // Left
        gridA[(cell + 1)*Dim+d]   +                  // Right
        gridA[(cell - (X+BW))*Dim+d]   +             // Up
        gridA[(cell + (X+BW))*Dim+d]   +             // Down
        gridA[(cell - (Y+BW)*(X+BW))*Dim+d] +        // Front
        gridA[(cell + (Y+BW)*(X+BW))*Dim+d]) *
        1.0f/6.0f;
    }
  }

  inline void gradientPressure(
      PrecisionType * press,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    PrecisionType pressGrad[3];

    pressGrad[0] = (
      press[(cell + 1)] -
      press[(cell - 1)]);

    pressGrad[1] = (
      press[(cell + (X+BW))] -
      press[(cell - (X+BW))]);

    pressGrad[2] = (
      press[(cell + (Y+BW)*(X+BW))] -
      press[(cell - (Y+BW)*(X+BW))]);

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d] * 0.5f * rIdx;
    }
  }

  inline void gradientPressureP(
      PrecisionType * press,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    PrecisionType pressGrad[3];

    pressGrad[0] = (
      press[(cell + 1)] -
      press[(cell - 1)]);

    pressGrad[1] = (
      press[(cell + (X+BW))] -
      press[(cell - (X+BW))]);

    pressGrad[2] = (
      press[(cell + (Y+BW)*(X+BW))] -
      press[(cell - (Y+BW)*(X+BW))]);

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d] * 0.5f * rIdx;
    }

    printf("\t:%.10f --- %.10f --- %.10f\n",press[(cell + 1)],press[(cell - 1)],gridB[cell*rDim+0]);
    printf("\t:%.10f --- %.10f --- %.10f\n",press[(cell + (X+BW))],press[(cell - (X+BW))],gridB[cell*rDim+1]);
    printf("\t:%.10f --- %.10f --- %.10f\n",press[(cell + (Y+BW)*(X+BW))],press[(cell - (Y+BW)*(X+BW))],gridB[cell*rDim+2]);
  }

  inline void divergenceVelocity(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    gridB[cell] = 0;

    gridB[cell] += (gridA[(cell + 1)            *rDim+0] - gridA[(cell - 1)            *rDim+0]) * 0.5f * rIdx;
    gridB[cell] += (gridA[(cell + (X+BW))       *rDim+1] - gridA[(cell - (X+BW))       *rDim+1]) * 0.5f * rIdx;
    gridB[cell] += (gridA[(cell + (Y+BW)*(X+BW))*rDim+2] - gridA[(cell - (Y+BW)*(X+BW))*rDim+2]) * 0.5f * rIdx;
  }

public:

  StencilSolver(Block * block, const PrecisionType& Dt, const PrecisionType& Pdt) :
      Solver(block,Dt,Pdt) {

  }

  ~StencilSolver() {

  }

  void Prepare() {
  }

  void Finish() {
  }

  /**
   * Executes the solver in parallel
   **/
  void Execute() {

    // Alias for the buffers
    PrecisionType * phi               = pPhiA;
    PrecisionType * pressure          = pPressA;
    PrecisionType * pressureGradient  = pPhiB;
    PrecisionType * phiLapplacian     = pPhiD;

    PrecisionType * phiDivergence     = pPressB;
    PrecisionType * pressDiff         = pPressB;

    PrecisionType force[3]            = {0.0f, 0.0f, -9.8f};


    // Apply the pressure gradient
    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          gradientPressure(pressure,pressureGradient,cell++,rX,rY,rZ);
        }
      }
    }

    // divergence of the gradient of the velocity
    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          stencilCross(phi,phiLapplacian,cell++,rX,rY,rZ,3);
        }
      }
    }

    // Combine it all together and store it back in A
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          if(!(pFlags[cell] & FIXED_VELOCITY_X))
            phi[cell*rDim+0] += (-rMu * phiLapplacian[cell*rDim+0] + pressureGradient[cell*rDim+0] - rRo * force[0]) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Y))
            phi[cell*rDim+1] += (-rMu * phiLapplacian[cell*rDim+1] + pressureGradient[cell*rDim+1] - rRo * force[1]) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Z))
            phi[cell*rDim+2] += (-rMu * phiLapplacian[cell*rDim+2] + pressureGradient[cell*rDim+2] - rRo * force[2]) * rDt;

          for(uint d = 0; d < 3; d++) {
            pPhiC[cell*rDim+d] = pressureGradient[cell*rDim+d] - rRo * force[d];
          }

          cell++;
        }
      }
    }

    printf("SUBS: %f ---- %f ---- %f\n",rDt,rPdt,rDt/rPdt);

    for(uint ss = 0 ; ss < rDt/rPdt; ss++) {

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            divergenceVelocity(phi,phiDivergence,cell,rX,rY,rZ);
            pressDiff[cell] = phiDivergence[cell] * -rRo*rCC2*rPdt;
            cell++;
          }
        }
      }

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            gradientPressure(pressDiff,pressureGradient,cell,rX,rY,rZ);
            if(!(pFlags[cell] & FIXED_VELOCITY_X))
              phi[cell*rDim+0] -= pressureGradient[cell*rDim+0] * rPdt;
            if(!(pFlags[cell] & FIXED_VELOCITY_Y))
              phi[cell*rDim+1] -= pressureGradient[cell*rDim+1] * rPdt;
            if(!(pFlags[cell] & FIXED_VELOCITY_Z))
              phi[cell*rDim+2] -= pressureGradient[cell*rDim+2] * rPdt;
            cell++;
          }
        }
      }

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            if(!(pFlags[cell] & FIXED_PRESSURE))
              pressure[cell] += pressDiff[cell];
            cell++;
          }
        }
      }

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            stencilCross(pPressA,pPressB,cell++,rX,rY,rZ,1);
          }
        }
      }

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            pPressA[cell] = pPressB[cell];
            cell++;
          }
        }
      }

    }

  }

  void ExecuteVector() {

    // TODO: Implement this with the new arrays

    // uint cell, pcellb, pcelle;
    //
    // PrecisionType __attribute__((aligned(ALIGN))) tmpa[VP];
    // PrecisionType __attribute__((aligned(ALIGN))) tmpb[VP];
    //
    // for(uint kk = 0; kk < rNB; kk++) {
    //   for(uint jj = 0; jj < rNB; jj++) {
    //     for(uint k = rBWP + (kk * rNE); k < rBWP + ((kk+1) * rNE); k++) {
    //       for(uint j = rBWP + (jj * rNE); j < rBWP + ((jj+1) * rNE); j++) {
    //
    //         pcellb = k*(rX+rBW)*(rX+rBW)/VP+j*(rX+rBW)/VP+(rBWP/VP);
    //         pcelle = k*(rX+rBW)*(rX+rBW)/VP+j*(rX+rBW)/VP+((rX/VP) + rBWP/VP - 1);
    //
    //         VectorType * left  = &pPhiA[pcellb-1];
    //         VectorType * right = &pPhiA[pcellb+1];
    //         VectorType * bott  = &pPhiA[pcellb-(rX+rBW)/VP];
    //         VectorType * top   = &pPhiA[pcellb+(rX+rBW)/VP];
    //         VectorType * front = &pPhiA[pcellb-(rX+rBW)*(rX+rBW)/VP];
    //         VectorType * back  = &pPhiA[pcellb+(rX+rBW)*(rX+rBW)/VP];
    //
    //         // Prefix
    //         cell = pcellb;
    //         VSTORE(tmpa,pPhiA[pcelle]);
    //         tmpb[0] = 0.0;
    //         for(uint p = 1; p < VP; p++) {
    //           tmpb[p] = tmpa[p-1];
    //         }
    //
    //         left++;
    //         pPhiB[cell++] = VSTENCIL(VLOAD(tmpb),*right++,*top++,*bott++,*front++,*back++);
    //
    //         // Body
    //         for(uint i = rBWP/VP + 1; i < (rX/VP) + BWP/VP - 1; i++) {
    //           pPhiB[cell++] = VSTENCIL(*left++,*right++,*top++,*bott++,*front++,*back++);
    //         }
    //
    //         // Sufix
    //         cell = pcelle;
    //         VSTORE(tmpa,pPhiA[pcellb]);
    //         for(uint p = 1; p < VP; p++) {
    //           tmpb[p-1] = tmpa[p];
    //         }
    //         tmpb[VP-1] = 0.0;
    //
    //         pPhiB[cell] = VSTENCIL(*left++,VLOAD(tmpb),*top++,*bott++,*front++,*back++);
    //       }
    //     }
    //   }
    // }

  }

  void SetDiffTerm(PrecisionType diffTerm) {
    mDiffTerm = diffTerm;
  }

private:

  double mDiffTerm;

};
