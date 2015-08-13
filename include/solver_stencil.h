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
      const uint &Z) {

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = gridA[cell*rDim+d] - (
        gridA[(cell - 1)*rDim+d]   +                  // Left
        gridA[(cell + 1)*rDim+d]   +                  // Right
        gridA[(cell - (X+BW))*rDim+d]   +             // Up
        gridA[(cell + (X+BW))*rDim+d]   +             // Down
        gridA[(cell - (Y+BW)*(X+BW))*rDim+d] +        // Front
        gridA[(cell + (Y+BW)*(X+BW))*rDim+d]) *
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

  inline void divergenceVelocityP(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    gridB[cell] = 0;

    PrecisionType a = (gridA[(cell + 1)            *rDim+0] - gridA[(cell - 1)            *rDim+0]);
    PrecisionType b = (gridA[(cell + (X+BW))       *rDim+1] - gridA[(cell - (X+BW))       *rDim+1]);
    PrecisionType c = (gridA[(cell + (Y+BW)*(X+BW))*rDim+2] - gridA[(cell - (Y+BW)*(X+BW))*rDim+2]);

    printf("\t:%.17f --- %.17f --- %.17f\n",gridA[(cell+1) *rDim+0],gridA[(cell-1)*rDim+0],a);
    printf("\t:%.17f --- %.17f --- %.17f\n",gridA[(cell+(X+BW))*rDim+1],gridA[(cell-(X+BW))*rDim+1],b);
    printf("\t:%.17f --- %.17f --- %.17f\n",gridA[(cell+(Y+BW)*(X+BW))*rDim+2],gridA[(cell-(Y+BW)*(X+BW))*rDim+2],c);

    gridB[cell] = (a + b + c) * 0.5f * rIdx;
  }

  inline void fixedGradient(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    PrecisionType pressGrad[3];

    pressGrad[0] = 0.0f;// 1.0f/64.0f;
    pressGrad[1] = 0.0f;
    pressGrad[2] = 0.0f;

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d];
    }

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
    PrecisionType * phiLapplacian     = pPhiC;

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
          stencilCross(phi,phiLapplacian,cell++,rX,rY,rZ);
        }
      }
    }

    // Combine it all together and store it back in A
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          for (uint d = 0; d < rDim; d++) {
            phi[cell*rDim+d] += (-rMu * phiLapplacian[cell*rDim+d] + pressureGradient[cell*rDim+d] + rRo * force[rDim+d]) * rDt;
          }
          cell++;
        }
      }
    }

    printf("SUBS: %f ---- %f ---- %f\n",rDt,rPdt,rDt/rPdt);

    for(uint ss = 0 ; ss < 1; ss++) {

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
            for (uint d = 0; d < rDim; d++) {
              phi[cell*rDim+d] += pressureGradient[cell*rDim+d] * rPdt;
            }
            cell++;
          }
        }
      }

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            pressure[cell] += pressDiff[cell];
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
