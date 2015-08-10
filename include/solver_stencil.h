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
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    PrecisionType pressGrad[3];

    pressGrad[0] = (
      pPressA[(cell + 1)] -
      pPressA[(cell - 1)]) * 0.5 * rIdx;

    pressGrad[1] = (
      pPressA[(cell + (X+BW))] -
      pPressA[(cell - (X+BW))]) * 0.5 * rIdx;

    pressGrad[2] = (
      pPressA[(cell + (Y+BW)*(X+BW))] -
      pPressA[(cell - (Y+BW)*(X+BW))]) * 0.5 * rIdx;

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d];
    }
  }

  inline void divergenceGradientVelocity(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = 0;
    }

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+0] += ((gridA[(cell+1)*rDim+d] + gridA[(cell-1)*rDim+d])  * 0.5 * rIdx);
    }

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+1] += ((gridA[(cell + (X+BW))*rDim+d] + gridA[(cell - (X+BW))*rDim+d])  * 0.5 * rIdx);
    }

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+2] += ((gridA[(cell + (Y+BW)*(X+BW))*rDim+d] + gridA[(cell - (Y+BW)*(X+BW))*rDim+d])  * 0.5 * rIdx);
    }
  }

  inline void divergenceGradientVelocity2(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = 0;
    }

    for (uint d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] += ((gridA[(cell+1)*rDim+d]               - gridA[(cell-1)*rDim+d])  * 0.5 * rIdx);
      gridB[cell*rDim+d] += ((gridA[(cell + (X+BW))*rDim+d]        - gridA[(cell - (X+BW))*rDim+d])  * 0.5 * rIdx);
      gridB[cell*rDim+d] += ((gridA[(cell + (Y+BW)*(X+BW))*rDim+d] - gridA[(cell - (Y+BW)*(X+BW))*rDim+d])  * 0.5 * rIdx);
    }
  }

  inline void divergenceVelocity(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const uint &cell,
      const uint &X,
      const uint &Y,
      const uint &Z) {

    gridB[cell] = 0;

    gridB[cell] += ((gridA[(cell+1)*rDim+0]               - gridA[(cell-1)*rDim+0]) * 0.5 * rIdx);
    gridB[cell] += ((gridA[(cell + (X+BW))*rDim+1]        - gridA[(cell - (X+BW))*rDim+1]) * 0.5 * rIdx);
    gridB[cell] += ((gridA[(cell + (Y+BW)*(X+BW))*rDim+2] - gridA[(cell - (Y+BW)*(X+BW))*rDim+2]) * 0.5 * rIdx);
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

    // Apply the pressure gradient
    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          gradientPressure(pPhiA,pPhiB,cell++,rX,rY,rZ);
        }
      }
    }

    // divergence of the gradient of the velocity
    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          divergenceGradientVelocity2(pPhiA,pPhiD,cell++,rX,rY,rZ);
        }
      }
    }

    // Combine it all together and store it back in A
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          for (uint d = 0; d < rDim; d++) {
            pPhiD[cell*rDim+d] = -rMu * pPhiD[cell*rDim+d] + pPhiB[cell*rDim+d];
            pPhiA[cell*rDim+d] = pPhiA[cell*rDim+d] - pPhiD[cell*rDim+d] * rDt;
          }
          cell++;
        }
      }
    }

    copyLeft(pPhiA);
    copyRight(pPhiA);

    printf("%f\n",rDt/rPdt);
    for(uint ss = 0 ; ss < 1; ss++) {

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            divergenceVelocity(pPhiA,pPressB,cell,rX,rY,rZ);
            pPressB[cell] *= -rRo * 343.2f*343.2f * rPdt * 0.0001;
            cell++;
          }
        }
      }

      std::swap(pPressA,pPressB);

      for(uint k = rBWP; k < rZ + rBWP; k++) {
        for(uint j = rBWP; j < rY + rBWP; j++) {
          uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
          for(uint i = rBWP; i < rX + rBWP; i++) {
            gradientPressure(pPhiB,pPhiB,cell,rX,rY,rZ);
            for (uint d = 0; d < rDim; d++) {
              pPhiA[cell*rDim+d] += pPhiB[cell*rDim+d] * rPdt;
            }
            cell++;
          }
        }
      }

      std::swap(pPressA,pPressB);

      // for(uint k = rBWP; k < rZ + rBWP; k++) {
      //   for(uint j = rBWP; j < rY + rBWP; j++) {
      //     uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
      //     for(uint i = rBWP; i < rX + rBWP; i++) {
      //       pPressA[cell] += pPressB[cell];
      //       cell++;
      //     }
      //   }
      // } 

    }

    copyLeft(pPhiA);
    copyRight(pPhiA);

    // copyLeftToRight(pPhiA);

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
