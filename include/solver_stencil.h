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
      gridB[cell*rDim+d] = mDiffTerm * (
        gridA[(cell - 1)*rDim+d]   +                       // Left
        gridA[(cell + 1)*rDim+d]   +                       // Right
        gridA[(cell - (X+BW))*rDim+d]   +                  // Up
        gridA[(cell + (X+BW))*rDim+d]   +                  // Down
        gridA[(cell - (Y+BW)*(X+BW))*rDim+d] +             // Front
        gridA[(cell + (Y+BW)*(X+BW))*rDim+d] -             // Back
        6 * gridA[cell*rDim+d]);                           // Self
    }
  }


public:

  StencilSolver(Block * block, const double& Dt) :
      Solver(block,Dt) {

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

    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          stencilCross(pPhiA,pPhiB,cell++,rX,rY,rZ);
        }
      }
    }

    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        uint cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(uint i = rBWP; i < rX + rBWP; i++) {
          for (uint d = 0; d < rDim; d++) {
            pPhiA[cell*rDim+d] += pPhiB[cell*rDim+d];
          }
          cell++;
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

  void SetDiffTerm(double diffTerm) {
    mDiffTerm = diffTerm;
  }

private:

  double mDiffTerm;

};
