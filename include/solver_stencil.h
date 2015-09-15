#include "solver.h"
#include "simd.h"

class StencilSolver : public Solver {
private:

  inline void calculateAcceleration(
      PrecisionType * gridA,
      PrecisionType * gridB,
      PrecisionType * gridC,
      const size_t &cell,
      const size_t &Dim) {

    for (size_t d = 0; d < Dim; d++) {
      gridC[cell*Dim+d] = (gridB[cell*Dim+d] - gridA[cell*Dim+d])/rDt;
    }
  }

  inline void smoothing(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const size_t &cell,
      const size_t &Dim) {

    for (size_t d = 0; d < Dim; d++) {
      gridB[cell*Dim+d] = (
        gridA[(cell - 1)*Dim+d]   +               // Left
        gridA[(cell + 1)*Dim+d]   +               // Right
        gridA[(cell - (rX+BW))*Dim+d]   +         // Up
        gridA[(cell + (rX+BW))*Dim+d]   +         // Down
        gridA[(cell - (rY+BW)*(rX+BW))*Dim+d] +   // Front
        gridA[(cell + (rY+BW)*(rX+BW))*Dim+d]) / 1.0f/6.0f;
    }
  }

  inline void lapplacian(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const size_t &cell,
      const size_t &Dim) {

    for (size_t d = 0; d < Dim; d++) {
      gridB[cell*Dim+d] = (
        gridA[(cell - 1)*Dim+d]   +                  // Left
        gridA[(cell + 1)*Dim+d]   +                  // Right
        gridA[(cell - (rX+BW))*Dim+d]   +             // Up
        gridA[(cell + (rX+BW))*Dim+d]   +             // Down
        gridA[(cell - (rY+BW)*(rX+BW))*Dim+d] +        // Front
        gridA[(cell + (rY+BW)*(rX+BW))*Dim+d] -
        6.0f * gridA[(cell)*Dim+d]) / (rDx * rDx * rDx);
    }
  }

  inline void gradient(
      PrecisionType * press,
      PrecisionType * gridB,
      const size_t &cell ) {

    PrecisionType pressGrad[3];

    pressGrad[0] = (
      press[(cell + 1)] -
      press[(cell - 1)]);

    pressGrad[1] = (
      press[(cell + (rX+BW))] -
      press[(cell - (rX+BW))]);

    pressGrad[2] = (
      press[(cell + (rY+BW)*(rX+BW))] -
      press[(cell - (rY+BW)*(rX+BW))]);

    for (size_t d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d] * 0.5f * rIdx;
    }
  }

  inline void divergence(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const size_t &cell ) {

    gridB[cell] =
      (gridA[(cell + 1) * rDim + 0] - gridA[(cell - 1) * rDim + 0]) +
      (gridA[(cell + (rX+BW)) *rDim + 1] - gridA[(cell - (rX+BW)) * rDim + 1]) +
      (gridA[(cell + (rY+BW)*(rX+BW)) * rDim + 2] - gridA[(cell - (rY+BW)*(rX+BW)) * rDim + 2]);

    gridB[cell] *= (0.5f * rIdx);
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
    PrecisionType * initVel   = pBuffers[VELOCITY];
    PrecisionType * vel       = pBuffers[AUX_3D_1];
    PrecisionType * acc       = pBuffers[AUX_3D_3];
    PrecisionType * press     = pBuffers[PRESSURE];

    PrecisionType * pressGrad = pBuffers[AUX_3D_4];
    PrecisionType * velLapp   = pBuffers[AUX_3D_2];

    PrecisionType * velDiv    = pBuffers[AUX_3D_5];
    PrecisionType * pressDiff = pBuffers[AUX_3D_6];
    PrecisionType * pressLapp = pBuffers[AUX_3D_7];

    // PrecisionType force[3]    = {0.0f, 0.0f, -9.8f};
    PrecisionType force[3]    = {0.0f, 0.0f, 0.0f};

    size_t listL[rX*rX];
    size_t listR[rX*rX];
    size_t listF[rX*rX];
    size_t listB[rX*rX];
    size_t listT[rX*rX];
    size_t listD[rX*rX];

    size_t normalL[3] = {0,-1,0};
    size_t normalR[3] = {0,1,0};
    size_t normalF[3] = {-1,0,0};
    size_t normalB[3] = {1,0,0};
    size_t normalT[3] = {0,0,-1};
    size_t normalD[3] = {0,0,1};

    uint counter = 0;

    for(uint a = rBWP; a < rZ + rBWP; a++) {
      for(uint b = rBWP; b < rY + rBWP; b++) {

        listL[counter] = a*(rZ+rBW)*(rY+rBW)+2*(rZ+rBW)+b;
        listR[counter] = a*(rZ+rBW)*(rY+rBW)+(rY-1)*(rZ+rBW)+b;
        listF[counter] = a*(rZ+rBW)*(rY+rBW)+b*(rZ+rBW)+2;
        listB[counter] = a*(rZ+rBW)*(rY+rBW)+b*(rZ+rBW)+(rX-1);
        listT[counter] = 1*(rZ+rBW)*(rY+rBW)+a*(rZ+rBW)+b;
        listD[counter] = rZ*(rZ+rBW)*(rY+rBW)+a*(rZ+rBW)+b;

        counter++;
      }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Calculate acceleration
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          calculateAcceleration(initVel,vel,acc,cell++,3);
        }
      }
    }

    // Apply the pressure gradient
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          gradient(press,pressGrad,cell++);
        }
      }
    }

    // divergence of the gradient of the velocity
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          lapplacian(vel,velLapp,cell++,3);
        }
      }
    }

    // Combine it all together and store it back in A
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          if(!(pFlags[cell] & FIXED_VELOCITY_X))
            initVel[cell*rDim+0] += (rMu * velLapp[cell*rDim+0] - pressGrad[cell*rDim+0] + force[0] / rRo - acc[cell*rDim+0]) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Y))
            initVel[cell*rDim+1] += (rMu * velLapp[cell*rDim+1] - pressGrad[cell*rDim+1] + force[1] / rRo - acc[cell*rDim+1]) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Z))
            initVel[cell*rDim+2] += (rMu * velLapp[cell*rDim+2] - pressGrad[cell*rDim+2] + force[2] / rRo - acc[cell*rDim+2]) * rDt;
          cell++;
        }
      }
    }

    // Combine it all together and store it back in A
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          divergence(initVel,velDiv,cell);
          pressDiff[cell] = -rRo*rCC2*rDt * velDiv[cell];
          cell++;
        }
      }
    }

    // Combine it all together and store it back in A
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          smoothing(pressDiff,pressLapp,cell,1);
          cell++;
        }
      }
    }

    // Combine it all together and store it back in A
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          if(!(pFlags[cell] & FIXED_PRESSURE))
            press[cell] += pressDiff[cell];
          cell++;
        }
      }
    }

    // applyBc(press,listL,rX*rX,normalL,1,1);
    // applyBc(press,listR,rX*rX,normalR,1,1);
    // applyBc(press,listF,rX*rX,normalF,1,1);
    // applyBc(press,listB,rX*rX,normalB,1,1);

    // copyUpToDown(initVel,3);

    // applyBc(initVel,listT,rX*rX,normalT,1,3);
    // applyBc(initVel,listD,rX*rX,normalD,1,3);

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
