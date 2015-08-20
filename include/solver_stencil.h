#include "solver.h"
#include "simd.h"

class StencilSolver : public Solver {
private:

  // This does not belong here! put it in a class
  inline void calculateAcceleration(
      PrecisionType * gridA,
      PrecisionType * gridB,
      PrecisionType * gridC,
      const size_t &cell,
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const size_t &Dim) {

    for (size_t d = 0; d < Dim; d++) {
      gridC[cell*Dim+d] = (gridB[cell*Dim+d] - gridA[cell*Dim+d])/rDt;
    }
  }

  inline void lapplacian(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const size_t &cell,
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const size_t &Dim) {

    for (size_t d = 0; d < Dim; d++) {
      gridB[cell*Dim+d] = (
        gridA[(cell - 1)*Dim+d]   +                  // Left
        gridA[(cell + 1)*Dim+d]   +                  // Right
        gridA[(cell - (X+BW))*Dim+d]   +             // Up
        gridA[(cell + (X+BW))*Dim+d]   +             // Down
        gridA[(cell - (Y+BW)*(X+BW))*Dim+d] +        // Front
        gridA[(cell + (Y+BW)*(X+BW))*Dim+d]) /
        1.0f/6.0f;
    }
  }

  inline void gradientPressure(
      PrecisionType * press,
      PrecisionType * gridB,
      const size_t &cell,
      const size_t &X,
      const size_t &Y,
      const size_t &Z) {

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

    for (size_t d = 0; d < rDim; d++) {
      gridB[cell*rDim+d] = pressGrad[d] * 0.5f * rIdx;
    }
  }

  inline void divergenceVelocity(
      PrecisionType * gridA,
      PrecisionType * gridB,
      const size_t &cell,
      const size_t &X,
      const size_t &Y,
      const size_t &Z) {

    gridB[cell] = 0;

    gridB[cell] += (gridA[(cell + 1)              *rDim+0] - gridA[(cell - 1)              *rDim+0]) * 0.5f * rIdx;
    gridB[cell] += (gridA[(cell + (X+BW))         *rDim+1] - gridA[(cell - (X+BW))         *rDim+1]) * 0.5f * rIdx;
    gridB[cell] += (gridA[(cell + (Y+rBW)*(X+rBW))*rDim+2] - gridA[(cell - (Y+rBW)*(X+rBW))*rDim+2]) * 0.5f * rIdx;
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
    PrecisionType * initVel   = pPhiA;
    PrecisionType * vel       = pPhiB;
    PrecisionType * acc       = pPhiD;
    PrecisionType * press     = pPressA;

    PrecisionType * pressGrad = pPhiB;
    PrecisionType * velLapp   = pPhiC;

    PrecisionType * velDiv    = pPressB;

    PrecisionType force[3]    = {0.0f, 0.0f, -9.8f};

    size_t listL[16*16];
    size_t listR[16*16];
    size_t listF[16*16];
    size_t listB[16*16];
    size_t listT[16*16];
    size_t listD[16*16];

    size_t normalL[3] = {0,-1,0};
    size_t normalR[3] = {0,1,0};
    size_t normalF[3] = {-1,0,0};
    size_t normalB[3] = {1,0,0};
    size_t normalT[3] = {0,0,-1};
    size_t normalD[3] = {0,0,1};

    uint counter = 0;

    for(uint a = rBWP; a < rZ + rBWP; a++) {
      for(uint b = rBWP; b < rY + rBWP; b++) {

        listL[counter] = a*(rZ+rBW)*(rY+rBW)+1*(rZ+rBW)+b;
        listR[counter] = a*(rZ+rBW)*(rY+rBW)+(rY)*(rZ+rBW)+b;
        listF[counter] = a*(rZ+rBW)*(rY+rBW)+b*(rZ+rBW)+1;
        listB[counter] = a*(rZ+rBW)*(rY+rBW)+b*(rZ+rBW)+(rX);
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
          calculateAcceleration(initVel,vel,acc,cell++,rX,rY,rZ,3);
        }
      }
    }

    // Apply the pressure gradient
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          gradientPressure(press,pressGrad,cell++,rX,rY,rZ);
        }
      }
    }

    // divergence of the gradient of the velocity
    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          lapplacian(initVel,velLapp,cell++,rX,rY,rZ,3);
        }
      }
    }

    // Combine it all together and store it back in A
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          if(!(pFlags[cell] & FIXED_VELOCITY_X))
            initVel[cell*rDim+0] += (/*rMu * velLapp[cell*rDim+0]*/ - pressGrad[cell*rDim+0] + force[0] / rRo /*+ acc[cell*rDim+0]*/) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Y))
            initVel[cell*rDim+1] += (/*rMu * velLapp[cell*rDim+1]*/ - pressGrad[cell*rDim+1] + force[1] / rRo /*+ acc[cell*rDim+1]*/) * rDt;
          if(!(pFlags[cell] & FIXED_VELOCITY_Z))
            initVel[cell*rDim+2] += (/*rMu * velLapp[cell*rDim+2]*/ - pressGrad[cell*rDim+2] + force[2] / rRo /*+ acc[cell*rDim+2]*/) * rDt;

          cell++;
        }
      }
    }

    // Combine it all together and store it back in A
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        size_t cell = k*(rZ+rBW)*(rY+rBW)+j*(rY+BW)+rBWP;
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          divergenceVelocity(initVel,velDiv,cell,rX,rY,rZ);
          press[cell] += -rRo*rCC2*rDt * velDiv[cell];
          cell++;
        }
      }
    }

    applyBc(press,listL,16*16,normalL,1,1);
    applyBc(press,listR,16*16,normalR,1,1);
    applyBc(press,listF,16*16,normalF,1,1);
    applyBc(press,listB,16*16,normalB,1,1);
    applyBc(press,listT,16*16,normalT,1,1);
    applyBc(press,listD,16*16,normalD,1,1);

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
