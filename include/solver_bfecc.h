#include "solver.h"

class BfeccSolver : public Solver<BfeccSolver> {
public:

  BfeccSolver(Block * block, const PrecisionType& Dt, const PrecisionType& Pdt) :
      Solver(block,Dt,Pdt) {

  }

  ~BfeccSolver() {

  }

  void Prepare_impl() {
  }

  void Finish_impl() {
  }

  /**
   * Executes the solver in parallel
   **/
  void Execute_impl() {

    PrecisionType * aux_3d_0 = pBuffers[VELOCITY];
    PrecisionType * aux_3d_1 = pBuffers[AUX_3D_1];
    PrecisionType * aux_3d_3 = pBuffers[AUX_3D_3];

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

    applyBc(aux_3d_0,listL,rX*rX,normalL,1,3);
    applyBc(aux_3d_0,listR,rX*rX,normalR,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          ApplyBack(aux_3d_1,aux_3d_0,aux_3d_0,i,j,k);
        }
      }
    }

    applyBc(aux_3d_1,listL,rX*rX,normalL,1,3);
    applyBc(aux_3d_1,listR,rX*rX,normalR,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          ApplyForth(aux_3d_3,aux_3d_1,aux_3d_0,i,j,k);
        }
      }
    }

    applyBc(aux_3d_3,listL,rX*rX,normalL,1,3);
    applyBc(aux_3d_3,listR,rX*rX,normalR,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          ApplyEcc(aux_3d_1,aux_3d_3,i,j,k);
        }
      }
    }

    applyBc(aux_3d_1,listL,rX*rX,normalL,1,3);
    applyBc(aux_3d_1,listR,rX*rX,normalR,1,3);

  }

  /**
   * Executes the solver in parallel using blocking
   **/
  void ExecuteTask_impl() {

    #define BOT(_i_) std::max(rBWP,(_i_ * rNE))
    #define TOP(_i_) rBWP + std::min(rNE*rNB-rBWP,((_i_+1) * rNE))

    PrecisionType * aux_3d_0 = pBuffers[VELOCITY];
    PrecisionType * aux_3d_1 = pBuffers[AUX_3D_1];
    PrecisionType * aux_3d_3 = pBuffers[AUX_3D_3];

    int bt = (rX + rBW);
    int ss = 2;
    int CFL = 1;
    int slice = bt*bt;
    int slice3D = slice * 3;

    #pragma omp parallel
    #pragma omp single
    {

      for(size_t kk = rBWP; kk < rZ + rBWP; kk+=ss) {
        #pragma omp task                                  \
          depend(in:aux_3d_0[(kk)*slice3D:(CFL)*slice3D]) \
          depend(out:aux_3d_1[(kk)*slice3D:(CFL)*slice3D])
        {
          for(size_t k = kk; k < kk+ss; k++) {
            for(size_t j = rBWP; j < rY + rBWP; j++) {
              for(size_t i = rBWP; i < rX + rBWP; i++) {
                ApplyBack(aux_3d_1,aux_3d_0,aux_3d_0,i,j,k);
              }
            }
          }
        }
      }

      #pragma omp taskwait

      for(size_t kk = rBWP; kk < rZ + rBWP; kk+=ss) {
        #pragma omp task                                  \
          depend(in:aux_3d_0[(kk)*bt*bt*3:(CFL)*bt*bt*3]) \
          depend(in:aux_3d_1[(kk)*bt*bt*3:(CFL)*bt*bt*3]) \
          depend(out:aux_3d_3[(kk)*bt*bt*3:(CFL)*bt*bt*3])
        {
          for(size_t k = kk; k < kk+ss; k++) {
            for(size_t j = rBWP; j < rY + rBWP; j++) {
              for(size_t i = rBWP; i < rX + rBWP; i++) {
                ApplyForth(aux_3d_3,aux_3d_1,aux_3d_0,i,j,k);
              }
            }
          }
        }
      }

      #pragma omp taskwait

      for(size_t kk = rBWP; kk < rZ + rBWP; kk+=ss) {
        #pragma omp task                                    \
          depend(in:aux_3d_3[(kk)*slice3D:(CFL)*slice3D])   \
          depend(out:aux_3d_1[(kk)*slice3D:(CFL)*slice3D])
        {
          for(size_t k = kk; k < kk+ss; k++) {
            for(size_t j = rBWP; j < rY + rBWP; j++) {
              for(size_t i = rBWP; i < rX + rBWP; i++) {
                ApplyEcc(aux_3d_1,aux_3d_3,i,j,k);
              }
            }
          }
        }
      }

    }

    #pragma omp taskwait

    #undef BOT
    #undef TOP

  }

  /**
   * Executes the solver in parallel using blocking
   **/
  void ExecuteBlock_impl() {

    #define BOT(_i_) std::max(rBWP,(_i_ * rNE))
    #define TOP(_i_) rBWP + std::min(rNE*rNB-rBWP,((_i_+1) * rNE))

    PrecisionType * aux_3d_0 = pBuffers[VELOCITY];
    PrecisionType * aux_3d_1 = pBuffers[AUX_3D_1];
    PrecisionType * aux_3d_3 = pBuffers[AUX_3D_3];

    #pragma omp parallel for
    for(size_t kk = 0; kk < rNB; kk++) {
      for(size_t jj = 0; jj < rNB; jj++) {
        for(size_t ii = 0; ii < rNB; ii++) {
          for(size_t k = BOT(kk); k < TOP(kk); k++) {
            for(size_t j = BOT(jj); j < TOP(jj); j++) {
              for(size_t i = BOT(ii); i < TOP(ii); i++) {
                ApplyBack(aux_3d_1,aux_3d_0,aux_3d_0,i,j,k);
              }
            }
          }
        }
      }
    }

    #pragma omp parallel for
    for(size_t kk = 0; kk < rNB; kk++) {
      for(size_t jj = 0; jj < rNB; jj++) {
        for(size_t ii = 0; ii < rNB; ii++) {
          for(size_t k = BOT(kk); k < TOP(kk); k++) {
            for(size_t j = BOT(jj); j < TOP(jj); j++) {
              for(size_t i = BOT(ii); i < TOP(ii); i++) {
                ApplyForth(aux_3d_3,aux_3d_1,aux_3d_0,i,j,k);
              }
            }
          }
        }
      }
    }

    #pragma omp parallel for
    for(size_t kk = 0; kk < rNB; kk++) {
      for(size_t jj = 0; jj < rNB; jj++) {
        for(size_t ii = 0; ii < rNB; ii++) {
          for(size_t k = BOT(kk); k < TOP(kk); k++) {
            for(size_t j = BOT(jj); j < TOP(jj); j++) {
              for(size_t i = BOT(ii); i < TOP(ii); i++) {
                ApplyEcc(aux_3d_1,aux_3d_3,i,j,k);
              }
            }
          }
        }
      }
    }

    #undef BOT
    #undef TOP

  }

  void ExecuteVector_impl() {}
  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/
  void ApplyBack(
      PrecisionType * Phi,
      PrecisionType * PhiAuxA,
      PrecisionType * PhiAuxB,
      const size_t &i,
      const size_t &j,
      const size_t &k) {

    size_t cell = IndexType::GetIndex(i,j,k,pBlock->mPaddY,pBlock->mPaddZ);

    PrecisionType iPhi[MAX_DIM];
    PrecisionType origin[MAX_DIM];
    PrecisionType displacement[MAX_DIM];

    origin[0] = (PrecisionType)i * rDx;
    origin[1] = (PrecisionType)j * rDx;
    origin[2] = (PrecisionType)k * rDx;

    for(size_t d = 0; d < 3; d++) {
      displacement[d] = origin[d] - pBuffers[VELOCITY][cell*rDim+d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxB,(PrecisionType*)iPhi,displacement,rDim);

    Phi[cell*rDim+0] = iPhi[0];
    Phi[cell*rDim+1] = iPhi[1];
    Phi[cell*rDim+2] = iPhi[2];

    // for(size_t d = 0; d < 3; d++) {
    //   if(Phi[cell*rDim+d] > 1.0f) {
    //     printf("%f -*- %f -*- %f\n",Phi[cell*rDim+d],displacement[d],PhiAuxB[cell*rDim+d]);
    //     abort();
    //   }
    // }
  }

  void ApplyForth(
      PrecisionType * Phi,
      PrecisionType * PhiAuxA,
      PrecisionType * PhiAuxB,
      const size_t &i,
      const size_t &j,
      const size_t &k) {

    size_t cell = IndexType::GetIndex(i,j,k,pBlock->mPaddY,pBlock->mPaddZ);

    PrecisionType iPhi[MAX_DIM];
    PrecisionType origin[MAX_DIM];
    PrecisionType displacement[MAX_DIM];

    origin[0] = (PrecisionType)i * rDx;
    origin[1] = (PrecisionType)j * rDx;
    origin[2] = (PrecisionType)k * rDx;

    for(size_t d = 0; d < 3; d++) {
      displacement[d] = origin[d] + pBuffers[VELOCITY][cell*rDim+d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxA,(PrecisionType*)iPhi,displacement,rDim);

    Phi[cell*rDim+0] = 1.5f * PhiAuxB[cell*rDim+0] - 0.5f * iPhi[0];
    Phi[cell*rDim+1] = 1.5f * PhiAuxB[cell*rDim+1] - 0.5f * iPhi[1];
    Phi[cell*rDim+2] = 1.5f * PhiAuxB[cell*rDim+2] - 0.5f * iPhi[2];
  }

  void ApplyEcc(
      PrecisionType * Phi,
      PrecisionType * PhiAuxA,
      const size_t &i,
      const size_t &j,
      const size_t &k) {

    size_t cell = IndexType::GetIndex(i,j,k,pBlock->mPaddY,pBlock->mPaddZ);

    PrecisionType iPhi[MAX_DIM];
    PrecisionType origin[MAX_DIM];
    PrecisionType displacement[MAX_DIM];

    origin[0] = (PrecisionType)i * rDx;
    origin[1] = (PrecisionType)j * rDx;
    origin[2] = (PrecisionType)k * rDx;

    for(size_t d = 0; d < 3; d++) {
      displacement[d] = origin[d] - pBuffers[VELOCITY][cell*rDim+d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxA,(PrecisionType*)iPhi,displacement,rDim);

    Phi[cell*rDim+0] = iPhi[0];
    Phi[cell*rDim+1] = iPhi[1];
    Phi[cell*rDim+2] = iPhi[2];
  }
};
