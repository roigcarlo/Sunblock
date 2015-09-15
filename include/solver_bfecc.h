#include "solver.h"

class BfeccSolver : public Solver {
public:

  BfeccSolver(Block * block, const PrecisionType& Dt, const PrecisionType& Pdt) :
      Solver(block,Dt,Pdt) {

  }

  ~BfeccSolver() {

  }

  void Prepare() {
  }

  void Finish() {
  }

  /**
   * Executes the solver in parallel
   **/
  void Execute() {

    PrecisionType * aux_3d_0 = pBuffers[VELOCITY];
    PrecisionType * aux_3d_1 = pBuffers[AUX_3D_1];
    PrecisionType * aux_3d_3 = pBuffers[AUX_3D_3];

    size_t * listL = (size_t *)malloc(sizeof(size_t)*rX*rX);
    size_t * listR = (size_t *)malloc(sizeof(size_t)*rX*rX);
    size_t * listF = (size_t *)malloc(sizeof(size_t)*rX*rX);
    size_t * listB = (size_t *)malloc(sizeof(size_t)*rX*rX);
    size_t * listT = (size_t *)malloc(sizeof(size_t)*rX*rX);
    size_t * listD = (size_t *)malloc(sizeof(size_t)*rX*rX);

    int normalL[3] = {0,-1,0};
    int normalR[3] = {0,1,0};
    int normalF[3] = {-1,0,0};
    int normalB[3] = {1,0,0};
    int normalT[3] = {0,0,-1};
    int normalD[3] = {0,0,1};

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

    applyBc(aux_3d_0,listT,rX*rX,normalT,1,3);
    applyBc(aux_3d_0,listD,rX*rX,normalD,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          Apply(aux_3d_1,aux_3d_0,aux_3d_0,-1.0f,0.0f,1.0f,i,j,k);
        }
      }
    }

    applyBc(aux_3d_1,listT,rX*rX,normalT,1,3);
    applyBc(aux_3d_1,listD,rX*rX,normalD,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          Apply(aux_3d_3,aux_3d_0,aux_3d_1,1.0f,1.5f,-0.5f,i,j,k);
        }
      }
    }

    applyBc(aux_3d_3,listT,rX*rX,normalT,1,3);
    applyBc(aux_3d_3,listD,rX*rX,normalD,1,3);

    #pragma omp parallel for
    for(size_t k = rBWP; k < rZ + rBWP; k++) {
      for(size_t j = rBWP; j < rY + rBWP; j++) {
        for(size_t i = rBWP; i < rX + rBWP; i++) {
          Apply(aux_3d_1,aux_3d_0,aux_3d_3,-1.0f,0.0f,1.0f,i,j,k);
        }
      }
    }

    applyBc(aux_3d_1,listT,rX*rX,normalT,1,3);
    applyBc(aux_3d_1,listD,rX*rX,normalD,1,3);

    free(listL);
    free(listR);
    free(listF);
    free(listB);
    free(listT);
    free(listD);
  }

  /**
   * Executes the solver in parallel using blocking
   **/
  void ExecuteBlock() {

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
                Apply(aux_3d_1,aux_3d_0,aux_3d_0,-1.0f,0.0f,1.0f,i,j,k);
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
                Apply(aux_3d_3,aux_3d_0,aux_3d_1,1.0f,1.5f,-0.5f,i,j,k);
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
                Apply(aux_3d_1,aux_3d_0,aux_3d_3,-1.0f,0.0f,1.0f,i,j,k);
              }
            }
          }
        }
      }
    }

    #undef BOT
    #undef TOP

  }

  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/
  void Apply(
      PrecisionType * Phi,
      PrecisionType * PhiAuxA,
      PrecisionType * PhiAuxB,
      const PrecisionType &Sign,
      const PrecisionType &WeightA,
      const PrecisionType &WeightB,
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
      displacement[d] = origin[d] + Sign * pBuffers[VELOCITY][cell*rDim+d] * rDt;
      if(displacement[d] < 0.0f)
        printf(
          "Error: Displacement for component %d: %f ( %f with velocity: %f) is lt 0\n",
          (int)d,
          displacement[d],
          origin[d],
          pBuffers[VELOCITY][cell*rDim+d]*rDt);
    }

    InterpolateType::Interpolate(pBlock,PhiAuxB,(PrecisionType*)iPhi,displacement,rDim);

    // TODO: This provably only needs to be done in the last part. Take into account that
    // if(!(pFlags[cell] & FIXED_VELOCITY_X))
      Phi[cell*rDim+0] = WeightA * PhiAuxA[cell*rDim+0] + WeightB * iPhi[0];
    // if(!(pFlags[cell] & FIXED_VELOCITY_Y))
      Phi[cell*rDim+1] = WeightA * PhiAuxA[cell*rDim+1] + WeightB * iPhi[1];
    // if(!(pFlags[cell] & FIXED_VELOCITY_Z))
      Phi[cell*rDim+2] = WeightA * PhiAuxA[cell*rDim+2] + WeightB * iPhi[2];
  }
};
