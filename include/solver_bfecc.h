#include "solver.h"

class BfeccSolver : public Solver {
public:

  BfeccSolver(Block * block, const double& Dt) :
      Solver(block,Dt) {

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

    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiB,pPhiA,pPhiA,-1.0f,0.0f,1.0f,i,j,k);
        }
      }
    }

    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiC,pPhiA,pPhiB,1.0f,1.5f,-0.5f,i,j,k);
        }
      }
    }

    #pragma omp parallel for
    for(uint k = rBWP; k < rZ + rBWP; k++) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiD,pPhiA,pPhiC,-1.0f,0.0f,1.0f,i,j,k);
        }
      }
    }

  }

  /**
   * Executes the solver in parallel using blocking
   **/
  void ExecuteBlock() {

    #define BOT(_i_) std::max(rBWP,(_i_ * rNE))
    #define TOP(_i_) rBWP + std::min(rNE*rNB-rBWP,((_i_+1) * rNE))

    #pragma omp parallel for
    for(uint kk = 0; kk < rNB; kk++) {
      for(uint jj = 0; jj < rNB; jj++) {
        for(uint ii = 0; ii < rNB; ii++) {
          for(uint k = BOT(kk); k < TOP(kk); k++) {
            for(uint j = BOT(jj); j < TOP(jj); j++) {
              for(uint i = BOT(ii); i < TOP(ii); i++) {
                Apply(pPhiB,pPhiA,pPhiA,-1.0f,0.0f,1.0f,i,j,k);
              }
            }
          }
        }
      }
    }

    #pragma omp parallel for
    for(uint kk = 0; kk < rNB; kk++) {
      for(uint jj = 0; jj < rNB; jj++) {
        for(uint ii = 0; ii < rNB; ii++) {
          for(uint k = BOT(kk); k < TOP(kk); k++) {
            for(uint j = BOT(jj); j < TOP(jj); j++) {
              for(uint i = BOT(ii); i < TOP(ii); i++) {
                Apply(pPhiC,pPhiA,pPhiB,1.0f,1.5f,-0.5f,i,j,k);
              }
            }
          }
        }
      }
    }

    #pragma omp parallel for
    for(uint kk = 0; kk < rNB; kk++) {
      for(uint jj = 0; jj < rNB; jj++) {
        for(uint ii = 0; ii < rNB; ii++) {
          for(uint k = BOT(kk); k < TOP(kk); k++) {
            for(uint j = BOT(jj); j < TOP(jj); j++) {
              for(uint i = BOT(ii); i < TOP(ii); i++) {
                Apply(pPhiD,pPhiA,pPhiC,-1.0f,0.0f,1.0f,i,j,k);
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
      const double &Sign,
      const double &WeightA,
      const double &WeightB,
      const uint &i,
      const uint &j,
      const uint &k) {

    uint cell = IndexType::GetIndex(i,j,k,pBlock->mPaddY,pBlock->mPaddZ);

    PrecisionType iPhi[MAX_DIM];
    PrecisionType origin[MAX_DIM];
    PrecisionType displacement[MAX_DIM];

    origin[0] = i * rDx;
    origin[1] = j * rDx;
    origin[2] = k * rDx;

    for(uint d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * pVelocity[cell*rDim+d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxB,(PrecisionType*)iPhi,displacement,rDim);

    for(uint d = 0; d < rDim; d++) {
      Phi[cell*rDim+d] = WeightA * PhiAuxA[cell*rDim+d] + WeightB * iPhi[d];
    }
  }
};
