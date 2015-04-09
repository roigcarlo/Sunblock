#include <sys/types.h>

#include "defines.h"
#include "utils.h"
#include "block.h"

template <
  typename ResultType, 
  typename IndexType,
  typename BlockType,
  typename InterpolateType
  >
class Solver {
public:
  Solver(){}
  ~Solver(){}

  virtual void Execute(){}

};

template <
  typename ResultType, 
  typename IndexType,
  typename BlockType,
  typename InterpolateType
  >
class BfeccSolver : public Solver<ResultType,IndexType,BlockType,InterpolateType> {
public:

  BfeccSolver(BlockType * block) :
    Solver<ResultType,IndexType,BlockType,InterpolateType>(),
    pBlock(block),
    pPhiA(block->pPhiA),
    pPhiB(block->pPhiB), 
    pPhiC(block->pPhiC),
    pVelocity(block->pVelocity),
    rDx(block->rDx),
    rIdx(1.0/block->rDx),
    rDt(block->rDt),
    rBW(block->rBW),
    rBWP(block->rBW/2),
    rX(block->rX),
    rY(block->rY),
    rZ(block->rZ),
    rNB(block->rNB),
    rNE(block->rNE) {

      pFactors = (double *)malloc(sizeof(double) * (rX+rBW) * (rY+rBW) * (rZ+rBW) * 8);

    }

  ~BfeccSolver() {

    free(pFactors);
  }

  /**
   * Executes the solver 
   **/
  virtual void Execute() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    // Preinterpolation();

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
      for(uint j = rBWP; j < rY + rBWP; j++)
        for(uint i = rBWP; i < rX + rBWP; i++)
          Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k);
          // Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);

    #pragma omp barrier

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
      for(uint j = rBWP; j < rY + rBWP; j++)
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k);
          // ReverseApply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);
        }

    #pragma omp barrier

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
      for(uint j = rBWP; j < rY + rBWP; j++)
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k);
          // Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);
        }
  }

  /**
   * Executes the solver using blocking
   **/
  virtual void ExecuteBlock() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    // Preinterpolation();

    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k);
                // Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);

    #pragma omp barrier

    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k);
                // ReverseApply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);

    #pragma omp barrier
   
    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k);
                // Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k,&pFactors[IndexType::GetIndex(pBlock,i,j,k)*8]);
  }

  void Preinterpolation() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize) {
      for(uint j = rBWP; j < rY + rBWP; j++) {
        for(uint i = rBWP; i < rX + rBWP; i++) {

          uint cell = IndexType::GetIndex(pBlock,i,j,k);

          Variable3DType  origin;
          Variable3DType  displacement;

          origin[0] = i * rDx;
          origin[1] = j * rDx;
          origin[2] = k * rDx;

          for(int d = 0; d < 3; d++) {
            displacement[d] = origin[d] - pVelocity[cell][d] * rDt;
          }

          InterpolateType::CalculateFactors(pBlock,displacement,&pFactors[cell*8]);
        }
      }
    }
  }

  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/ 
  void Apply(VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      const double &Sign, const double &WeightA, const double &WeightB,
      const uint &i, const uint &j, const uint &k, double * Factors) {

    uint cell = IndexType::GetIndex(pBlock,i,j,k);
    
    VariableType    iPhi;
    Variable3DType  origin;
    Variable3DType  displacement;

    origin[0] = i * rDx;
    origin[1] = j * rDx;
    origin[2] = k * rDx;

    for(int d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * pVelocity[cell][d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxB,&iPhi,displacement,Factors);

    Phi[cell] = WeightA * PhiAuxA[cell] + WeightB * iPhi;
  }

  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/ 
  void ReverseApply(VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      const double &Sign, const double &WeightA, const double &WeightB,
      const uint &i, const uint &j, const uint &k, double * Factors) {

    uint cell = IndexType::GetIndex(pBlock,i,j,k);
    
    VariableType    iPhi;
    Variable3DType  origin;
    Variable3DType  displacement;

    origin[0] = i * rDx;
    origin[1] = j * rDx;
    origin[2] = k * rDx;

    for(int d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * pVelocity[cell][d] * rDt;
    }

    InterpolateType::ReverseInterpolate(pBlock,PhiAuxB,&iPhi,displacement,Factors);

    Phi[cell] = WeightA * PhiAuxA[cell] + WeightB * iPhi;
  }


  /**
   * Performs the bfecc operation over a given element
   * sign:    direction of the interpolation ( -1.0 backward, 1.0 forward )
   * weightA: weigth of the first  operator (A)
   * weightB: weigth of the second operator (B)
   * @i,j,k:  Index of the cell
   **/ 
  void Apply(VariableType * Phi, VariableType * PhiAuxA, VariableType * PhiAuxB,
      const double &Sign, const double &WeightA, const double &WeightB,
      const uint &i, const uint &j, const uint &k) {

    uint cell = IndexType::GetIndex(pBlock,i,j,k);
    
    VariableType    iPhi;
    Variable3DType  origin;
    Variable3DType  displacement;

    origin[0] = i * rDx;
    origin[1] = j * rDx;
    origin[2] = k * rDx;

    for(int d = 0; d < 3; d++) {
      displacement[d] = origin[d] + Sign * pVelocity[cell][d] * rDt;
    }

    InterpolateType::Interpolate(pBlock,PhiAuxB,&iPhi,displacement);

    Phi[cell] = WeightA * PhiAuxA[cell] + WeightB * iPhi;
  }

private:

  double * pFactors;

  BlockType * pBlock;

  ResultType * pPhiA;
  ResultType * pPhiB;
  ResultType * pPhiC;

  Variable3DType * pVelocity;

  const double & rDx;
  const double rIdx;
  const double & rDt;

  const uint & rBW;
  const uint rBWP;

  const uint & rX; 
  const uint & rY; 
  const uint & rZ;

  const uint & rNB;
  const uint & rNE;
   
};

// This does not belong here! put it in a class
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