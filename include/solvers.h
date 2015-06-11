#include <sys/types.h>

#include "defines.h"
#include "utils.h"
#include "block.h"
#include "kernels.h"

#define TILE_X 16
#define TILE_Y 16
#define TILE_Z 4
#define TE     2

#define BBX 1
#define BBY 1
#define BBZ 16

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
  virtual void ExecuteBlock(){}
  virtual void ExecuteCUDAf(){}

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
   * Executes the solver using CPU resources without blocking
   **/
  virtual void Execute() {

    uint tid;
    uint tsize;

    #pragma omp parallel
    {
      tid   = omp_get_thread_num();
      tsize = omp_get_num_threads();

      for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
        for(uint j = rBWP; j < rY + rBWP; j++)
          for(uint i = rBWP; i < rX + rBWP; i++)
            Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k);

      #pragma omp barrier

      for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
        for(uint j = rBWP; j < rY + rBWP; j++)
          for(uint i = rBWP; i < rX + rBWP; i++)
            Apply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k);

      #pragma omp barrier

      for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
        for(uint j = rBWP; j < rY + rBWP; j++)
          for(uint i = rBWP; i < rX + rBWP; i++)
            Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k);
    }
  }

  /**
   * Executes the solver using blocking
   **/
  virtual void ExecuteBlock() {

    uint tid   = omp_get_thread_num();
    uint tsize = omp_get_num_threads();

    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiB,pPhiA,pPhiA,-1.0,0.0,1.0,i,j,k);

    #pragma omp barrier

    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k);

    #pragma omp barrier
   
    for(uint kk = 0 + tid; kk < rNB; kk+= tsize)
      for(uint jj = 0; jj < rNB; jj++)
        for(uint ii = 0; ii < rNB; ii++)
          for(uint k = std::max(rBWP,(kk * rNE)); k < std::min(rNE*rNB-rBWP,((kk+1) * rNE)); k++)
            for(uint j = std::max(rBWP,(jj * rNE)); j < rBWP + std::min(rNE*rNB-rBWP,((jj+1) * rNE)); j++)
              for(uint i = std::max(rBWP,(ii * rNE)); i < rBWP + std::min(rNE*rNB-rBWP,((ii+1) * rNE)); i++)
                Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k);
  }


  virtual void PrepareCUDA() {

    num_bytes = (rX + rBW) * (rY + rBW) * (rZ + rBW);

    ELX = (rX + rBW) / BBX;
    ELY = (rX + rBW) / BBY;
    ELZ = (rX + rBW) / BBZ;

    cudaMalloc((void**)&d_PhiA, num_bytes * sizeof(double));
    cudaMalloc((void**)&d_PhiB, num_bytes * sizeof(double));
    cudaMalloc((void**)&d_PhiC, num_bytes * sizeof(double));
    cudaMalloc((void**)&d_vel,  num_bytes * sizeof(double) * 3);

    for (int c = 0; c < BBZ + 2; c++) {
      cudaStreamCreate(&dstream1[c]);
    }

    cudaMemset(d_PhiB, 0, num_bytes * sizeof(double));
    cudaMemset(d_PhiC, 0, num_bytes * sizeof(double));
  }

  virtual void FinishCUDA() {

    for (int c = 0; c < BBZ + 2; c++) {
      cudaStreamDestroy(dstream1[c]);
    }

    cudaFree(d_PhiA);
    cudaFree(d_PhiB);
    cudaFree(d_PhiC);
    cudaFree(d_vel);

    for (int c = 0; c < 3; c++) {
      cudaEventDestroy(trail_eec[c]);
    }

    cudaDeviceReset();
  }

  /**
  * Executes the solver with CUDA
  **/
  virtual void ExecuteCUDA() {

    dim3 threads(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks(ELX / TILE_X, ELY / TILE_Y, ELZ / TILE_Z);

    dim3 threads_full(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks_full((rX + rBW) / TILE_X, (rX + rBW) / TILE_Y, (rX + rBW) / TILE_Z);

    int chunk_size = num_bytes / BBZ;

    ResultType     * itr_phi = pPhiA;
    double         * d_itr_phi = d_PhiA;

    Variable3DType * itr_vel = pVelocity;
    double         * d_itr_vel = d_vel;

    ResultType     * itr_phi_res = pPhiA;
    double         * d_itr_phi_res = d_PhiA;

    cudaMemcpyAsync(d_itr_vel, itr_vel, chunk_size * sizeof(double) * 3, cudaMemcpyHostToDevice, dstream1[0]);
    cudaMemcpyAsync(d_itr_phi, itr_phi, chunk_size * sizeof(double)    , cudaMemcpyHostToDevice, dstream1[0]);

    for (int c = 1; c < BBZ; c++) {

      itr_phi   += chunk_size;
      d_itr_phi += chunk_size;
      itr_vel   += chunk_size;
      d_itr_vel += (chunk_size * 3);

      cudaMemcpyAsync(d_itr_vel, itr_vel, chunk_size * sizeof(double) * 3, cudaMemcpyHostToDevice, dstream1[c]);
      cudaMemcpyAsync(d_itr_phi, itr_phi, chunk_size * sizeof(double), cudaMemcpyHostToDevice, dstream1[c]);

      BackCUDA <<< threads, blocks, 0, dstream1[c] >>>(d_PhiB, d_PhiA, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-1) * ELZ);
      if (c > 1) {
        ForthCUDA<<< threads, blocks, 0, dstream1[c] >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-2) * ELZ);
        if (c > 2) {
          EccCUDA <<< threads, blocks, 0, dstream1[c] >>>(d_PhiA, d_PhiC, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-3) * ELZ);
        }
      }
    }

    for (int c = 0; c < BBZ - 3; c++) {
      cudaMemcpyAsync(itr_phi_res, d_itr_phi_res, chunk_size * sizeof(double), cudaMemcpyDeviceToHost, dstream1[3 + c]);
      itr_phi_res += chunk_size;
      d_itr_phi_res += chunk_size;
    }

    for (int c = BBZ - 1; c < BBZ; c++) {
      BackCUDA <<< threads, blocks, 0, dstream1[(BBZ)] >>>(d_PhiB, d_PhiA, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
    }

    for (int c = BBZ - 2; c < BBZ; c++) {
      ForthCUDA<<< threads, blocks, 0, dstream1[(BBZ)] >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
    }

    // This needs to be implemented with events, since extrange rance conditions are present
    for (int c = 0; c < 3; c++) {
      cudaEventCreate(&trail_eec[c]);
    }
    
    for (int c = 1; c <= 3; c++) {

      if (c > 1)
        cudaStreamWaitEvent(dstream1[(BBZ - 3 + c - 2)], trail_eec[c - 2], 0);

      EccCUDA  <<< threads, blocks, 0, dstream1[(BBZ - 3 + c)] >>>(d_PhiA, d_PhiC, d_vel, rDx, 1/rDx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
      cudaEventRecord(trail_eec[c - 1], dstream1[(BBZ - 3 + c)]);
      cudaMemcpyAsync(itr_phi_res, d_itr_phi_res, chunk_size * sizeof(double), cudaMemcpyDeviceToHost, dstream1[(BBZ - 3 + c)]);
      itr_phi_res += chunk_size;
      d_itr_phi_res += chunk_size;
    }

    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize(); // Not sure if this is necessary
  }

  /**
  * Executes the solver with CUDA
  **/
  virtual void ExecuteSimpleCUDA() {

    cudaMemcpy(d_vel , pVelocity, num_bytes * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_PhiA, pPhiA    , num_bytes * sizeof(double)    , cudaMemcpyHostToDevice);

    for (int i = 0; i < BBX; i++)
      for (int j = 0; j < BBY; j++)
        for (int k = 0; k < BBZ; k++)
          BackCUDA <<< threads, blocks >>>(d_PhiB, d_PhiA, d_vel, rDx, rDt, rX + rBW, i * ELX, j * ELY, k * ELZ);

    for (int i = 0; i < BBX; i++)
      for (int j = 0; j < BBY; j++)
        for (int k = 0; k < BBZ; k++)
          ForthCUDA<<< threads, blocks >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, rDx, rDt, rX + rBW, i * ELX, j * ELY, k * ELZ);

    for (int i = 0; i < BBX; i++)
      for (int j = 0; j < BBY; j++)
        for (int k = 0; k < BBZ; k++)
          EccCUDA  <<< threads, blocks >>>(d_PhiA, d_PhiC, d_vel, rDx, rDt, rX + rBW, i * ELX, j * ELY, k * ELZ);

    cudaMemcpy(pPhiA , d_PhiA   , num_bytes * sizeof(double)    , cudaMemcpyDeviceToHost);
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

  int num_bytes;

  int ELX;
  int ELY;
  int ELZ;

  double * pFactors;

  BlockType * pBlock;

  ResultType * pPhiA;
  ResultType * pPhiB;
  ResultType * pPhiC;

  double * d_PhiA = 0;
  double * d_PhiB = 0;
  double * d_PhiC = 0;
  double * d_vel = 0;

  cudaStream_t dstream1[BBZ + 2];
  cudaEvent_t trail_eec[3];

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