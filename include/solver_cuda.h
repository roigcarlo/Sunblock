#include "solver.h"

#define TILE_X 16
#define TILE_Y 16
#define TILE_Z 4
#define TE     2

#define BBX 1
#define BBY 1
#define BBZ 16

#ifdef USE_CUDA
class BfeccSolverCUDA : public Solver {
public:

  BfeccSolverCUDA(BlockType * block, const double& Dt) : 
      Solver(block,Dt) {

  }

  ~BfeccSolverCUDA() {
  }

  void Prepare() {

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

  void Finish() {

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
  void ExecuteCUDA() {

    dim3 threads(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks(ELX / TILE_X, ELY / TILE_Y, ELZ / TILE_Z);

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

      BackCUDA <<< threads, blocks, 0, dstream1[c] >>>(d_PhiB, d_PhiA, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-1) * ELZ);
      if (c > 1) {
        ForthCUDA<<< threads, blocks, 0, dstream1[c] >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-2) * ELZ);
        if (c > 2) {
          EccCUDA <<< threads, blocks, 0, dstream1[c] >>>(d_PhiA, d_PhiC, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, (c-3) * ELZ);
        }
      }
    }

    for (int c = 0; c < BBZ - 3; c++) {
      cudaMemcpyAsync(itr_phi_res, d_itr_phi_res, chunk_size * sizeof(double), cudaMemcpyDeviceToHost, dstream1[3 + c]);
      itr_phi_res += chunk_size;
      d_itr_phi_res += chunk_size;
    }

    for (int c = BBZ - 1; c < BBZ; c++) {
      BackCUDA <<< threads, blocks, 0, dstream1[(BBZ)] >>>(d_PhiB, d_PhiA, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
    }

    for (int c = BBZ - 2; c < BBZ; c++) {
      ForthCUDA<<< threads, blocks, 0, dstream1[(BBZ)] >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
    }

    // This needs to be implemented with events, since extrange rance conditions are present
    for (int c = 0; c < 3; c++) {
      cudaEventCreate(&trail_eec[c]);
    }
    
    for (int c = 1; c <= 3; c++) {

      if (c > 1)
        cudaStreamWaitEvent(dstream1[(BBZ - 3 + c - 2)], trail_eec[c - 2], 0);

      EccCUDA  <<< threads, blocks, 0, dstream1[(BBZ - 3 + c)] >>>(d_PhiA, d_PhiC, d_vel, rDx, rIdx, rDt, rX + rBW, 0 * ELX, 0 * ELY, c * ELZ);
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
  void ExecuteSimpleCUDA() {

    dim3 threads(TILE_X, TILE_Y, TILE_Z);
    dim3 blocks(rX / TILE_X, rY / TILE_Y, rZ / TILE_Z);

    cudaMemcpy(d_vel , pVelocity, num_bytes * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_PhiA, pPhiA    , num_bytes * sizeof(double)    , cudaMemcpyHostToDevice);

    for (int i = 0; i < BBX; i++)
      for (int j = 0; j < BBY; j++)
        for (int k = 0; k < BBZ; k++)
          BackCUDA <<< threads, blocks >>>(d_PhiB, d_PhiA, d_vel, rDx, 1.0f/rDx, rDt, rX + rBW, i * ELX, j * ELY, k * ELZ);

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

private:

  int num_bytes;

  int ELX;
  int ELY;
  int ELZ;

  double * d_PhiA;
  double * d_PhiB;
  double * d_PhiC;
  double * d_vel;

  cudaStream_t dstream1[BBZ + 2];
  cudaEvent_t trail_eec[3];
   
};
#endif