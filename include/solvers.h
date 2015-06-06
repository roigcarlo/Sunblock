#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sys/types.h>

#include "defines.h"
#include "utils.h"
#include "block.h"

#define TILE_X 8
#define TILE_Y 4
#define TILE_Z 4
#define TE     2

#define __GETINDEX(_i,_j,_k) (_k)*N*N + (_j)*N + (_i)

void __device__ InterpolateCUDA(double * Coords, double * in, double * out, double dx, int N, int index) {
	int pi, pj, pk, ni, nj, nk;

	for (int d = 0; d < 3; d++)
		Coords[d] *= (1/dx);

	pi = (int)floor(Coords[0]); ni = pi + 1;
	pj = (int)floor(Coords[1]); nj = pj + 1;
	pk = (int)floor(Coords[2]); nk = pk + 1;

	double Nx, Ny, Nz;

	Nx = 1 - (Coords[0] - pi);
	Ny = 1 - (Coords[1] - pj);
	Nz = 1 - (Coords[2] - pk);

	double a = in[__GETINDEX(pi, pj, pk)] * (Nx)* (Ny)* (Nz);
	double b = in[__GETINDEX(ni, pj, pk)] * (1 - Nx) * (Ny)* (Nz);
	double c = in[__GETINDEX(pi, nj, pk)] * (Nx)* (1 - Ny) * (Nz);
	double d = in[__GETINDEX(ni, nj, pk)] * (1 - Nx) * (1 - Ny) * (Nz);
	double e = in[__GETINDEX(pi, pj, nk)] * (Nx)* (Ny)* (1 - Nz);
	double f = in[__GETINDEX(ni, pj, nk)] * (1 - Nx) * (Ny)* (1 - Nz);
	double g = in[__GETINDEX(pi, nj, nk)] * (Nx)* (1 - Ny) * (1 - Nz);
	double h = in[__GETINDEX(ni, nj, nk)] * (1 - Nx) * (1 - Ny) * (1 - Nz);

	out[0] = (a + b + c + d + e + f + g + h);
}

__global__ void ApplyCUDA(
	double * out, 
	double * PhiAuxA, double * PhiAuxB, double * vel,
	const double Sign, const double WeightA, const double WeightB, 
	const double dx, const double dt, const int N,
	const int ii, const int jj, const int kk) {

	int i = blockIdx.x * blockDim.x + threadIdx.x + ii;
	int j = blockIdx.y * blockDim.y + threadIdx.y + jj;
	int k = blockIdx.z * blockDim.z + threadIdx.z + kk;

	int index = k*N*N + j*N + i;

	double org[3], dsp[3], itp[1];

	if (i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1) {
		org[0] = i * dx;
		org[1] = j * dx;
		org[2] = k * dx;

		for (int d = 0; d < 3; d++) {
			dsp[d] = org[d] + Sign * vel[index*3+d] * dt;
		}

		InterpolateCUDA(dsp, PhiAuxB, itp, dx, N, index);

		out[index] = WeightA * PhiAuxA[index] + WeightB * itp[0];
	}
}

void __device__ InterpolateSharedCUDA(double * Coords, double sh_in[TILE_X + TE][TILE_Y + TE][TILE_Z + TE], double * out, double dx, int N, int index) {
	int pi, pj, pk, ni, nj, nk;

	for (int d = 0; d < 3; d++)
		Coords[d] *= (1 / dx);

	pi = ((int)floor(Coords[0])); ni = pi + 1;
	pj = ((int)floor(Coords[1])); nj = pj + 1;
	pk = ((int)floor(Coords[2])); nk = pk + 1;

	double Nx, Ny, Nz;

	Nx = 1 - (Coords[0] - pi);
	Ny = 1 - (Coords[1] - pj);
	Nz = 1 - (Coords[2] - pk);

	pi = ((int)floor(Coords[0]) % TILE_X) + 1; ni = pi + 1;
	pj = ((int)floor(Coords[1]) % TILE_Y) + 1; nj = pj + 1;
	pk = ((int)floor(Coords[2]) % TILE_Z) + 1; nk = pk + 1;

	double a = sh_in[pi][pj][pk] * (Nx)* (Ny)* (Nz);
	double b = sh_in[ni][pj][pk] * (1 - Nx) * (Ny)* (Nz);
	double c = sh_in[pi][nj][pk] * (Nx)* (1 - Ny) * (Nz);
	double d = sh_in[ni][nj][pk] * (1 - Nx) * (1 - Ny) * (Nz);
	double e = sh_in[pi][pj][nk] * (Nx)* (Ny)* (1 - Nz);
	double f = sh_in[ni][pj][nk] * (1 - Nx) * (Ny)* (1 - Nz);
	double g = sh_in[pi][nj][nk] * (Nx)* (1 - Ny) * (1 - Nz);
	double h = sh_in[ni][nj][nk] * (1 - Nx) * (1 - Ny) * (1 - Nz);

	out[0] = (a + b + c + d + e + f + g + h);
}

__global__ void ApplySharedCUDA(
	double * out,
	double * PhiAuxA, double * PhiAuxB, double * vel,
	const double Sign, const double WeightA, const double WeightB,
	const double dx, const double dt, const int N,
	const int ii, const int jj, const int kk) {

	int i = blockIdx.x * blockDim.x + threadIdx.x + ii;
	int j = blockIdx.y * blockDim.y + threadIdx.y + jj;
	int k = blockIdx.z * blockDim.z + threadIdx.z + kk;

	double org[3], dsp[3], itp[1];

	__shared__ double sh_in[TILE_X + TE][TILE_Y + TE][TILE_Z + TE];

	// sh_in[0][threadIdx.x][threadIdx.y][threadIdx.z] = PhiAuxA[__GETINDEX(i,j,k)];
	sh_in[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 1] = PhiAuxB[__GETINDEX(i, j, k)];  // Current
	
	sh_in[threadIdx.x + 0][threadIdx.y + 1][threadIdx.z + 1] = PhiAuxB[__GETINDEX(max(0    , i - 1), j, k)];  // Left
	sh_in[threadIdx.x + 2][threadIdx.y + 1][threadIdx.z + 1] = PhiAuxB[__GETINDEX(min(N - 1, i + 1), j, k)];  // Right
	sh_in[threadIdx.x + 1][threadIdx.y + 0][threadIdx.z + 1] = PhiAuxB[__GETINDEX(i, max(0    , j - 1), k)];  // Top
	sh_in[threadIdx.x + 1][threadIdx.y + 2][threadIdx.z + 1] = PhiAuxB[__GETINDEX(i, min(N - 1, j + 1), k)];  // Bott
	sh_in[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 0] = PhiAuxB[__GETINDEX(i, j, max(0    , k - 1))];  // Front
	sh_in[threadIdx.x + 1][threadIdx.y + 1][threadIdx.z + 2] = PhiAuxB[__GETINDEX(i, j, max(N - 1, k + 1))];  // Back
	
	__syncthreads();

	double tmp;

	if (i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1) {
		org[0] = i * dx; org[1] = j * dx; org[2] = k * dx;

		for (int d = 0; d < 3; d++) {
			dsp[d] = org[d] + Sign * vel[__GETINDEX(i, j, k) * 3 + d] * dt;
		}

		InterpolateSharedCUDA(dsp, sh_in, itp, dx, N, __GETINDEX(i, j, k));

		tmp = WeightA * PhiAuxA[__GETINDEX(i, j, k)] + WeightB * itp[0];
	}

	__syncthreads();

	out[__GETINDEX(i, j, k)] = tmp;
}

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

    #pragma omp barrier

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
      for(uint j = rBWP; j < rY + rBWP; j++)
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiC,pPhiA,pPhiB,1.0,1.5,-0.5,i,j,k);
        }

    #pragma omp barrier

    for(uint k = rBWP + tid; k < rZ + rBWP; k+= tsize)
      for(uint j = rBWP; j < rY + rBWP; j++)
        for(uint i = rBWP; i < rX + rBWP; i++) {
          Apply(pPhiA,pPhiA,pPhiC,-1.0,0.0,1.0,i,j,k);
        }
  }


  virtual void PrepareCUDA() {

	  int num_bytes = (rX + rBW) * (rY + rBW) * (rZ + rBW);

	  cudaMalloc((void**)&d_PhiA, num_bytes * sizeof(double));
	  cudaMalloc((void**)&d_PhiB, num_bytes * sizeof(double));
	  cudaMalloc((void**)&d_PhiC, num_bytes * sizeof(double));
	  cudaMalloc((void**)&d_vel, num_bytes * sizeof(double) * 3);
  }

  virtual void FinishCUDA() {
	  cudaFree(d_PhiA);
	  cudaFree(d_PhiB);
	  cudaFree(d_PhiC);
	  cudaFree(d_vel);
  }

  /**
  * Executes the solver with CUDA
  **/
  virtual void ExecuteCUDA() {

	int num_bytes = (rX + rBW) * (rY + rBW) * (rZ + rBW);

	cudaError err = cudaGetLastError();

	int BBK = 8;
	int ELE = (rX + rBW) / BBK;

	dim3 threads(TILE_X, TILE_Y, TILE_Z);
	dim3 blocks(ELE / TILE_X, ELE / TILE_Y, ELE / TILE_Z);

	cudaMemset(d_PhiB, 0, num_bytes * sizeof(double));
	cudaMemset(d_PhiC, 0, num_bytes * sizeof(double));

	cudaMemcpy(d_PhiA, pPhiA, num_bytes * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, pVelocity, num_bytes * sizeof(double) * 3, cudaMemcpyHostToDevice);

	// printf("Executing with CUDA... BLOCKS: %d, THREADS: %d\n", blocks.x, threads.x);

	cudaFuncSetCacheConfig(ApplySharedCUDA, cudaFuncCachePreferShared);

	for (int i = 0; i < BBK; i++)
	  for (int j = 0; j < BBK; j++)
		for (int k = 0; k < BBK; k++) {
		  ApplyCUDA<<< threads, blocks >>>(d_PhiB, d_PhiA, d_PhiA, d_vel, -1.0, 0.0, 1.0, rDx, rDt, rX + rBW, i * ELE, j * ELE , k * ELE);
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < BBK; i++)
	  for (int j = 0; j < BBK; j++)
		for (int k = 0; k < BBK; k++) {
		  ApplyCUDA<<< threads, blocks >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, 1.0, 1.5, -0.5, rDx, rDt, rX + rBW, i * ELE, j * ELE , k * ELE);
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < BBK; i++)
	  for (int j = 0; j < BBK; j++)
		for (int k = 0; k < BBK; k++) {
		  ApplyCUDA<<< threads, blocks >>>(d_PhiA, d_PhiA, d_PhiC, d_vel, -1.0, 0.0, 1.0, rDx, rDt, rX + rBW, i * ELE, j * ELE , k * ELE);
	}

	// ApplyCUDA<<< threads, blocks >>>(d_PhiB, d_PhiA, d_PhiA, d_vel, -1.0, 0.0, 1.0, rDx, rDt, rX+rBW);
	// ApplyCUDA<<< threads, blocks >>>(d_PhiC, d_PhiA, d_PhiB, d_vel, 1.0, 1.5, -0.5, rDx, rDt, rX+rBW);
	// ApplyCUDA<<< threads, blocks >>>(d_PhiA, d_PhiA, d_PhiC, d_vel, -1.0, 0.0, 1.0, rDx, rDt, rX+rBW);
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(pPhiA, d_PhiA, num_bytes * sizeof(double), cudaMemcpyDeviceToHost);

	// printf("Finish! printing sample pPhiA array... %f\n", pPhiA[0]);
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

  double * d_PhiA = 0;
  double * d_PhiB = 0;
  double * d_PhiC = 0;
  double * d_vel = 0;

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