#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __GETINDEX(_i,_j,_k) ((_k)*N*N + (_j)*N + (_i))

void __device__ InterpolateCUDA(
    double * Coords, double * in, double * out, 
    const double &idx, const int &N) {

  int pi, pj, pk, ni, nj, nk;

  for (int d = 0; d < 3; d++)
    Coords[d] *= (idx);

  pi = (int)floor(Coords[0]); ni = pi + 1;
  pj = (int)floor(Coords[1]); nj = pj + 1;
  pk = (int)floor(Coords[2]); nk = pk + 1;

  double Nx, Ny, Nz;

  Nx = ni - Coords[0];
  Ny = nj - Coords[1];
  Nz = nk - Coords[2];

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

__global__ void BackCUDA(
    double * out, 
    double * PhiAux, double * vel,
    const double dx, const double idx, const double dt, const int N,
    const int ii, const int jj, const int kk) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + ii;
  int j = blockIdx.y * blockDim.y + threadIdx.y + jj;
  int k = blockIdx.z * blockDim.z + threadIdx.z + kk;

  double dsp[3];

  if (i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1) {

    dsp[0] = fma((double)i, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 0] * dt));
    dsp[1] = fma((double)j, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 1] * dt));
    dsp[2] = fma((double)k, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 2] * dt));

    InterpolateCUDA(dsp, PhiAux, &out[__GETINDEX(i, j, k)], idx, N);
  }
}

__global__ void ForthCUDA(
    double * out,
    double * PhiAuxA, double * PhiAuxB, double * vel,
    const double dx, const double idx, const double dt, const int N,
    const int ii, const int jj, const int kk) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + ii;
  int j = blockIdx.y * blockDim.y + threadIdx.y + jj;
  int k = blockIdx.z * blockDim.z + threadIdx.z + kk;

  double dsp[3], itp;

  if (i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1) {

    dsp[0] = fma((double)i, (double)dx, (double)(vel[__GETINDEX(i, j, k) * 3 + 0] * dt));
    dsp[1] = fma((double)j, (double)dx, (double)(vel[__GETINDEX(i, j, k) * 3 + 1] * dt));
    dsp[2] = fma((double)k, (double)dx, (double)(vel[__GETINDEX(i, j, k) * 3 + 2] * dt));

    InterpolateCUDA(dsp, PhiAuxB, &itp, idx, N);

    out[__GETINDEX(i, j, k)] = 1.5 * PhiAuxA[__GETINDEX(i, j, k)] - 0.5 * itp;
  }
}

__global__ void EccCUDA(
    double * out,
    double * PhiAux, double * vel,
    const double dx, const double idx, const double dt, const int N,
    const int ii, const int jj, const int kk) {

  int i = blockIdx.x * blockDim.x + threadIdx.x + ii;
  int j = blockIdx.y * blockDim.y + threadIdx.y + jj;
  int k = blockIdx.z * blockDim.z + threadIdx.z + kk;

  double dsp[3];

  if (i > 0 && j > 0 && k > 0 && i < N - 1 && j < N - 1 && k < N - 1) {

    dsp[0] = fma((double)i, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 0] * dt));
    dsp[1] = fma((double)j, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 1] * dt));
    dsp[2] = fma((double)k, (double)dx, (double)(-vel[__GETINDEX(i, j, k) * 3 + 2] * dt));

    InterpolateCUDA(dsp, PhiAux, &out[__GETINDEX(i, j, k)], idx, N);
  }
}

#endif