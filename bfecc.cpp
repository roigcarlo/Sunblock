#ifndef _WIN32
#include <sys/time.h>
#else
#include <Windows.h>
#endif

#include <sys/types.h>
#include <omp.h>

// Solver
#include "include/utils.h"
#include "include/block.h"
#include "include/defines.h"
#include "include/solver_stencil.h"
#include "include/solver_bfecc.h"
#include "include/file_io.h"
#include "include/interpolator.h"

#define WRITE_INIT_R(_STEP_)                                          \
io.WriteGidMeshBin(N,N,N);                                            \
io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,0,Dim,"TMP");     \
io.WriteGidResultsBin3D((PrecisionType*)step2,N,N,N,0,Dim,"GRD");     \
io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,0,Dim,"VEL");     \
io.WriteGidResultsBin1D((PrecisionType*)pres0,N,N,N,0    ,"PRES");    \

#define WRITE_RESULT(_STEP_)                                          \
if (OutputStep == 0) {                                                \
  io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,i+1,Dim,"TMP"); \
  io.WriteGidResultsBin3D((PrecisionType*)step2,N,N,N,i+1,Dim,"GRD"); \
  io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,i+1,Dim,"VEL"); \
  io.WriteGidResultsBin1D((PrecisionType*)pres0,N,N,N,i+1    ,"PRES");\
  OutputStep = _STEP_;                                                \
}                                                                     \
OutputStep--;                                                         \

PrecisionType calculateMaxDt_CFL(PrecisionType CFL, PrecisionType h, PrecisionType maxv) {
  return CFL*h / maxv;
}

PrecisionType calculateMaxDt_Fourier() {
  return 0.0;
}

PrecisionType calculatePressDt(PrecisionType dt, PrecisionType limit) {
  int base = ceil(dt/limit);
  return dt/base;
}

int main(int argc, char *argv[]) {

#ifndef _WIN32
  struct timeval start, end;
#else
  int start, end;
#endif
  PrecisionType duration = 0.0;

  uint N          = atoi(argv[1]);
  uint steeps     = atoi(argv[2]);
  uint NB         = atoi(argv[4]);
  uint NE         = (N+BW)/NB;
  uint OutputStep = 0;
  uint Dim        = 3;
  uint frec       = steeps/steeps;

  PrecisionType h        = atoi(argv[3]);
  PrecisionType omega    = 1.0f;
  PrecisionType maxv     = 0.0f;
  PrecisionType oldmaxv  = 0.0f;
  PrecisionType CFL      = 0.8f;

  PrecisionType dx       = h/(PrecisionType)N;
  PrecisionType dt       = 0.1f;
  PrecisionType pdt      = 0.1f;

  PrecisionType ro       = 1.0f;
  PrecisionType mu       = 1.93e-5f;
  PrecisionType ka       = 1.0e-5f;
  PrecisionType cc2      = 343.2f*343.2f;

  FileIO io("grid",N);

  Block      * block = NULL;

  PrecisionType * step0 = NULL;
  PrecisionType * step1 = NULL;
  PrecisionType * step2 = NULL;
  PrecisionType * step3 = NULL;

  PrecisionType * pres0 = NULL;
  PrecisionType * pres1 = NULL;

  PrecisionType * velf0 = NULL;

  MemManager memmrg(false);

  // Variable
  memmrg.AllocateGrid(&step0, N, N, N, 3, 1);
  memmrg.AllocateGrid(&step1, N, N, N, 3, 1);
  memmrg.AllocateGrid(&step2, N, N, N, 3, 1);
  memmrg.AllocateGrid(&step3, N, N, N, 3, 1);

  // Pressure
  memmrg.AllocateGrid(&pres0, N, N, N, 1, 1);
  memmrg.AllocateGrid(&pres1, N, N, N, 1, 1);

  // Velocity
  memmrg.AllocateGrid(&velf0, N, N, N, 3, 1);

  printf("Allocation correct\n");
  printf("Initialize\n");

  block = new Block(
    (PrecisionType*) velf0,
    (PrecisionType*) step1,
    (PrecisionType*) step2,
    (PrecisionType*) step3,
    (PrecisionType*) pres0,
    (PrecisionType*) pres1,
    (PrecisionType*) velf0,
    dx,
    omega,
    ro,
    mu,
    ka,
    cc2,
    BW,
    N,
    N,
    N,
    NB,
    NE,
    Dim
  );

  block->Zero();
  block->InitializeVelocity();
  block->InitializePressure();
  // block->WriteHeatFocus();

  block->calculateMaxVelocity(maxv);
  dt = calculateMaxDt_CFL(CFL,dx,maxv);
  pdt = ro/cc2;

  printf("Calculated dt: %f -- %f, %f, %f \n",dt,CFL,h/N,maxv);

  BfeccSolver   AdvectionSolver(block,dt,pdt);
  StencilSolver DiffusionSolver(block,dt,pdt);

  WRITE_INIT_R(frec)

  #pragma omp parallel
  #pragma omp single
  {
    printf("-------------------\n");
    printf("Running with OMP %d\n",omp_get_num_threads());
    printf("-------------------\n");
  }

#ifndef _WIN32
  gettimeofday(&start, NULL);
#else
  start = GetTickCount(); // At Program Start
#endif

  AdvectionSolver.Prepare();

  for (uint i = 0; i < steeps; i++) {

    oldmaxv = maxv;
    block->calculateMaxVelocity(maxv);
    dt = calculateMaxDt_CFL(CFL,dx,maxv);
    pdt = ro/cc2;
    printf("Step %d: %f -- %f, %f, MAXV: %f, [%f,%f] \n",i,calculateMaxDt_CFL(CFL,dx,maxv),CFL,h/N,maxv,(1.0f/64.0f)/dt,(maxv-oldmaxv));

    AdvectionSolver.Execute();
    DiffusionSolver.Execute();
    WRITE_RESULT(frec)
  }

  AdvectionSolver.Finish();

#ifndef _WIN32
  gettimeofday(&end, NULL);
  duration = FETCHTIME(start,end)
#else
  end = GetTickCount();
  duration = (end - start) / 1000.0f;
#endif

  printf("Total time:\t %f s\n",duration);
  printf("Step  time:\t %f s\n",duration/steeps);

  free(block);

  memmrg.ReleaseGrid(&step0, 1);
  memmrg.ReleaseGrid(&step1, 1);
  memmrg.ReleaseGrid(&step2, 1);
  memmrg.ReleaseGrid(&step3, 1);

  memmrg.ReleaseGrid(&pres0, 1);
  memmrg.ReleaseGrid(&pres1, 1);

  memmrg.ReleaseGrid(&velf0, 1);

  printf("De-Allocation correct\n");

  return 0;
}
