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

uint N          =  0;
uint NB         =  0;
uint NE         =  0;
uint OutputStep =  0;
uint Dim        =  0;

double dx       =  0.0f;
double idx      =  0.0f;
double dt       =  0.1f;
double pdt      =  0.1f;
double h        = 16.0f;
double omega    =  1.0f;
double maxv     =  0.0f;
double oldmaxv  =  0.0f;
double CFL      =  0.8f;
double cellSize =  1.0f;
double diffTerm =  1.0e-5f;

double ro       =  1.0f;
double mu       =  1.93e-5;
double ka       =  1.0e-5;

#define WRITE_INIT_R(_STEP_)                                          \
io.WriteGidMeshBin(N,N,N);                                    \
io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,0,Dim,"TMP");     \
io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,0,Dim,"VEL");     \
io.WriteGidResultsBin1D((PrecisionType*)pres0,N,N,N,0    ,"PRES");    \
OutputStep = _STEP_;                                                  \

#define WRITE_RESULT(_STEP_)                                          \
if (OutputStep == 0) {                                                \
  io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,i,Dim,"TMP");   \
  io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,i,Dim,"VEL");   \
  io.WriteGidResultsBin1D((PrecisionType*)pres0,N,N,N,i    ,"PRES");  \
  OutputStep = _STEP_;                                                \
}                                                                     \
OutputStep--;                                                         \

double calculateMaxDt_CFL(double CFL, double h, double maxv) {
  return CFL*h / maxv;
}

double calculateMaxDt_Fourier() {
  return 0.0;
}

double calculatePressDt(double dt, double limit) {
  int base = ceil(dt/limit);
  return dt/base;
}

int main(int argc, char *argv[]) {

#ifndef _WIN32
  struct timeval start, end;
#else
  int start, end;
#endif
  double duration = 0.0;

  N           = atoi(argv[1]);
  int steeps  = atoi(argv[2]);
  h           = atoi(argv[3]);

  NB          = atoi(argv[4]);
  NE          = (N+BW)/NB;

  dx          = h/N;
  idx         = 1.0/dx;
  Dim         = 3;

  FileIO io("grid",N);

  Block      * block = NULL;

  Variable3D * step0 = NULL;
  Variable3D * step1 = NULL;
  Variable3D * step2 = NULL;
  Variable3D * step3 = NULL;

  Variable1D * pres0 = NULL;
  Variable1D * pres1 = NULL;

  Variable3D * velf0 = NULL;

  MemManager memmrg(false);

  // Variable
  memmrg.AllocateGrid(&step0, N, N, N, 1);
  memmrg.AllocateGrid(&step1, N, N, N, 1);
  memmrg.AllocateGrid(&step2, N, N, N, 1);
  memmrg.AllocateGrid(&step3, N, N, N, 1);

  // Pressure
  memmrg.AllocateGrid(&pres0, N, N, N, 1);
  memmrg.AllocateGrid(&pres1, N, N, N, 1);

  // Velocity
  memmrg.AllocateGrid(&velf0, N, N, N, 1);

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
  pdt = calculatePressDt(dt,1.0f/(343.2f*343.2f));

  printf("Calculated dt: %f -- %f, %f, %f \n",dt,CFL,h/N,maxv);

  BfeccSolver   AdvectionSolver(block,dt,pdt);
  StencilSolver DiffusionSolver(block,dt,pdt);

  DiffusionSolver.SetDiffTerm(diffTerm);

  WRITE_INIT_R(steeps/20)

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

  for (int i = 0; i < steeps; i++) {
    AdvectionSolver.Execute();
    DiffusionSolver.Execute();
    oldmaxv = maxv;
    block->calculateMaxVelocity(maxv);
    dt = calculateMaxDt_CFL(CFL,dx,maxv);
    pdt = calculatePressDt(dt,1.0f/(343.2f*343.2f));
    // printf("Step %d: %f -- %f, %f, %f, [%f,%f] \n",i,calculateMaxDt_CFL(CFL,dx,maxv),CFL,h/N,maxv,(1.0f/64.0f)/dt,(maxv-oldmaxv));
    WRITE_RESULT(steeps/20)
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
