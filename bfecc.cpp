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

double dx       =  0.0;
double idx      =  0.0;
double dt       =  0.1;
double h        = 16.0;
double omega    =  1.0;
double maxv     =  0.0;
double CFL      =  0.25;
double cellSize =  1.0;
double diffTerm =  1e-5;

#define WRITE_INIT_R(_STEP_)                                          \
io.WriteGidMeshWithSkinBin(N,N,N);                                            \
io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,0,Dim,"TMP");         \
io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,0,Dim,"VEL");     \
OutputStep = _STEP_;                                                  \

#define WRITE_RESULT(_STEP_)                                          \
if (OutputStep == 0) {                                                \
  io.WriteGidResultsBin3D((PrecisionType*)step0,N,N,N,i,Dim,"TMP");       \
  io.WriteGidResultsBin3D((PrecisionType*)velf0,N,N,N,i,Dim,"VEL");   \
  OutputStep = _STEP_;                                                \
}                                                                     \
OutputStep--;                                                         \

double calculateMaxDt_CFL(double CFL, double h, double maxv) {
  return CFL*h / maxv;
}

double calculateMaxDt_Fourier() {
  return 0.0;
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

  Variable1D * pres0 = NULL;
  Variable1D * pres1 = NULL;
  Variable1D * pres2 = NULL;

  Variable3D * velf0 = NULL;

  MemManager memmrg(false);

  // Variable
  memmrg.AllocateGrid(&step0, N, N, N, 1);
  memmrg.AllocateGrid(&step1, N, N, N, 1);
  memmrg.AllocateGrid(&step2, N, N, N, 1);

  // Pressure
  memmrg.AllocateGrid(&pres0, N, N, N, 1);
  memmrg.AllocateGrid(&pres1, N, N, N, 1);
  memmrg.AllocateGrid(&pres2, N, N, N, 1);

  // Velocity
  memmrg.AllocateGrid(&velf0, N, N, N, 1);

  printf("Allocation correct\n");
  printf("Initialize\n");

  block = new Block(
    (PrecisionType*) step0,
    (PrecisionType*) step1,
    (PrecisionType*) step2,
    (PrecisionType*) velf0,
    dx,
    omega,
    BW,
    N,
    N,
    N,
    NB,
    NE,
    Dim
  );

  block->Zero();
  block->InitializeVelocity(maxv);
  // block->WriteHeatFocus();

  dt = calculateMaxDt_CFL(CFL,dx,maxv);

  printf("Calculated dt: %f -- %f, %f, %f \n",calculateMaxDt_CFL(CFL,dx,maxv),CFL,h/N,maxv);

  BfeccSolver   AdvectionSolver(block,dt);
  StencilSolver DiffusionSolver(block,dt);

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
    //DiffusionSolver.Execute();
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

  memmrg.ReleaseGrid(&pres0, 1);
  memmrg.ReleaseGrid(&pres1, 1);
  memmrg.ReleaseGrid(&pres2, 1);

  memmrg.ReleaseGrid(&velf0, 1);

  printf("De-Allocation correct\n");

  return 0;
}
