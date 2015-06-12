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
#include "include/solvers.h"
#include "include/file_io.h"
#include "include/interpolator.h"

uint N          =  0;
uint NB         =  0;
uint NE         =  0;
uint OutputStep =  0;

double dx       =  0.0;
double idx      =  0.0;
double dt       =  0.1; 
double h        = 16.0;
double omega    =  1.0;
double maxv     =  0.0;
double CFL      =  1.0;
double cellSize =  1.0;

typedef Indexer                                                       IndexType;
// typedef MortonIndexer                                                 IndexType;
typedef Block<VariableType,IndexType>                                 BlockType;
typedef TrilinealInterpolator<VariableType,IndexType,BlockType>       InterpolateType;
typedef BfeccSolver<VariableType,IndexType,BlockType,InterpolateType> BfeccSolverType;

template <typename T>
void difussion(T * &gridA, T * &gridB,
    const uint &X, const uint &Y, const uint &Z) {

  for(uint k = BWP + omp_get_thread_num(); k < Z + BWP; k+=omp_get_num_threads()) {
    for(uint j = BWP; j < Y + BWP; j++) {
      uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
      for(uint i = BWP; i < X + BWP; i++) {
        stencilCross(gridA,gridB,cell++,X,Y,Z);
      } 
    }
  }
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

  IndexType::PreCalculateIndexTable(N+BW);

  FileIO<VariableType> io("grid",N);

  BlockType      * block = NULL;

  VariableType   * step0 = NULL;
  VariableType   * step1 = NULL;
  VariableType   * step2 = NULL;

  Variable3DType * velf0 = NULL;
  Variable3DType * velf1 = NULL;
  Variable3DType * velf2 = NULL;

  MemManager memmrg(false);

  // Temperature
  memmrg.AllocateGrid(&step0, N, N, N, 1);
  memmrg.AllocateGrid(&step1, N, N, N, 1);
  memmrg.AllocateGrid(&step2, N, N, N, 1);

  // Velocity
  memmrg.AllocateGrid(&velf0, N, N, N, 1);
  memmrg.AllocateGrid(&velf1, N, N, N, 1);
  memmrg.AllocateGrid(&velf2, N, N, N, 1);

  printf("Allocation correct\n");
  printf("Initialize\n");

  block = new BlockType(step0,step1,step2,velf0,dx,omega,BW,N,N,N,NB,NE);

  block->InitializeVariable();
  block->InitializeVelocity(maxv);
  block->WriteHeatFocus();

  dt = 0.0018;// *CFL*h / maxv;
  printf("CFL: %f \t Dt: %f\n", CFL, dt);

  BfeccSolverType AdvectionSolver(block,dt);

  //io.WriteGidMesh(step0,N,N,N);
  //io.WriteGidResults(step0, N, N, N, 0);
  //OutputStep = 200;
  
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
    // if (OutputStep == 0) {
    //   io.WriteGidResults(step0,N,N,N,i);
    //   OutputStep = 200;
    // }
    // OutputStep--;
  }
  AdvectionSolver.Finish();

#ifndef _WIN32
  gettimeofday(&end, NULL);
  duration = FETCHTIME
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

  memmrg.ReleaseGrid(&velf0, 1);
  memmrg.ReleaseGrid(&velf1, 1);
  memmrg.ReleaseGrid(&velf2, 1);

  IndexType::ReleaseIndexTable(N+BW);

  printf("De-Allocation correct\n");

  return 0;
}
