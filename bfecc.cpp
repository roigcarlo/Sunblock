#include <sys/time.h>
#include <sys/types.h>
#include <omp.h>

// Solver
#include "include/utils.h"
#include "include/block.h"
#include "include/defines.h"
#include "include/solvers.h"
#include "include/file_io.h"
#include "include/interpolator.h"

const double PI = 3.14159265;

uint N          = 0;
uint NB         = 0;
uint NE         = 0;
uint OutputStep = 0;

double dx       =  0.0;
double idx      =  0.0;
double dt       =  0.1; 
double h        = 16.0;
double omega    =  1.0;
double maxv     =  0.0;
double CFL      =  2.0;
double cellSize =  1.0;

typedef Indexer IndexType;
// typedef MortonIndexer IndexType;
typedef Block<VariableType,IndexType> BlockType;
typedef TrilinealInterpolator<VariableType,IndexType,BlockType>  InterpolateType;
typedef BfeccSolver<VariableType,IndexType,BlockType,InterpolateType> BfeccSolverType;

template <typename U>
void precalculateBackAndForw(U * fieldA, U * backward, U * forward,
    const uint &X, const uint &Y, const uint &Z) {

  Variable3DType origin;

  for(uint k = BWP + omp_get_thread_num(); k < Z + BWP; k+=omp_get_num_threads()) {
    for(uint j = BWP; j < Y + BWP; j++) {
      for(uint i = BWP; i < X + BWP; i++) {
        uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+i;

        origin[0] = i;
        origin[1] = j;
        origin[2] = k;

        for(int d = 0; d < 3; d++) {
          backward[cell][d] = origin[d]*h-fieldA[cell][d]*dt;
          forward[cell][d]  = origin[d]*h+fieldA[cell][d]*dt;
        }
      }
    }
  }

}

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

  N           = atoi(argv[1]);
  int steeps  = atoi(argv[2]);
  h           = atoi(argv[3]);

  NB          = atoi(argv[4]);
  NE          = (N+BW)/NB;

  dx          = h/N;
  idx         = 1.0/dx;

  struct timeval start, end;

  IndexType::PreCalculateIndexTable(N+BW);

  FileIO<VariableType> io("grid",N);

  BlockType * block = NULL;

  VariableType * step0 = NULL;
  VariableType * step1 = NULL;
  VariableType * step2 = NULL;

  Variable3DType * velf0 = NULL;
  Variable3DType * velf1 = NULL;
  Variable3DType * velf2 = NULL;

  double duration = 0.0;

  // Temperature
  AllocateGrid(&step0,N,N,N);
  AllocateGrid(&step1,N,N,N);
  AllocateGrid(&step2,N,N,N);

  // Velocity
  AllocateGrid(&velf0,N,N,N);
  AllocateGrid(&velf1,N,N,N);
  AllocateGrid(&velf2,N,N,N);

  printf("Allocation correct\n");
  printf("Initialize\n");

  block = new BlockType(step0,step1,step2,velf0,dx,dt,omega,BW,N,N,N,NB,NE);

  block->InitializeVariable();
  maxv = block->InitializeVelocity();
  block->WriteHeatFocus();

  dt = 0.05; //0.25 * CFL*h/maxv;

  io.WriteGidMesh(step0,N,N,N);

  BfeccSolverType AdvectonStep(block);
  
  #pragma omp parallel
  #pragma omp single
  {
    printf("-------------------\n");
    printf("Running with OMP %d\n",omp_get_num_threads());
    printf("-------------------\n");
  }

  gettimeofday(&start, NULL);

  #pragma omp parallel
  { 
    for(int i = 0; i < steeps; i++) {
      AdvectonStep.Execute();
      // #pragma omp single
      // {
      //   if(OutputStep == 0) {
      //     io.WriteGidResults(step0,N,N,N,i);
      //     OutputStep = 20;
      //   }
      //   OutputStep--;
      // }
    }
  }

  gettimeofday(&end, NULL);

  duration = FETCHTIME

  printf("Total time:\t %f s\n",duration);
  printf("Step  time:\t %f s\n",duration/steeps);

  free(block);

  ReleaseGrid(&step0);
  ReleaseGrid(&step1);
  ReleaseGrid(&step2);

  ReleaseGrid(&velf0);
  ReleaseGrid(&velf1);
  ReleaseGrid(&velf2);

  IndexType::ReleaseIndexTable(N+BW);

  printf("De-Allocation correct\n");

  return 0;
}
