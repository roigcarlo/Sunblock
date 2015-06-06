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
double CFL      =  1.0;
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

#ifndef _WIN32
  struct timeval start, end;
#else
  int start, end;
#endif

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
  AllocateGridCUDA(&step0, N, N, N);
  AllocateGridCUDA(&step1, N, N, N);
  AllocateGridCUDA(&step2, N, N, N);

  // Velocity
  AllocateGridCUDA(&velf0, N, N, N);
  AllocateGridCUDA(&velf1, N, N, N);
  AllocateGridCUDA(&velf2, N, N, N);

  printf("Allocation correct\n");
  printf("Initialize\n");

  block = new BlockType(step0,step1,step2,velf0,dx,dt,omega,BW,N,N,N,NB,NE);

  block->InitializeVariable();
  maxv = block->InitializeVelocity();
  block->WriteHeatFocus();

  dt = 0.0018;// *CFL*h / maxv;

  printf("CFL: %f \t Dt: %f\n", CFL, dt);

  //io.WriteGidMesh(step0,N,N,N);
  //io.WriteGidResults(step0, N, N, N, 0);
  //OutputStep = 1;
  
  BfeccSolverType AdvectonStep(block);
  
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

  int test = 0;

  if (test == 0) {
	AdvectonStep.PrepareCUDA();
	for (int i = 0; i < steeps; i++) {
		AdvectonStep.ExecuteCUDA();
		//if (OutputStep == 0) {
		//io.WriteGidResults(step0,N,N,N,i);
		//OutputStep = 200;
		//}
		//OutputStep--;
	}
	AdvectonStep.FinishCUDA();
	cudaDeviceReset();
  } else {
	#pragma omp parallel
	{
		for (int i = 0; i < steeps; i++) {
			AdvectonStep.Execute();
			#pragma omp single
			{
				if (OutputStep == 0) {
					io.WriteGidResults(step0,N,N,N,i);
					OutputStep = 200;
				}
				OutputStep--;
			}
		}
	}
  }

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

  ReleaseGridCUDA(&step0);
  ReleaseGridCUDA(&step1);
  ReleaseGridCUDA(&step2);

  ReleaseGridCUDA(&velf0);
  ReleaseGridCUDA(&velf1);
  ReleaseGridCUDA(&velf2);

  IndexType::ReleaseIndexTable(N+BW);

  printf("De-Allocation correct\n");

  return 0;
}
