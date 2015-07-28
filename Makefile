CC  = g++
SRC = proxySolver.cpp
OBJ = proxySolver.o bfecc.o
CXXFLAGS = -Wall -Werror -pedantic -msse3 -mavx -mfma -O3
CXXSAFEF = -Wall -Werror -pedantic -O3
CONFIG   = -DUSE_NOVEC -DNO_USE_CUDA
PROFILE  =
OMP = -fopenmp
OMP4 = -fopenmp-simd

all: proxySolver bfecc

proxySolver: proxySolver.o
	$(CC) -o proxySolver $(OMP) proxySolver.o

proxySolver.o: proxySolver.cpp
	$(CC) -c proxySolver.cpp $(OMP) $(CXXFLAGS) $(CONFIG)

bfecc: bfecc.o
	$(CC) -o bfecc $(OMP) $(PROFILE) bfecc.o -L include/gidpost/source -lgidpost -lz

bfecc.o: bfecc.cpp
	$(CC) -c bfecc.cpp $(PROFILE) $(OMP) $(CXXSAFEF) $(CONFIG)

clean:
	$(RM) $(OBJ)
