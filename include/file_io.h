#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "hacks.h"

// GiD IO
#include "gidpost/source/gidpost.h"

class FileIO {
private:

  std::stringstream name_mesh;
  std::stringstream name_post;
  std::stringstream name_raw;

  std::ofstream * mesh_file;
  std::ofstream * post_file;

  /**
   * Sets the name for the mesh file
   * @name:     name of the mesh file
   **/
  void SetMeshName(const char * mesh, const uint &N) {

    name_mesh << mesh << N << ".post.msh";
  }

  /**
   * Sets the name for the post file
   * @name:     name of the post file
   **/
  void setPostName(const char * post, const uint &N) {

    name_post << post << N << ".post.res";
  }

  /**
   * Initial formating for the post file
   **/
  void PreparePostFile() {
    (*post_file) << "GiD Post Results File 1.0" << std::endl << std::endl;
  }

public:

  // Creator & destructor
  FileIO(const char * name, const uint &N) {

    SetMeshName(name,N);
    setPostName(name,N);

    name_raw << name;

    // GiD_OpenPostMeshFile(name_mesh.str().c_str(), GiD_PostAscii);
    GiD_OpenPostResultFile(name_post.str().c_str(), GiD_PostBinary);

    mesh_file = new std::ofstream(name_mesh.str().c_str());
    post_file = new std::ofstream(name_post.str().c_str());
  };

  ~FileIO() {

    mesh_file->close();
    post_file->close();

    // GiD_ClosePostMeshFile();
    GiD_ClosePostResultFile();

    delete mesh_file;
    delete post_file;
  };

  /**
   * Writes the mesh in raw format and wipes previous content in the file.
   * @grid:     Value of the grid in Local or Global coordinatr system
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   * @fileName: Name of the output file
   **/
  void WriteGridWipe(
      PrecisionType * grid,
      const uint &X,
      const uint &Y,
      const uint &Z,
      const char * fileName) {

    std::ofstream outputFile(fileName);

    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        for(uint i = 0; i < X + BW; i++) {
          outputFile << grid[k*(Y+BW)*(X+BW)+j*(X+BW)+i] << " ";
        }
        outputFile << std::endl;
      }
      outputFile << std::endl;
    }
    outputFile << "%" << std::endl;
  }

  /**
   * Writes the mesh in raw format without wiping previous content in the file.
   * @grid:     Value of the grid in Local or Global coordinatr system
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   * @fileName: Name of the output file
   **/
  void WriteGrid(
      PrecisionType * grid,
      const uint &X,
      const uint &Y,
      const uint &Z,
      const char * fileName) {

    std::ofstream outputFile(fileName,std::ofstream::app);

    outputFile << std::fixed;
    outputFile << std::setprecision(2);

    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        for(uint i = 0; i < X + BW; i++) {
          outputFile << grid[k*(Y+BW)*(X+BW)+j*(X+BW)+i] << " ";
        }
        outputFile << std::endl;
      }
      outputFile << std::endl;
    }
    outputFile << "%" << std::endl;
  }

  /**
   * Writes the mesh in GiD format.
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   **/
  void WriteGidMesh(
      const uint &X,
      const uint &Y,
      const uint &Z) {

    (*mesh_file) << "MESH \"Grid\" dimension 3 ElemType Hexahedra Nnode 8" << std::endl;
    (*mesh_file) << "# color 96 96 96" << std::endl;
    (*mesh_file) << "Coordinates" << std::endl;
    (*mesh_file) << "# node number coordinate_x coordinate_y coordinate_z  " << std::endl;

    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(uint i = 0; i < X + BW; i++) {
          (*mesh_file) << cell++ << "  " << i << "  " << j << "  " << k << std::endl;
        }
      }
    }

    (*mesh_file) << "end coordinates" << std::endl;
    (*mesh_file) << "Elements" << std::endl;
    (*mesh_file) << "# Element node_1 node_2 node_3 node_4 node_5 node_6 node_7 node_8" << std::endl;

    for(uint k = BWP; k < Z + BWP; k++) {
      for(uint j = BWP; j < Y + BWP; j++) {
        uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(uint i = BWP; i < X + BWP; i++) {
          (*mesh_file) << cell++ << " ";

          (*mesh_file) << cell                        << " " << cell+1                    << "  ";
          (*mesh_file) << cell+1+(Y+BW)               << " " << cell+(Y+BW)               << "  ";
          (*mesh_file) << cell+(Z+BW)*(Y+BW)          << " " << cell+1+(Z+BW)*(Y+BW)      << "  ";
          (*mesh_file) << cell+1+(Z+BW)*(Y+BW)+(Y+BW) << " " << cell+(Z+BW)*(Y+BW)+(Y+BW) << "  ";

          (*mesh_file) << std::endl;
        }
      }
    }

    (*mesh_file) << "end Elements" << std::endl;
  }

  /**
   * Writes the results in GiD format.
   * @grid:     Value of the grid in Local or Global coordinatr system
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   * @step:     Step of the result
   **/
  void WriteGidResults(
      PrecisionType * grid,
      const uint &X,
      const uint &Y,
      const uint &Z,
      int step) {

    (*post_file) << "Result \"Temperature\" \"Kratos\" " << step << " Scalar OnNodes" << std::endl;
    (*post_file) << "Values" << std::endl;

    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        for(uint i = 0; i < X + BW; i++) {
          uint celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP+i;
          uint cell = celln; //interleave64(i,j,k);
          (*post_file) << celln << "  " << grid[cell] << std::endl; cell++;
        }
      }
    }

    (*post_file) << "End Values" << std::endl;
  }


  /**
   * Writes the mesh in GiD format.
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   **/
  void WriteGidMeshBin(
      const uint &X,
      const uint &Y,
      const uint &Z) {

    int elemi[8];

    GiD_BeginMesh(name_raw.str().c_str(), GiD_3D, GiD_Hexahedra, 8);

    GiD_BeginCoordinates();
    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(uint i = 0; i < X + BW; i++) {
          GiD_WriteCoordinates(cell++, i, j, k);
        }
      }
    }
    GiD_EndCoordinates();

    GiD_BeginElements();
    for(uint k = 0; k < Z + BW - 1; k++) {
      for(uint j = 0; j < Y + BW - 1; j++) {
        uint cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+1;
        for(uint i = 0; i < X + BW - 1; i++) {
          elemi[0] = cell;                        elemi[1] = cell+1;
          elemi[2] = cell+1+(Y+BW);               elemi[3] = cell+(Y+BW);
          elemi[4] = cell+(Z+BW)*(Y+BW);          elemi[5] = cell+1+(Z+BW)*(Y+BW);
          elemi[6] = cell+1+(Z+BW)*(Y+BW)+(Y+BW); elemi[7] = cell+(Z+BW)*(Y+BW)+(Y+BW);

          GiD_WriteElementMat(cell++, elemi);
        }
      }
    }

    GiD_EndElements();
    GiD_EndMesh();
  }

  /**
   * Writes the results in GiD format.
   * @grid:     Value of the grid in Local or Global coordinatr system
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   * @step:     Step of the result
   **/
  void WriteGidResultsBin2D(
      PrecisionType * grid,
      const uint &X,
      const uint &Y,
      const uint &Z,
      int step,
      const char * name) {

    GiD_BeginResult(name, "Static", step, GiD_Scalar, GiD_OnNodes, NULL, NULL, 0, NULL);
    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        for(uint i = 0; i < X + BW; i++) {
          uint celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+i;
          uint cell = celln; //interleave64(i,j,k);
          GiD_WriteScalar(celln+1, grid[cell]);
        }
      }
    }
    GiD_EndResult();
  }

  /**
   * Writes the results in GiD format.
   * @grid:     Value of the grid in Local or Global coordinatr system
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   * @step:     Step of the result
   **/
  void WriteGidResultsBin3D(
      PrecisionType * grid,
      const uint &X,
      const uint &Y,
      const uint &Z,
      int step,
      int dim,
      const char * name) {

    GiD_BeginResult(
      name,
      "Static",
      step,
      GiD_Vector,
      GiD_OnNodes,
      NULL,
      NULL,
      0,
      NULL);

    for(uint k = 0; k < Z + BW; k++) {
      for(uint j = 0; j < Y + BW; j++) {
        for(uint i = 0; i < X + BW; i++) {
          uint celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+i;
          uint cell = celln; //interleave64(i,j,k);

          GiD_WriteVector(
            celln+1,
            grid[cell*dim+0],
            grid[cell*dim+1],
            grid[cell*dim+2]);
        }


      }
    }
    GiD_EndResult();
  }

};
