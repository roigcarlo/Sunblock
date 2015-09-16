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
  void SetMeshName(const char * mesh, const size_t &N) {

    name_mesh << mesh << N << ".post.msh";
  }

  /**
   * Sets the name for the post file
   * @name:     name of the post file
   **/
  void setPostName(const char * post, const size_t &N) {

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
  FileIO(const char * name, const size_t &N) {

    SetMeshName(name,N);
    setPostName(name,N);

    name_raw << name;

    GiD_OpenPostMeshFile(name_mesh.str().c_str(), GiD_PostAscii);
    GiD_OpenPostResultFile(name_post.str().c_str(), GiD_PostAscii);

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

  // void ReadModelPart(
  //     const MemManager & memmrg,
  //     ) {
  // }

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
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const char * fileName) {

    std::ofstream outputFile(fileName);

    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        for(size_t i = 0; i < X + BW; i++) {
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
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const char * fileName) {

    std::ofstream outputFile(fileName,std::ofstream::app);

    outputFile << std::fixed;
    outputFile << std::setprecision(2);

    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        for(size_t i = 0; i < X + BW; i++) {
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
      const PrecisionType &Dx,
      const size_t &X,
      const size_t &Y,
      const size_t &Z) {

    (*mesh_file) << "MESH \"Grid\" dimension 3 ElemType Hexahedra Nnode 8" << std::endl;
    (*mesh_file) << "# color 96 96 96" << std::endl;
    (*mesh_file) << "Coordinates" << std::endl;
    (*mesh_file) << "# node number coordinate_x coordinate_y coordinate_z  " << std::endl;

    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(size_t i = 0; i < X + BW; i++) {
          (*mesh_file) << cell++ << "  " << i*Dx << "  " << j*Dx << "  " << k*Dx << std::endl;
        }
      }
    }

    (*mesh_file) << "end coordinates" << std::endl;
    (*mesh_file) << "Elements" << std::endl;
    (*mesh_file) << "# Element node_1 node_2 node_3 node_4 node_5 node_6 node_7 node_8" << std::endl;

    for(size_t k = BWP; k < Z + BWP; k++) {
      for(size_t j = BWP; j < Y + BWP; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(size_t i = BWP; i < X + BWP; i++) {
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
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const int &step) {

    (*post_file) << "Result \"Temperature\" \"Kratos\" " << step << " Scalar OnNodes" << std::endl;
    (*post_file) << "Values" << std::endl;

    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        for(size_t i = 0; i < X + BW; i++) {
          size_t celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP+i;
          size_t cell = celln; //interleave64(i,j,k);
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
  void WriteGidMeshWithSkinBin(
      const PrecisionType &Dx,
      const size_t &X,
      const size_t &Y,
      const size_t &Z) {

    int elemi[8];

    GiD_BeginMesh(name_raw.str().c_str(), GiD_3D, GiD_Hexahedra, 8);

    GiD_BeginCoordinates();
    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(size_t i = 0; i < X + BW; i++) {
          GiD_WriteCoordinates(
            (int)cell++,
            (PrecisionType)i*Dx,
            (PrecisionType)j*Dx,
            (PrecisionType)k*Dx
          );
        }
      }
    }
    GiD_EndCoordinates();

    GiD_BeginElements();
    for(size_t k = 0; k < Z + BW - 1; k++) {
      for(size_t j = 0; j < Y + BW - 1; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+1;
        for(size_t i = 0; i < X + BW - 1; i++) {
          elemi[0] = (int)(cell);
          elemi[1] = (int)(cell+1);
          elemi[2] = (int)(cell+1+(Y+BW));
          elemi[3] = (int)(cell+(Y+BW));
          elemi[4] = (int)(cell+(Z+BW)*(Y+BW));
          elemi[5] = (int)(cell+1+(Z+BW)*(Y+BW));
          elemi[6] = (int)(cell+1+(Z+BW)*(Y+BW)+(Y+BW));
          elemi[7] = (int)(cell+(Z+BW)*(Y+BW)+(Y+BW));

          GiD_WriteElement(
            (int)cell++,
            elemi);
        }
      }
    }

    GiD_EndElements();
    GiD_EndMesh();
  }


  /**
   * Writes the mesh in GiD format.
   * @X:        X-Size of the grid
   * @Y:        Y-Size of the grid
   * @Z:        Z-Size of the grid
   **/
  void WriteGidMeshBin(
      const PrecisionType &Dx,
      const size_t &X,
      const size_t &Y,
      const size_t &Z) {

    int elemi[8];

    GiD_BeginMesh(name_raw.str().c_str(), GiD_3D, GiD_Hexahedra, 8);

    GiD_BeginCoordinates();
    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP;
        for(size_t i = 0; i < X + BW; i++) {
          GiD_WriteCoordinates(
            (int)cell++,
            (PrecisionType)1.0f-i*Dx,
            (PrecisionType)1.0f-j*Dx,
            (PrecisionType)1.0f-k*Dx
          );
        }
      }
    }
    GiD_EndCoordinates();

    GiD_BeginElements();
    for(size_t k = BWP; k < Z + BWP - 1; k++) {
      for(size_t j = BWP; j < Y + BWP - 1; j++) {
        size_t cell = k*(Z+BW)*(Y+BW)+j*(Y+BW)+BWP+1;
        for(size_t i = BWP; i < X + BWP - 1; i++) {
          elemi[0] = (int)(cell);
          elemi[1] = (int)(cell+1);
          elemi[2] = (int)(cell+1+(Y+BW));
          elemi[3] = (int)(cell+(Y+BW));
          elemi[4] = (int)(cell+(Z+BW)*(Y+BW));
          elemi[5] = (int)(cell+1+(Z+BW)*(Y+BW));
          elemi[6] = (int)(cell+1+(Z+BW)*(Y+BW)+(Y+BW));
          elemi[7] = (int)(cell+(Z+BW)*(Y+BW)+(Y+BW));

          GiD_WriteElement(
            (int)cell++,
            elemi);
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
  void WriteGidResultsBin1D(
      PrecisionType * grid,
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const int step,
      const char * name) {

    GiD_BeginResult(name, "Static", step, GiD_Scalar, GiD_OnNodes, NULL, NULL, 0, NULL);
    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        for(size_t i = 0; i < X + BW; i++) {
          size_t celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+i;
          size_t cell = celln; //interleave64(i,j,k);

          GiD_WriteScalar(
            (int)(celln+1),
            grid[cell]);
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
      const size_t &X,
      const size_t &Y,
      const size_t &Z,
      const int &step,
      const size_t &dim,
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

    for(size_t k = 0; k < Z + BW; k++) {
      for(size_t j = 0; j < Y + BW; j++) {
        for(size_t i = 0; i < X + BW; i++) {
          size_t celln = k*(Z+BW)*(Y+BW)+j*(Y+BW)+i;
          size_t cell = celln; //interleave64(i,j,k);

          GiD_WriteVector(
            (int)(celln+1),
            grid[cell*dim+0],
            grid[cell*dim+1],
            grid[cell*dim+2]);
        }
      }
    }
    GiD_EndResult();
  }

};
