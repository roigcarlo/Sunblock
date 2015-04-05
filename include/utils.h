template <typename T>
void AllocateGrid(T ** grid,
    const uint &X, const uint &Y, const uint &Z) {

  *grid = (T *)malloc(sizeof(T) * 
    (X+BW) * (Y+BW) * (Z+BW));
}

template <typename T>
void AllocateGrid(T * grid,
    const uint &X, const uint &Y, const uint &Z, const uint &align) {

  uint size = (X+BW) * (Y+BW) * (Z+BW);

  while((sizeof(T) * size) % align) size++;

#ifdef _WIN32
  *grid = (T *)_aligned_malloc(sizeof(T) * size,align);
#else
  *grid = (T *)memalign(align,sizeof(T) * size);
#endif
}

template <typename T>
void ReleaseGrid(T * grid) {

  free(*grid);
}

template <typename T>
void ReleaseGrid(T * grid, const uint &align) {
  
// Memory allocated with _aligned_malloc cannot be 
//  released using free. 
#ifdef _WIN32
  _aligned_free(*grid);
#else
  free(*grid);
#endif
}