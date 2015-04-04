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

  // Note: free function should be used with the appropiate _aligned_free if
  //  compiled in widnows systems.

  free(*grid);


}