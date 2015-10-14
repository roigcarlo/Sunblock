#ifndef BHACKS_H
#define BHACKS_H

// BitHacks and other shady things
#include <sys/types.h>

__uint64_t interleave64(
    const __uint64_t &IX,
    const __uint64_t &IY
  ) {

  // Strip the first half of the numbers
  __uint64_t X = IX & ( 0x00000000FFFFFFFFU );
  __uint64_t Y = IY & ( 0x00000000FFFFFFFFU );

  X = ( X | ( X << 16 ) ) & 0x0000FFFF0000FFFFU;
  X = ( X | ( X << 8  ) ) & 0x00FF00FF00FF00FFU;
  X = ( X | ( X << 4  ) ) & 0x0F0F0F0F0F0F0F0FU;
  X = ( X | ( X << 2  ) ) & 0x3333333333333333U;
  X = ( X | ( X << 1  ) ) & 0x5555555555555555U;

  Y = ( Y | ( Y << 16 ) ) & 0x0000FFFF0000FFFFU;
  Y = ( Y | ( Y << 8  ) ) & 0x00FF00FF00FF00FFU;
  Y = ( Y | ( Y << 4  ) ) & 0x0F0F0F0F0F0F0F0FU;
  Y = ( Y | ( Y << 2  ) ) & 0x3333333333333333U;
  Y = ( Y | ( Y << 1  ) ) & 0x5555555555555555U;

  return X | ( Y << 1 );
}

__uint64_t interleave64(
    const __uint64_t &IX,
    const __uint64_t &IY,
    const __uint64_t &IZ
  ) {

  // Strip the first half of the numbers
  __uint64_t X = IX & ( 0x000000000000FFFFU );
  __uint64_t Y = IY & ( 0x000000000000FFFFU );
  __uint64_t Z = IZ & ( 0x000000000000FFFFU );

  X = ( X | ( X << 16 ) ) & 0x00FF0000FF0000FFU;
  X = ( X | ( X << 8  ) ) & 0xF00F00F00F00F00FU;
  X = ( X | ( X << 4  ) ) & 0x30C30C30C30C30C3U;
  X = ( X | ( X << 2  ) ) & 0x0249249249249249U;

  Y = ( Y | ( Y << 16 ) ) & 0x00FF0000FF0000FFU;
  Y = ( Y | ( Y << 8  ) ) & 0xF00F00F00F00F00FU;
  Y = ( Y | ( Y << 4  ) ) & 0x30C30C30C30C30C3U;
  Y = ( Y | ( Y << 2  ) ) & 0x0249249249249249U;

  Z = ( Z | ( Z << 16 ) ) & 0x00FF0000FF0000FFU;
  Z = ( Z | ( Z << 8  ) ) & 0xF00F00F00F00F00FU;
  Z = ( Z | ( Z << 4  ) ) & 0x30C30C30C30C30C3U;
  Z = ( Z | ( Z << 2  ) ) & 0x0249249249249249U;

  return ( X | ( Y << 1 ) | ( Z << 2 ) ) & 0x0000FFFFFFFFFFFFU;
}

#endif
