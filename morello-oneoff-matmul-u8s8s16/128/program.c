// Impl
// ---------------------------------------------------------------------------------------------------------------------------------------------------------
// /* Matmul((128×128, u8), (128×128, i8), (128×128, i16), serial), 0, 1024, 32768, 0, 20885504 */
// alloc aa: (128×128, i8, L1) <- #1
//   /* Matmul((128×128, u8), (128×128, i8, L1), (128×128, i16), serial), 0, 1024, 4096, 0, 20829184 */
//   tile (ab: (8×128, u8) <-[0, 2]- #0, ac: (8×128, i16) <-[0, 1]- #2)
//     /* Matmul((8×128, u8), (128×128, i8, L1), (8×128, i16), serial), 0, 1024, 4096, 0, 1301824 */
//     alloc ad: (8×128, i16, L1) <- ac
//       /* Matmul((8×128, u8), (128×128, i8, L1), (8×128, i16, L1), serial), 0, 1024, 1024, 0, 1287744 */
//       alloc ae: (8×128, u8, L1) <- ab
//         /* Matmul((8×128, u8, L1), (128×128, i8, L1), (8×128, i16, L1), serial), 0, 1024, 0, 0, 1284224 */
//         tile (af: (128×32, i8, L1, c1) <-[3, 1]- aa, ag: (8×32, i16, L1, c1) <-[0, 1]- ad)
//           /* Matmul((8×128, u8, L1), (128×32, i8, L1, c1), (8×32, i16, L1, c1), serial), 0, 1024, 0, 0, 321056 */
//           alloc ah: (8×32, i16, VRF, 16)
//               /* Zero((8×32, i16, VRF, 16), serial), 0, 0, 0, 0, 16 */
//               tile (ai: (1×32, i16, VRF, 16) <-[0, 1]- ah)
//                 /* Zero((1×32, i16, VRF, 16), serial), 0, 0, 0, 0, 2 */
//                 tile (aj: (1×16, i16, VRF, 16) <-[0, 1]- ai)
//                   /* Zero((1×16, i16, VRF, 16), serial), 0, 0, 0, 0, 1 */
//                   VectorZero(aj)
//               /* MatmulAccum((8×128, u8, L1), (128×32, i8, L1, c1), (8×32, i16, VRF, 16), serial), 0, 128, 0, 0, 320384 */
//               tile (ak: (8×2, u8, L1, c1) <-[0, 1]- ae, al: (2×32, i8, L1, c1, ua) <-[1, 2]- af)
//                 /* MatmulAccum((8×2, u8, L1, c1), (2×32, i8, L1, c1, ua), (8×32, i16, VRF, 16), serial), 0, 128, 0, 0, 5006 */
//                 alloc am: (2×32, i8, VRF, <[1,0], [None, None]>, 32)
//                   /* Move((2×32, i8, L1, c1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), serial), 0, 64, 0, 0, 6 */
//                   alloc an: (2×32, i8, VRF, 32)
//                     /* Move((2×32, i8, L1, c1, ua), (2×32, i8, VRF, 32), serial), 0, 0, 0, 0, 2 */
//                     tile (ao: (1×32, i8, L1, ua) <-[0, 1]- al, ap: (1×32, i8, VRF, 32) <-[0, 1]- an)
//                       /* Move((1×32, i8, L1, ua), (1×32, i8, VRF, 32), serial), 0, 0, 0, 0, 1 */
//                       VectorAssign(ao, ap)
//                     /* Move((2×32, i8, VRF, 32), (2×32, i8, VRF, <[1,0], [None, None]>, 32), serial), 0, 0, 0, 0, 4 */
//                     PhysicalTransposeByte256(an, am)
//                   /* MatmulAccum((8×2, u8, L1, c1), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (8×32, i16, VRF, 16), serial), 0, 0, 0, 0, 4960 */
//                   tile (aq: (1×2, u8, L1, ua) <-[0, 2]- ak, ar: (1×32, i16, VRF, 16) <-[0, 1]- ah)
//                     /* MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), 0, 0, 0, 0, 620 */
//                     tile (as: (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32) <-[3, 1]- am, at: (1×16, i16, VRF, 16) <-[0, 1]- ar)
//                       /* MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), 0, 0, 0, 0, 310 */
//                       TwoVecBroadcastVecMultAdd(aq, as, at)
//             /* Move((8×32, i16, VRF, 16), (8×32, i16, L1, c1), serial), 0, 0, 0, 0, 16 */
//             tile (au: (1×32, i16, VRF, 16) <-[0, 1]- ah, av: (1×32, i16, L1, ua) <-[0, 1]- ag)
//               /* Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), 0, 0, 0, 0, 2 */
//               tile (aw: (1×16, i16, VRF, 16) <-[0, 1]- au, ax: (1×16, i16, L1, ua) <-[0, 1]- av)
//                 /* Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), 0, 0, 0, 0, 1 */
//                 VectorAssign(aw, ax)

#include <inttypes.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#ifdef BYTE_ORDER
#if BYTE_ORDER == BIG_ENDIAN
#define LE_TO_CPU32(val) (((val & 0x000000FFU) << 24) | \
                          ((val & 0x0000FF00U) << 8) |  \
                          ((val & 0x00FF0000U) >> 8) |  \
                          ((val & 0xFF000000U) >> 24))
#define LE_TO_CPU16(val) (((val & 0x00FFU) << 8) | \
                          ((val & 0xFF00U) >> 8))
#else
#define LE_TO_CPU32(val) (val)
#define LE_TO_CPU16(val) (val)
#endif
#else
#error "BYTE_ORDER is not defined"
#endif

#include <immintrin.h>
typedef int16_t vsi16 __attribute__ ((vector_size (16 * sizeof(int16_t))));
typedef int8_t vsb32 __attribute__ ((vector_size (32 * sizeof(int8_t))));


struct timespec ts_diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

__attribute__((noinline))
void kernel(
  uint8_t *restrict aa,
  int8_t *restrict ab,
  int16_t *restrict ac
) {
  // (Matmul((128×128, u8), (128×128, i8), (128×128, i16), serial), [64, 1024, 32768, 1073741824])
  // (Matmul((128×128, u8), (128×128, i8, L1), (128×128, i16), serial), [64, 1024, 16384, 1073741824])
  for (int ad = 0; ad < 16; ad++) {
    // (Matmul((8×128, u8), (128×128, i8, L1), (8×128, i16), serial), [64, 1024, 16384, 1073741824])
    // (Matmul((8×128, u8), (128×128, i8, L1), (8×128, i16, L1), serial), [64, 1024, 8192, 1073741824])
    // (Matmul((8×128, u8, L1), (128×128, i8, L1), (8×128, i16, L1), serial), [64, 1024, 4096, 1073741824])
    for (int ae = 0; ae < 4; ae++) {
      // (Matmul((8×128, u8, L1), (128×32, i8, L1, c1), (8×32, i16, L1, c1), serial), [64, 1024, 4096, 1073741824])
      vsi16 ah;
      vsi16 ai;
      vsi16 aj;
      vsi16 ak;
      vsi16 al;
      vsi16 am;
      vsi16 an;
      vsi16 ao;
      vsi16 ap;
      vsi16 aq;
      vsi16 ar;
      vsi16 as;
      vsi16 at;
      vsi16 au;
      vsi16 av;
      vsi16 aw;
      // (Matmul((8×128, u8, L1), (128×32, i8, L1, c1), (8×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((8×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ah *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ai *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      aj *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ak *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      al *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      am *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      an *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ao *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ap *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      aq *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      ar *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      as *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      at *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      au *= 0;  /* VectorZero */
      // (Zero((1×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      av *= 0;  /* VectorZero */
      // (Zero((1×16, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      aw *= 0;  /* VectorZero */
      // (MatmulAccum((8×128, u8, L1), (128×32, i8, L1, c1), (8×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
      for (int ax = 0; ax < 64; ax++) {
        // (MatmulAccum((8×2, u8, L1, c1), (2×32, i8, L1, c1, ua), (8×32, i16, VRF, 16), serial), [64, 512, 4096, 1073741824])
        vsb32 ba;
        vsb32 bb;
        // (Move((2×32, i8, L1, c1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), serial), [64, 256, 4096, 1073741824])
        vsb32 be;
        vsb32 bf;
        // (Move((2×32, i8, L1, c1, ua), (2×32, i8, VRF, 32), serial), [64, 128, 4096, 1073741824])
        // (Move((1×32, i8, L1, ua), (1×32, i8, VRF, 32), serial), [64, 128, 4096, 1073741824])
        _mm_storeu_si128((__m256i *)(&be), _mm_loadu_si128((__m256i *)(ab + (256 * ax + 32 * ae))));  /* VectorAssign */
        // (Move((1×32, i8, L1, ua), (1×32, i8, VRF, 32), serial), [64, 128, 4096, 1073741824])
        _mm_storeu_si128((__m256i *)(&bf), _mm_loadu_si128((__m256i *)(ab + (256 * ax + 32 * ae + 128))));  /* VectorAssign */
        // (Move((2×32, i8, VRF, 32), (2×32, i8, VRF, <[1,0], [None, None]>, 32), serial), [64, 128, 4096, 1073741824])
        vsb32 bh;
        vsb32 bj;
        bh = _mm256_unpacklo_epi8(be, bf);
        bj = _mm256_unpackhi_epi8(be, bf);
        ba = _mm256_permute2f128_si256(bh, bj, 0x20);
        bb = _mm256_permute2f128_si256(bh, bj, 0x31);
        // (MatmulAccum((8×2, u8, L1, c1), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (8×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bk = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax)));
        ah = _mm256_add_epi16(ah, _mm256_maddubs_epi16(bk, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bl = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax)));
        ai = _mm256_add_epi16(ai, _mm256_maddubs_epi16(bl, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bm = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 128)));
        aj = _mm256_add_epi16(aj, _mm256_maddubs_epi16(bm, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bn = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 128)));
        ak = _mm256_add_epi16(ak, _mm256_maddubs_epi16(bn, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bo = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 256)));
        al = _mm256_add_epi16(al, _mm256_maddubs_epi16(bo, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bp = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 256)));
        am = _mm256_add_epi16(am, _mm256_maddubs_epi16(bp, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bq = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 384)));
        an = _mm256_add_epi16(an, _mm256_maddubs_epi16(bq, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i br = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 384)));
        ao = _mm256_add_epi16(ao, _mm256_maddubs_epi16(br, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bs = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 512)));
        ap = _mm256_add_epi16(ap, _mm256_maddubs_epi16(bs, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bt = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 512)));
        aq = _mm256_add_epi16(aq, _mm256_maddubs_epi16(bt, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bu = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 640)));
        ar = _mm256_add_epi16(ar, _mm256_maddubs_epi16(bu, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bv = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 640)));
        as = _mm256_add_epi16(as, _mm256_maddubs_epi16(bv, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bw = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 768)));
        at = _mm256_add_epi16(at, _mm256_maddubs_epi16(bw, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bx = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 768)));
        au = _mm256_add_epi16(au, _mm256_maddubs_epi16(bx, bb));
        // (MatmulAccum((1×2, u8, L1, ua), (2×32, i8, VRF, <[1,0], [None, None]>, 32), (1×32, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i by = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 896)));
        av = _mm256_add_epi16(av, _mm256_maddubs_epi16(by, ba));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, i8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, i16, VRF, 16), serial), [64, 256, 4096, 1073741824])
/* TwoVecBroadcastVecMultAdd */
        __m256i bz = _mm256_set1_epi16(*(int16_t *)(aa + (1024 * ad + 2 * ax + 896)));
        aw = _mm256_add_epi16(aw, _mm256_maddubs_epi16(bz, bb));
      }
      // (Move((8×32, i16, VRF, 16), (8×32, i16, L1, c1), serial), [64, 512, 4096, 1073741824])
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae)), _mm256_loadu_si256((__m256i *)(&ah)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 16)), _mm256_loadu_si256((__m256i *)(&ai)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 128)), _mm256_loadu_si256((__m256i *)(&aj)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 144)), _mm256_loadu_si256((__m256i *)(&ak)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 256)), _mm256_loadu_si256((__m256i *)(&al)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 272)), _mm256_loadu_si256((__m256i *)(&am)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 384)), _mm256_loadu_si256((__m256i *)(&an)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 400)), _mm256_loadu_si256((__m256i *)(&ao)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 512)), _mm256_loadu_si256((__m256i *)(&ap)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 528)), _mm256_loadu_si256((__m256i *)(&aq)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 640)), _mm256_loadu_si256((__m256i *)(&ar)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 656)), _mm256_loadu_si256((__m256i *)(&as)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 768)), _mm256_loadu_si256((__m256i *)(&at)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 784)), _mm256_loadu_si256((__m256i *)(&au)));  /* VectorAssign */
      // (Move((1×32, i16, VRF, 16), (1×32, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 896)), _mm256_loadu_si256((__m256i *)(&av)));  /* VectorAssign */
      // (Move((1×16, i16, VRF, 16), (1×16, i16, L1, ua), serial), [64, 512, 4096, 1073741824])
      _mm256_storeu_si256((__m256i *)(ac + (1024 * ad + 32 * ae + 912)), _mm256_loadu_si256((__m256i *)(&aw)));  /* VectorAssign */
    }
  }
}

int main(int argc, char *argv[]) {
  const char *inner_steps_env = getenv("CHERRYBENCH_LOOP_STEPS");
  if (inner_steps_env == NULL) {
    fprintf(stderr, "Environment variable CHERRYBENCH_LOOP_STEPS is not set.\n");
    exit(1);
  }
  const int bench_samples = atoi(inner_steps_env);

  uint8_t *restrict by;
  posix_memalign((void **)&by, 128, 16384*sizeof(uint8_t));
  for (size_t idx = 0; idx < 16384; idx++) {
    by[idx] = (uint8_t)rand();
  }

  int8_t *restrict bz;
  posix_memalign((void **)&bz, 128, 16384*sizeof(int8_t));
  for (size_t idx = 0; idx < 16384; idx++) {
    bz[idx] = (int8_t)rand();
  }

  int16_t *restrict ca;
  posix_memalign((void **)&ca, 128, 16384*sizeof(int16_t));
  for (size_t idx = 0; idx < 16384; idx++) {
    ca[idx] = (int16_t)rand();
  }

  if (argc != 1) {
    fprintf(stderr, "Unexpected number of arguments.\n");
    return 1;
  }

  // Inlined kernel follows. This is for warm-up.
  for (int i = 0; i < 10; ++i) {
    kernel(by + (0), bz + (0), ca + (0));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
  #pragma clang loop unroll(disable)
    for (long long bench_itr = 0; bench_itr < bench_samples; ++bench_itr) {
      kernel(by + (0), bz + (0), ca + (0));
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    struct timespec delta = ts_diff(start, end);
    long long elapsed_ns = delta.tv_sec * 1000000000L + delta.tv_nsec;
    printf("%lldns\n", elapsed_ns);
  }

  free(by);
  free(bz);
  free(ca);

  return 0;
}