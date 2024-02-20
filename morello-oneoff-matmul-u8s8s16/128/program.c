// Impl
// -------------------------------------------------------------------------------------------------------------------------------------------------------------------
// /* Matmul((128×128, u8), (128×128, s8), (128×128, s16), serial), 64, 1024, 32768, 16384, 20928512 */
// alloc aa: (128×128, s8, <[0,1,0], [None, None, Some(2)]>)
//   /* Move((128×128, s8), (128×128, s8, <[0,1,0], [None, None, Some(2)]>), serial), 64, 64, 64, 0, 83968 */
//   tile (ab: (2×128, s8) <-[0, 1]- #1, ac: (2×128, s8, <[1,0], [None, None]>, ua) <-[0, 1]- aa)
//     /* Move((2×128, s8), (2×128, s8, <[1,0], [None, None]>, ua), serial), 64, 64, 64, 0, 1312 */
//     tile (ad: (2×32, s8, c1) <-[0, 1]- ab, ae: (2×32, s8, <[1,0], [None, None]>, ua) <-[0, 1]- ac)
//       /* Move((2×32, s8, c1), (2×32, s8, <[1,0], [None, None]>, ua), serial), 64, 64, 64, 0, 328 */
//       alloc af: (2×32, s8, L1, <[1,0], [None, None]>, ua) <- ae
//         /* Move((2×32, s8, c1), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), 64, 64, 0, 0, 108 */
//         alloc ag: (2×32, s8, VRF, 32)
//           /* Move((2×32, s8, c1), (2×32, s8, VRF, 32), serial), 0, 0, 0, 0, 2 */
//           tile (ah: (1×32, s8, ua) <-[0, 1]- ad, ai: (1×32, s8, VRF, 32) <-[0, 1]- ag)
//             /* Move((1×32, s8, ua), (1×32, s8, VRF, 32), serial), 0, 0, 0, 0, 1 */
//             VectorAssign(ah, ai)
//           /* Move((2×32, s8, VRF, 32), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), 64, 0, 0, 0, 86 */
//           alloc aj: (2×32, s8, RF)
//             /* Move((2×32, s8, VRF, 32), (2×32, s8, RF), serial), 0, 0, 0, 0, 2 */
//             tile (ak: (1×32, s8, VRF, 32) <-[0, 1]- ag, al: (1×32, s8, RF) <-[0, 1]- aj)
//               /* Move((1×32, s8, VRF, 32), (1×32, s8, RF), serial), 0, 0, 0, 0, 1 */
//               VectorAssign(ak, al)
//             /* Move((2×32, s8, RF), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), 0, 0, 0, 0, 64 */
//             tile (am: (1×32, s8, RF) <-[0, 1]- aj, an: (1×32, s8, L1, <[1,0], [None, None]>, c1, ua) <-[0, 1]- af)
//               /* Move((1×32, s8, RF), (1×32, s8, L1, <[1,0], [None, None]>, c1, ua), serial), 0, 0, 0, 0, 32 */
//               tile (ao: (1×1, s8, RF) <-[0, 1]- am, ap: (1×1, s8, L1, ua) <-[0, 1]- an)
//                 /* Move((1×1, s8, RF), (1×1, s8, L1, ua), serial), 0, 0, 0, 0, 1 */
//                 ValueAssign(ao, ap)
//   /* Matmul((128×128, u8), (128×128, s8, <[0,1,0], [None, None, Some(2)]>), (128×128, s16), serial), 0, 1024, 32768, 0, 20742144 */
//   alloc aq: (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>) <- aa
//     /* Matmul((128×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (128×128, s16), serial), 0, 1024, 8192, 0, 20685824 */
//     tile (ar: (16×128, u8) <-[0, 2]- #0, as: (16×128, s16) <-[0, 1]- #2)
//       /* Matmul((16×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16), serial), 0, 1024, 8192, 0, 2585728 */
//       alloc at: (16×128, s16, L1) <- as
//         /* Matmul((16×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16, L1), serial), 0, 1024, 2048, 0, 2557568 */
//         alloc au: (16×128, u8, L1) <- ar
//           /* Matmul((16×128, u8, L1), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16, L1), serial), 0, 1024, 0, 0, 2550528 */
//           tile (av: (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua) <-[3, 1]- aq, aw: (16×16, s16, L1, c1) <-[0, 1]- at)
//             /* Matmul((16×128, u8, L1), (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua), (16×16, s16, L1, c1), serial), 0, 1024, 0, 0, 318816 */
//             alloc ax: (16×16, s16, VRF, 16)
//                 /* Zero((16×16, s16, VRF, 16), serial), 0, 0, 0, 0, 16 */
//                 tile (ay: (1×16, s16, VRF, 16) <-[0, 1]- ax)
//                   /* Zero((1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 1 */
//                   VectorZero(ay)
//                 /* MatmulAccum((16×128, u8, L1), (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua), (16×16, s16, VRF, 16), serial), 0, 32, 0, 0, 318144 */
//                 tile (az: (16×2, u8, L1, c1) <-[0, 1]- au, ba: (2×16, s8, L1, <[1,0], [None, None]>, ua) <-[1, 2]- av)
//                   /* MatmulAccum((16×2, u8, L1, c1), (2×16, s8, L1, <[1,0], [None, None]>, ua), (16×16, s16, VRF, 16), serial), 0, 32, 0, 0, 4971 */
//                   alloc bb: (2×16, s8, VRF, <[1,0], [None, None]>, 32)
//                     /* Move((2×16, s8, L1, <[1,0], [None, None]>, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), serial), 0, 0, 0, 0, 1 */
//                     VectorAssign(ba, bb)
//                     /* MatmulAccum((16×2, u8, L1, c1), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (16×16, s16, VRF, 16), serial), 0, 0, 0, 0, 4960 */
//                     tile (bc: (1×2, u8, L1, ua) <-[0, 2]- az, bd: (1×16, s16, VRF, 16) <-[0, 1]- ax)
//                       /* MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 310 */
//                       TwoVecBroadcastVecMultAdd(bc, bb, bd)
//               /* Move((16×16, s16, VRF, 16), (16×16, s16, L1, c1), serial), 0, 0, 0, 0, 16 */
//               tile (be: (1×16, s16, VRF, 16) <-[0, 1]- ax, bf: (1×16, s16, L1, ua) <-[0, 1]- aw)
//                 /* Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), 0, 0, 0, 0, 1 */
//                 VectorAssign(be, bf)

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
  // (Matmul((128×128, u8), (128×128, s8), (128×128, s16), serial), [64, 1024, 32768, 1073741824])
  int8_t *restrict ad;
  posix_memalign((void **)&ad, 128, 16384*sizeof(int8_t));
  // (Move((128×128, s8), (128×128, s8, <[0,1,0], [None, None, Some(2)]>), serial), [64, 1024, 32768, 536870912])
  for (int ae = 0; ae < 64; ae++) {
    // (Move((2×128, s8), (2×128, s8, <[1,0], [None, None]>, ua), serial), [64, 1024, 32768, 536870912])
    for (int af = 0; af < 4; af++) {
      // (Move((2×32, s8, c1), (2×32, s8, <[1,0], [None, None]>, ua), serial), [64, 1024, 32768, 536870912])
      // (Move((2×32, s8, c1), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), [64, 1024, 16384, 536870912])
      vsb32 ah;
      vsb32 ai;
      // (Move((2×32, s8, c1), (2×32, s8, VRF, 32), serial), [64, 512, 16384, 536870912])
      // (Move((1×32, s8, ua), (1×32, s8, VRF, 32), serial), [64, 512, 16384, 536870912])
      _mm_storeu_si128((__m256i *)(&ah), _mm_loadu_si128((__m256i *)(ab + (256 * ae + 32 * af))));  /* VectorAssign */
      // (Move((1×32, s8, ua), (1×32, s8, VRF, 32), serial), [64, 512, 16384, 536870912])
      _mm_storeu_si128((__m256i *)(&ai), _mm_loadu_si128((__m256i *)(ab + (256 * ae + 32 * af + 128))));  /* VectorAssign */
      // (Move((2×32, s8, VRF, 32), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), [64, 512, 16384, 536870912])
      int8_t aj[64] __attribute__((aligned (128)));
      // (Move((2×32, s8, VRF, 32), (2×32, s8, RF), serial), [0, 512, 16384, 536870912])
      // (Move((1×32, s8, VRF, 32), (1×32, s8, RF), serial), [0, 512, 16384, 536870912])
      *(__m256i *)(&aj[(0)]) = (*(__m256i *)(&ah));  /* VectorAssign */
      // (Move((1×32, s8, VRF, 32), (1×32, s8, RF), serial), [0, 512, 16384, 536870912])
      *(__m256i *)(&aj[(32)]) = (*(__m256i *)(&ai));  /* VectorAssign */
      // (Move((2×32, s8, RF), (2×32, s8, L1, <[1,0], [None, None]>, ua), serial), [0, 512, 16384, 536870912])
      for (int ak = 0; ak < 2; ak++) {
        // (Move((1×32, s8, RF), (1×32, s8, L1, <[1,0], [None, None]>, c1, ua), serial), [0, 512, 16384, 536870912])
        for (int al = 0; al < 32; al++) {
          // (Move((1×1, s8, RF), (1×1, s8, L1, ua), serial), [0, 512, 16384, 536870912])
          ad[(256 * ((ak + 2 * ae) / 2) + 2 * al + 64 * af + ((((ak + 2 * ae) % 2)) / 1))] = aj[(32 * ak + al)];
        }
      }
    }
  }
  // (Matmul((128×128, u8), (128×128, s8, <[0,1,0], [None, None, Some(2)]>), (128×128, s16), serial), [64, 1024, 32768, 536870912])
  // (Matmul((128×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (128×128, s16), serial), [64, 1024, 16384, 536870912])
  for (int am = 0; am < 8; am++) {
    // (Matmul((16×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16), serial), [64, 1024, 16384, 536870912])
    // (Matmul((16×128, u8), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16, L1), serial), [64, 1024, 8192, 536870912])
    // (Matmul((16×128, u8, L1), (128×128, s8, L1, <[0,1,0], [None, None, Some(2)]>), (16×128, s16, L1), serial), [64, 1024, 4096, 536870912])
    for (int an = 0; an < 8; an++) {
      // (Matmul((16×128, u8, L1), (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua), (16×16, s16, L1, c1), serial), [64, 1024, 4096, 536870912])
      vsi16 ap;
      vsi16 aq;
      vsi16 ar;
      vsi16 as;
      vsi16 at;
      vsi16 au;
      vsi16 av;
      vsi16 aw;
      vsi16 ax;
      vsi16 ay;
      vsi16 az;
      vsi16 ba;
      vsi16 bb;
      vsi16 bc;
      vsi16 bd;
      vsi16 be;
      // (Matmul((16×128, u8, L1), (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua), (16×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      // (Zero((16×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      ap *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      aq *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      ar *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      as *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      at *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      au *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      av *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      aw *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      ax *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      ay *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      az *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      ba *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      bb *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      bc *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      bd *= 0;  /* VectorZero */
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      be *= 0;  /* VectorZero */
      // (MatmulAccum((16×128, u8, L1), (128×16, s8, L1, <[0,1,0], [None, None, Some(2)]>, c2, ua), (16×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
      for (int bf = 0; bf < 64; bf++) {
        // (MatmulAccum((16×2, u8, L1, c1), (2×16, s8, L1, <[1,0], [None, None]>, ua), (16×16, s16, VRF, 16), serial), [64, 512, 4096, 536870912])
        vsb32 bh;
        // (Move((2×16, s8, L1, <[1,0], [None, None]>, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), serial), [64, 256, 4096, 536870912])
        _mm_storeu_si128((__m256i *)(&bh), _mm_loadu_si128((__m256i *)(ad + (256 * ((2 * bf) / 2) + 32 * an + ((((2 * bf) % 2)) / 1)))));  /* VectorAssign */
        // (MatmulAccum((16×2, u8, L1, c1), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (16×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bi = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf)));
        ap = _mm256_add_epi16(ap, _mm256_maddubs_epi16(bi, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bj = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 128)));
        aq = _mm256_add_epi16(aq, _mm256_maddubs_epi16(bj, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bk = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 256)));
        ar = _mm256_add_epi16(ar, _mm256_maddubs_epi16(bk, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bl = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 384)));
        as = _mm256_add_epi16(as, _mm256_maddubs_epi16(bl, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bm = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 512)));
        at = _mm256_add_epi16(at, _mm256_maddubs_epi16(bm, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bn = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 640)));
        au = _mm256_add_epi16(au, _mm256_maddubs_epi16(bn, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bo = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 768)));
        av = _mm256_add_epi16(av, _mm256_maddubs_epi16(bo, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bp = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 896)));
        aw = _mm256_add_epi16(aw, _mm256_maddubs_epi16(bp, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bq = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1024)));
        ax = _mm256_add_epi16(ax, _mm256_maddubs_epi16(bq, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i br = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1152)));
        ay = _mm256_add_epi16(ay, _mm256_maddubs_epi16(br, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bs = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1280)));
        az = _mm256_add_epi16(az, _mm256_maddubs_epi16(bs, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bt = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1408)));
        ba = _mm256_add_epi16(ba, _mm256_maddubs_epi16(bt, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bu = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1536)));
        bb = _mm256_add_epi16(bb, _mm256_maddubs_epi16(bu, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bv = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1664)));
        bc = _mm256_add_epi16(bc, _mm256_maddubs_epi16(bv, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bw = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1792)));
        bd = _mm256_add_epi16(bd, _mm256_maddubs_epi16(bw, bh));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 536870912])
/* TwoVecBroadcastVecMultAdd */
        __m256i bx = _mm256_set1_epi16(*(int16_t *)(aa + (2048 * am + 2 * bf + 1920)));
        be = _mm256_add_epi16(be, _mm256_maddubs_epi16(bx, bh));
      }
      // (Move((16×16, s16, VRF, 16), (16×16, s16, L1, c1), serial), [64, 512, 4096, 536870912])
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an)), _mm256_loadu_si256((__m256i *)(&ap)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 128)), _mm256_loadu_si256((__m256i *)(&aq)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 256)), _mm256_loadu_si256((__m256i *)(&ar)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 384)), _mm256_loadu_si256((__m256i *)(&as)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 512)), _mm256_loadu_si256((__m256i *)(&at)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 640)), _mm256_loadu_si256((__m256i *)(&au)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 768)), _mm256_loadu_si256((__m256i *)(&av)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 896)), _mm256_loadu_si256((__m256i *)(&aw)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1024)), _mm256_loadu_si256((__m256i *)(&ax)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1152)), _mm256_loadu_si256((__m256i *)(&ay)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1280)), _mm256_loadu_si256((__m256i *)(&az)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1408)), _mm256_loadu_si256((__m256i *)(&ba)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1536)), _mm256_loadu_si256((__m256i *)(&bb)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1664)), _mm256_loadu_si256((__m256i *)(&bc)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1792)), _mm256_loadu_si256((__m256i *)(&bd)));  /* VectorAssign */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 512, 4096, 536870912])
      _mm256_storeu_si256((__m256i *)(ac + (2048 * am + 16 * an + 1920)), _mm256_loadu_si256((__m256i *)(&be)));  /* VectorAssign */
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