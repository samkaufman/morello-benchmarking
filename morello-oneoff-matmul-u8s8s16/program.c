// Impl
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// /* Matmul((64×64, u8), (64×64, s8), (64×64, s16), serial), 32, 1024, 16384, 0, 2573440 */
// alloc aa: (64×64, s16, L1) <- #2
//     /* Zero((64×64, s16, L1), serial), 0, 32, 0, 0, 5632 */
//     tile (ab: (1×64, s16, L1) <-[0, 1]- aa)
//       /* Zero((1×64, s16, L1), serial), 0, 32, 0, 0, 88 */
//       tile (ac: (1×16, s16, L1) <-[0, 1]- ab)
//         /* Zero((1×16, s16, L1), serial), 0, 32, 0, 0, 22 */
//         alloc ad: (1×16, s16, VRF, 16)
//           /* Zero((1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 1 */
//           VectorZero(ad)
//           /* Move((1×16, s16, VRF, 16), (1×16, s16, L1), serial), 0, 0, 0, 0, 1 */
//           VectorAssign(ad, ac)
//     /* MatmulAccum((64×64, u8), (64×64, s8), (64×64, s16, L1), serial), 32, 1024, 8192, 0, 2511488 */
//     alloc ae: (64×64, u8, L1) <- #0
//       /* MatmulAccum((64×64, u8, L1), (64×64, s8), (64×64, s16, L1), serial), 32, 1024, 2048, 0, 2497408 */
//       tile (af: (64×32, u8, L1, c1) <-[0, 1]- ae, ag: (32×64, s8) <-[1, 2]- #1)
//         /* MatmulAccum((64×32, u8, L1, c1), (32×64, s8), (64×64, s16, L1), serial), 32, 1024, 2048, 0, 1248704 */
//         alloc ah: (32×64, s8, L1) <- ag
//           /* MatmulAccum((64×32, u8, L1, c1), (32×64, s8, L1), (64×64, s16, L1), serial), 32, 1024, 0, 0, 1241664 */
//           tile (ai: (32×16, s8, L1, c1) <-[3, 1]- ah, aj: (64×16, s16, L1, c1) <-[0, 1]- aa)
//             /* MatmulAccum((64×32, u8, L1, c1), (32×16, s8, L1, c1), (64×16, s16, L1, c1), serial), 32, 1024, 0, 0, 310416 */
//             alloc ak: (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32)
//               /* Move((32×16, s8, L1, c1), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), serial), 32, 0, 0, 0, 1168 */
//               tile (al: (2×16, s8, L1, c1, ua) <-[0, 1]- ai, am: (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32) <-[0, 1]- ak)
//                 /* Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), 32, 0, 0, 0, 73 */
//                 alloc an: (2×16, s8, RF, <[1,0], [None, None]>)
//                   /* Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), 0, 0, 0, 0, 32 */
//                   tile (ao: (1×16, s8, L1, ua) <-[0, 1]- al, ap: (1×16, s8, RF, <[1,0], [None, None]>, c1, ua) <-[0, 1]- an)
//                     /* Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), 0, 0, 0, 0, 16 */
//                     tile (aq: (1×1, s8, L1, ua) <-[0, 1]- ao, ar: (1×1, s8, RF, ua) <-[0, 1]- ap)
//                       /* Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), 0, 0, 0, 0, 1 */
//                       ValueAssign(aq, ar)
//                   /* Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), 0, 0, 0, 0, 1 */
//                   VectorAssign(an, am)
//               /* MatmulAccum((64×32, u8, L1, c1), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (64×16, s16, L1, c1), serial), 0, 32, 0, 0, 308608 */
//               tile (as: (1×32, u8, L1, ua) <-[0, 2]- af, at: (1×16, s16, L1, ua) <-[0, 1]- aj)
//                 /* MatmulAccum((1×32, u8, L1, ua), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (1×16, s16, L1, ua), serial), 0, 32, 0, 0, 4822 */
//                 alloc au: (1×16, s16, VRF, 16)
//                   /* Move((1×16, s16, L1, ua), (1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 1 */
//                   VectorAssign(at, au)
//                   /* MatmulAccum((1×32, u8, L1, ua), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 4800 */
//                   tile (av: (1×2, u8, L1, ua) <-[0, 1]- as, aw: (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32) <-[1, 2]- ak)
//                     /* MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), 0, 0, 0, 0, 300 */
//                     TwoVecBroadcastVecMult(av, aw, au)
//                   /* Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), 0, 0, 0, 0, 1 */
//                   VectorAssign(au, at)

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <immintrin.h>

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
typedef int8_t vsb32 __attribute__ ((vector_size (32 * sizeof(int8_t))));
typedef int16_t vsi16 __attribute__ ((vector_size (16 * sizeof(int16_t))));


__attribute__((noinline))
void kernel(
  uint8_t *__restrict aa,
  int8_t *__restrict ab,
  int16_t *__restrict ac
) {
  // (Matmul((64×64, u8), (64×64, s8), (64×64, s16), serial), [64, 1024, 32768, 1073741824])
  // (Matmul((64×64, u8), (64×64, s8), (64×64, s16, L1), serial), [64, 1024, 16384, 1073741824])
  // (Zero((64×64, s16, L1), serial), [64, 1024, 16384, 1073741824])
  for (int ad = 0; ad < 64; ad++) {
    // (Zero((1×64, s16, L1), serial), [64, 1024, 16384, 1073741824])
    for (int ae = 0; ae < 4; ae++) {
      // (Zero((1×16, s16, L1), serial), [64, 1024, 16384, 1073741824])
      vsi16 ag;
      // (Zero((1×16, s16, VRF, 16), serial), [64, 512, 16384, 1073741824])
      ag *= 0;  /* VectorZero */
      // (Move((1×16, s16, VRF, 16), (1×16, s16, L1), serial), [64, 512, 16384, 1073741824])
      *(__m256i *)(ac + (64 * ad + 16 * ae)) = (*(__m256i *)(&ag));  /* VectorAssign */
    }
  }
  // (MatmulAccum((64×64, u8), (64×64, s8), (64×64, s16, L1), serial), [64, 1024, 16384, 1073741824])
  // (MatmulAccum((64×64, u8, L1), (64×64, s8), (64×64, s16, L1), serial), [64, 1024, 8192, 1073741824])
  for (int ah = 0; ah < 2; ah++) {
    // (MatmulAccum((64×32, u8, L1, c1), (32×64, s8), (64×64, s16, L1), serial), [64, 1024, 8192, 1073741824])
    // (MatmulAccum((64×32, u8, L1, c1), (32×64, s8, L1), (64×64, s16, L1), serial), [64, 1024, 4096, 1073741824])
    for (int ai = 0; ai < 4; ai++) {
      // (MatmulAccum((64×32, u8, L1, c1), (32×16, s8, L1, c1), (64×16, s16, L1, c1), serial), [64, 1024, 4096, 1073741824])
      vsb32 ak;
      vsb32 al;
      vsb32 am;
      vsb32 an;
      vsb32 ao;
      vsb32 ap;
      vsb32 aq;
      vsb32 ar;
      vsb32 as;
      vsb32 at;
      vsb32 au;
      vsb32 av;
      vsb32 aw;
      vsb32 ax;
      vsb32 ay;
      vsb32 az;
      // (Move((32×16, s8, L1, c1), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), serial), [64, 512, 4096, 1073741824])
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t ba[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bb = 0; bb < 2; bb++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bc = 0; bc < 16; bc++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          ba[(2 * bc + bb)] = ab[(64 * bb + 2048 * ah + bc + 16 * ai)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ak), _mm_loadu_si128((__m256i *)(&ba[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bd[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int be = 0; be < 2; be++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bf = 0; bf < 16; bf++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bd[(2 * bf + be)] = ab[(64 * be + 2048 * ah + bf + 16 * ai + 128)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&al), _mm_loadu_si128((__m256i *)(&bd[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bg[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bh = 0; bh < 2; bh++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bi = 0; bi < 16; bi++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bg[(2 * bi + bh)] = ab[(64 * bh + 2048 * ah + bi + 16 * ai + 256)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&am), _mm_loadu_si128((__m256i *)(&bg[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bj[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bk = 0; bk < 2; bk++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bl = 0; bl < 16; bl++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bj[(2 * bl + bk)] = ab[(64 * bk + 2048 * ah + bl + 16 * ai + 384)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&an), _mm_loadu_si128((__m256i *)(&bj[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bm[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bn = 0; bn < 2; bn++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bo = 0; bo < 16; bo++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bm[(2 * bo + bn)] = ab[(64 * bn + 2048 * ah + bo + 16 * ai + 512)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ao), _mm_loadu_si128((__m256i *)(&bm[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bp[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bq = 0; bq < 2; bq++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int br = 0; br < 16; br++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bp[(2 * br + bq)] = ab[(64 * bq + 2048 * ah + br + 16 * ai + 640)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ap), _mm_loadu_si128((__m256i *)(&bp[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bs[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bt = 0; bt < 2; bt++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bu = 0; bu < 16; bu++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bs[(2 * bu + bt)] = ab[(64 * bt + 2048 * ah + bu + 16 * ai + 768)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&aq), _mm_loadu_si128((__m256i *)(&bs[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t bv[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bw = 0; bw < 2; bw++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int bx = 0; bx < 16; bx++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          bv[(2 * bx + bw)] = ab[(64 * bw + 2048 * ah + bx + 16 * ai + 896)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ar), _mm_loadu_si128((__m256i *)(&bv[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t by[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int bz = 0; bz < 2; bz++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int ca = 0; ca < 16; ca++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          by[(2 * ca + bz)] = ab[(64 * bz + 2048 * ah + ca + 16 * ai + 1024)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&as), _mm_loadu_si128((__m256i *)(&by[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t cb[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int cc = 0; cc < 2; cc++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cd = 0; cd < 16; cd++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          cb[(2 * cd + cc)] = ab[(64 * cc + 2048 * ah + cd + 16 * ai + 1152)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&at), _mm_loadu_si128((__m256i *)(&cb[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t ce[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int cf = 0; cf < 2; cf++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cg = 0; cg < 16; cg++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          ce[(2 * cg + cf)] = ab[(64 * cf + 2048 * ah + cg + 16 * ai + 1280)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&au), _mm_loadu_si128((__m256i *)(&ce[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t ch[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int ci = 0; ci < 2; ci++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cj = 0; cj < 16; cj++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          ch[(2 * cj + ci)] = ab[(64 * ci + 2048 * ah + cj + 16 * ai + 1408)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&av), _mm_loadu_si128((__m256i *)(&ch[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t ck[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int cl = 0; cl < 2; cl++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cm = 0; cm < 16; cm++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          ck[(2 * cm + cl)] = ab[(64 * cl + 2048 * ah + cm + 16 * ai + 1536)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&aw), _mm_loadu_si128((__m256i *)(&ck[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t cn[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int co = 0; co < 2; co++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cp = 0; cp < 16; cp++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          cn[(2 * cp + co)] = ab[(64 * co + 2048 * ah + cp + 16 * ai + 1664)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ax), _mm_loadu_si128((__m256i *)(&cn[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t cq[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int cr = 0; cr < 2; cr++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cs = 0; cs < 16; cs++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          cq[(2 * cs + cr)] = ab[(64 * cr + 2048 * ah + cs + 16 * ai + 1792)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&ay), _mm_loadu_si128((__m256i *)(&cq[(0)])));  /* VectorAssign */
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [64, 512, 4096, 1073741824])
      int8_t ct[32] __attribute__((aligned (128)));
      // (Move((2×16, s8, L1, c1, ua), (2×16, s8, RF, <[1,0], [None, None]>), serial), [32, 512, 4096, 1073741824])
      for (int cu = 0; cu < 2; cu++) {
        // (Move((1×16, s8, L1, ua), (1×16, s8, RF, <[1,0], [None, None]>, c1, ua), serial), [32, 512, 4096, 1073741824])
        for (int cv = 0; cv < 16; cv++) {
          // (Move((1×1, s8, L1, ua), (1×1, s8, RF, ua), serial), [32, 512, 4096, 1073741824])
          ct[(2 * cv + cu)] = ab[(64 * cu + 2048 * ah + cv + 16 * ai + 1920)];
        }
      }
      // (Move((2×16, s8, RF, <[1,0], [None, None]>), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), serial), [32, 512, 4096, 1073741824])
      _mm_storeu_si128((__m256i *)(&az), _mm_loadu_si128((__m256i *)(&ct[(0)])));  /* VectorAssign */
      // (MatmulAccum((64×32, u8, L1, c1), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (64×16, s16, L1, c1), serial), [64, 512, 4096, 1073741824])
      for (int cw = 0; cw < 64; cw++) {
        // (MatmulAccum((1×32, u8, L1, ua), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (1×16, s16, L1, ua), serial), [64, 512, 4096, 1073741824])
        vsi16 cy;
        // (Move((1×16, s16, L1, ua), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        _mm256_storeu_si256((__m256i *)(&cy), _mm256_loadu_si256((__m256i *)(ac + (64 * cw + 16 * ai))));  /* VectorAssign */
        // (MatmulAccum((1×32, u8, L1, ua), (32×16, s8, VRF, <[0,1,0], [None, None, Some(2)]>, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i cz = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(cz, ak));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i da = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 2)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(da, al));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i db = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 4)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(db, am));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dc = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 6)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dc, an));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dd = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 8)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dd, ao));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i de = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 10)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(de, ap));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i df = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 12)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(df, aq));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dg = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 14)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dg, ar));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dh = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 16)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dh, as));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i di = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 18)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(di, at));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dj = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 20)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dj, au));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dk = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 22)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dk, av));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dl = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 24)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dl, aw));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dm = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 26)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dm, ax));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dn = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 28)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dn, ay));
        // (MatmulAccum((1×2, u8, L1, ua), (2×16, s8, VRF, <[1,0], [None, None]>, ua, 32), (1×16, s16, VRF, 16), serial), [64, 256, 4096, 1073741824])
        __m256i dp = _mm256_set1_epi16(*(int16_t *)(aa + (64 * cw + 32 * ah + 30)));
        cy = _mm256_add_epi16(cy, _mm256_maddubs_epi16(dp, az));
        // (Move((1×16, s16, VRF, 16), (1×16, s16, L1, ua), serial), [64, 256, 4096, 1073741824])
        _mm256_storeu_si256((__m256i *)(ac + (64 * cw + 16 * ai)), _mm256_loadu_si256((__m256i *)(&cy)));  /* VectorAssign */
      }
    }
  }
}

int main(int argc, char *argv[]) {
  const char *inner_steps_env = getenv("CHERRYBENCH_LOOP_STEPS");
  if (inner_steps_env == NULL) {
    fprintf(stderr, "Environment variable CHERRYBENCH_LOOP_STEPS is not set.\n");
    exit(1);
  }
  const int inner_steps = atoi(inner_steps_env);

  uint8_t *restrict ag;
  posix_memalign((void **)&ag, 128, 4096*sizeof(uint8_t));
  memset(ag, rand(), 4096*sizeof(uint8_t));

  int8_t *restrict ah;
  posix_memalign((void **)&ah, 128, 4096*sizeof(int8_t));
  memset(ah, rand(), 4096*sizeof(int8_t));

  int16_t *restrict ai;
  posix_memalign((void **)&ai, 128, 4096*sizeof(int16_t));
  memset(ai, rand(), 4096*sizeof(int16_t));

  kernel(&ag[(0)], &ah[(0)], &ai[(0)]);
  for (unsigned int i = 0; i < 10; i++)
  {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned int j = 0; j < inner_steps; j++)
      kernel(&ag[(0)], &ah[(0)], &ai[(0)]);
    clock_gettime(CLOCK_MONOTONIC, &end);
    struct timespec delta = ts_diff(start, end);

    long elapsed_ns = delta.tv_sec * 1000000000L + delta.tv_nsec;
    printf("%ldns\n", elapsed_ns);
  }

  return 0;
}