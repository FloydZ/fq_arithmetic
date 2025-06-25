#pragma once 

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "arith.h"

/// NOTE: everything row major

static inline
gf127* gf127_matrix_alloc(const uint32_t nrows,
                          const uint32_t ncols) {
#ifdef USE_AVX2
    return (gf127 *)aligned_alloc(32, nrows * ncols * sizeof(gf127));
#else
    return (gf127 *)malloc(nrows * ncols * sizeof(gf127));
#endif
}

///
static inline
gf127 gf127_matrix_get(const gf127 *m,
                       const uint32_t nrows,
                       const uint32_t i,
                       const uint32_t j) {
    return m[j*nrows + i];
}

/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param a
static inline
void gf127_matrix_set(gf127 *m,
                      const uint32_t nrows,
                      const uint32_t i,
                      const uint32_t j,
                      const gf127 a) {
    m[j*nrows + i] = a;
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf127_matrix_zero(gf127 *m,
                       const uint32_t nrows,
                       const uint32_t ncols) {
    memset(m, 0, sizeof(gf127)*nrows*ncols);
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf127_matrix_rng(gf127 *m,
                      const uint32_t nrows,
                      const uint32_t ncols) {
    for(uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf127_matrix_set(m, nrows, i, j, rand() & 0x7F);
        }
    }
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf127_matrix_id(gf127 *m,
               const uint32_t nrows,
               const uint32_t ncols) {
    gf127_matrix_zero(m, nrows, ncols);
    for (int i = 0; i < nrows; ++i) {
        gf127_matrix_set(m, nrows, i, i, 1);
    }
}

/// \param in
/// \param nrows
/// \param ncols
static inline void gf127_matrix_print(const uint8_t *in,
                                      const uint32_t nrows,
                                      const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for(uint32_t j = 0; j < ncols; j++) {
            printf("%03d ", in[i*ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/// simple transpose, needed for debugging
static inline void gf127_matrix_transpose(uint8_t *out,
                                          const uint8_t *in,
                                          const uint32_t nrows,
                                          const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for(uint32_t j = 0; j < ncols; j++) {
            out[j*nrows + i] = in[i*ncols + j];
        }
    }
}

/// TODO: move cryptanalyslib
void gf127_matrix_transpose4x4(uint8_t* dst,
                               const uint8_t* src,
                               const size_t stride) {
    // load rows of src matrix
    uint32_t a0 = *((uint32_t*)(src+0*stride));
    uint32_t a1 = *((uint32_t*)(src+1*stride));
    uint32_t a2 = *((uint32_t*)(src+2*stride));
    uint32_t a3 = *((uint32_t*)(src+3*stride));
    
    // 2x2 block matrices
    uint32_t b0 = (a0 & 0x00ff00ffU) | ((a1 << 8) & 0xff00ff00U);
    uint32_t b1 = (a1 & 0xff00ff00U) | ((a0 >> 8) & 0x00ff00ffU);
    uint32_t b2 = (a2 & 0x00ff00ffU) | ((a3 << 8) & 0xff00ff00U);
    uint32_t b3 = (a3 & 0xff00ff00U) | ((a2 >> 8) & 0x00ff00ffU);
    
    // 4x4 block matrices
    uint32_t c0 = (b0 & 0x0000ffffU) | ((b2 << 16) & 0xffff0000U);
    uint32_t c1 = (b1 & 0x0000ffffU) | ((b3 << 16) & 0xffff0000U);
    uint32_t c2 = (b2 & 0xffff0000U) | ((b0 >> 16) & 0x0000ffffU);
    uint32_t c3 = (b3 & 0xffff0000U) | ((b1 >> 16) & 0x0000ffffU);
    
    // write to dst matrix
    *(uint32_t*)(dst + 0*stride) = c0;
    *(uint32_t*)(dst + 1*stride) = c1;
    *(uint32_t*)(dst + 2*stride) = c2;
    *(uint32_t*)(dst + 3*stride) = c3;
}

/// TODO: move cryptanalyslib
void gf127_matrix_transpose8x8(uint8_t* dst,
                               const uint8_t* src,
                               const size_t src_stride,
                               const size_t dst_stride) {
    // load rows of src matrix
    const uint64_t a0 = *((uint64_t*)(src+0*src_stride));
    const uint64_t a1 = *((uint64_t*)(src+1*src_stride));
    const uint64_t a2 = *((uint64_t*)(src+2*src_stride));
    const uint64_t a3 = *((uint64_t*)(src+3*src_stride));
    const uint64_t a4 = *((uint64_t*)(src+4*src_stride));
    const uint64_t a5 = *((uint64_t*)(src+5*src_stride));
    const uint64_t a6 = *((uint64_t*)(src+6*src_stride));
    const uint64_t a7 = *((uint64_t*)(src+7*src_stride));
    
    // 2x2 block matrices
    const uint64_t b0 = (a0 & 0x00ff00ff00ff00ffULL) | ((a1 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b1 = (a1 & 0xff00ff00ff00ff00ULL) | ((a0 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b2 = (a2 & 0x00ff00ff00ff00ffULL) | ((a3 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b3 = (a3 & 0xff00ff00ff00ff00ULL) | ((a2 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b4 = (a4 & 0x00ff00ff00ff00ffULL) | ((a5 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b5 = (a5 & 0xff00ff00ff00ff00ULL) | ((a4 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b6 = (a6 & 0x00ff00ff00ff00ffULL) | ((a7 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b7 = (a7 & 0xff00ff00ff00ff00ULL) | ((a6 >> 8) & 0x00ff00ff00ff00ffULL);
    
    // 4x4 block matrices
    const uint64_t c0 = (b0 & 0x0000ffff0000ffffULL) | ((b2 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c1 = (b1 & 0x0000ffff0000ffffULL) | ((b3 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c2 = (b2 & 0xffff0000ffff0000ULL) | ((b0 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c3 = (b3 & 0xffff0000ffff0000ULL) | ((b1 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c4 = (b4 & 0x0000ffff0000ffffULL) | ((b6 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c5 = (b5 & 0x0000ffff0000ffffULL) | ((b7 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c6 = (b6 & 0xffff0000ffff0000ULL) | ((b4 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c7 = (b7 & 0xffff0000ffff0000ULL) | ((b5 >> 16) & 0x0000ffff0000ffffULL);
    
    // 8x8 block matrix
    const uint64_t d0 = (c0 & 0x00000000ffffffffULL) | ((c4 << 32) & 0xffffffff00000000ULL);
    const uint64_t d1 = (c1 & 0x00000000ffffffffULL) | ((c5 << 32) & 0xffffffff00000000ULL);
    const uint64_t d2 = (c2 & 0x00000000ffffffffULL) | ((c6 << 32) & 0xffffffff00000000ULL);
    const uint64_t d3 = (c3 & 0x00000000ffffffffULL) | ((c7 << 32) & 0xffffffff00000000ULL);
    const uint64_t d4 = (c4 & 0xffffffff00000000ULL) | ((c0 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d5 = (c5 & 0xffffffff00000000ULL) | ((c1 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d6 = (c6 & 0xffffffff00000000ULL) | ((c2 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d7 = (c7 & 0xffffffff00000000ULL) | ((c3 >> 32) & 0x00000000ffffffffULL);
    
    // write to dst matrix
    *(uint64_t*)(dst + 0*dst_stride) = d0;
    *(uint64_t*)(dst + 1*dst_stride) = d1;
    *(uint64_t*)(dst + 2*dst_stride) = d2;
    *(uint64_t*)(dst + 3*dst_stride) = d3;
    *(uint64_t*)(dst + 4*dst_stride) = d4;
    *(uint64_t*)(dst + 5*dst_stride) = d5;
    *(uint64_t*)(dst + 6*dst_stride) = d6;
    *(uint64_t*)(dst + 7*dst_stride) = d7;
}

/// Compute origin of the 64-block next to (rb, cb) in row-major order
static inline
const uint8_t* gf127_next_block(const uint8_t *src,
                          const uint64_t rb,
                          const uint64_t cb,
                          const size_t n) {
    uint64_t cb1 = cb + 1;
    uint64_t rb1 = rb;
    if (cb1 == n/64) {
        rb1 += 1;
        cb1 = 0;
    }

    return src + (rb1*n + cb1) * 64;
}

/// \param dst
/// \param src
/// \param n
static inline
void gf127_matrix_transpose_opt(uint8_t *dst,
                                const uint8_t *src,
                                const size_t n) {
    const size_t bsize = 32;
    
    // iterate over 64x64 block matrices
    for (uint64_t rb = 0; rb < n / bsize; rb++) {
        for (uint64_t cb = 0; cb < n / bsize; cb++) {
            const uint8_t *srcb_origin = src + (rb * n + cb) * bsize;
                  uint8_t *dstb_origin = dst + (cb * n + rb) * bsize;
#ifdef __x86_64__
            const uint8_t *prfb_origin = gf127_next_block(src, rb, cb, n);
#endif
            
            for (size_t rw = 0; rw < bsize / 8; rw++) {
#ifdef __x86_64__
                // preload sizeof(W) rows of the next 64x64 block
                for (size_t i = rw * 8; i < (rw + 1) * 8; i++) {
                    _mm_prefetch(prfb_origin + i * n, _MM_HINT_NTA);
                }
#endif

                for (size_t cw = 0; cw < bsize / 8; cw++) {
                    const uint8_t *srcw_origin = srcb_origin + (cw * n + rw) * 8;
                          uint8_t *dstw_origin = dstb_origin + (rw * n + cw) * 8;
                    gf127_matrix_transpose8x8(dstw_origin, srcw_origin, n, n);
                }
            }
        }
    }
}


#ifdef USE_AVX2 
#include <immintrin.h>

/// needed for gf127_matrix_transpose_64x64_avx2
static const uint8_t BLENDV_MASK[5][32] __attribute__((aligned(32)))= {
    { 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff },
    { 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff },
    { 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff },
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
    { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
};

// needed for gf127_matrix_transpose_64x64_avx2
static const uint8_t SHUFFLE_MASK[4][32] __attribute__((aligned(32))) = {
    { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30 },
    { 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29 },
    { 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19, 28, 29, 30, 31, 24, 25, 26, 27 },
    { 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23 },
};

/// TODO probably just make two functions, one for aligned access and one for unaligned
typedef __m256i_u LOAD_TYPE;
typedef __m256i_u STORE_TYPE;


/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param prf_origin[in]: lookahead pointer to prefetch it
/// \param src_stride[in]:
/// \param dst_stride[in]:
void gf127_matrix_transpose_32x32_u256(uint8_t* dst_origin,
                                       const uint8_t* src_origin,
                                       const size_t src_stride,
                                       const size_t dst_stride) {
    static const uint32_t matrix_transpose_table[] __attribute__((aligned(32))) = {
        0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
    };

    __m256i t[32];
    for (uint32_t i = 0; i < 32; i++) {
        t[i] = _mm256_loadu_si256((const __m256i *)(src_origin + i*src_stride));
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=2) {
        const __m256i t0 = _mm256_unpacklo_epi8(t[i+0], t[i+1]);
        const __m256i t1 = _mm256_unpackhi_epi8(t[i+0], t[i+1]);
        t[i+0] = t0;
        t[i+1] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=4) {
        const __m256i t0 = _mm256_unpacklo_epi16(t[i+0], t[i+2]);
        const __m256i t1 = _mm256_unpacklo_epi16(t[i+1], t[i+3]);
        const __m256i t2 = _mm256_unpackhi_epi16(t[i+0], t[i+2]);
        const __m256i t3 = _mm256_unpackhi_epi16(t[i+1], t[i+3]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=8) {
        const __m256i t0 = _mm256_unpacklo_epi32(t[i+0], t[i+4]);
        const __m256i t1 = _mm256_unpacklo_epi32(t[i+1], t[i+5]);
        const __m256i t2 = _mm256_unpacklo_epi32(t[i+2], t[i+6]);
        const __m256i t3 = _mm256_unpacklo_epi32(t[i+3], t[i+7]);
        const __m256i t4 = _mm256_unpackhi_epi32(t[i+0], t[i+4]);
        const __m256i t5 = _mm256_unpackhi_epi32(t[i+1], t[i+5]);
        const __m256i t6 = _mm256_unpackhi_epi32(t[i+2], t[i+6]);
        const __m256i t7 = _mm256_unpackhi_epi32(t[i+3], t[i+7]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
        t[i+4] = t4;
        t[i+5] = t5;
        t[i+6] = t6;
        t[i+7] = t7;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        const __m256i t0 = _mm256_unpacklo_epi64(t[i+ 0], t[i+ 8]);
        const __m256i t1 = _mm256_unpackhi_epi64(t[i+ 0], t[i+ 8]);
        const __m256i t2 = _mm256_unpacklo_epi64(t[i+16], t[i+24]);
        const __m256i t3 = _mm256_unpackhi_epi64(t[i+16], t[i+24]);
        t[i+ 0] = t0;
        t[i+ 8] = t1;
        t[i+16] = t2;
        t[i+24] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const __m256i t0 = _mm256_permute2x128_si256(t[i+0], t[i+16], 0b100000);
        const __m256i t1 = _mm256_permute2x128_si256(t[i+0], t[i+16], 0b110001);
        t[i+ 0] = t0;
        t[i+16] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t pos = matrix_transpose_table[i];
        _mm256_storeu_si256((__m256i *)(dst_origin + i*dst_stride), t[pos]);
    }

    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t pos = matrix_transpose_table[i];
        _mm256_storeu_si256((__m256i *)(dst_origin + (i+16)*dst_stride), t[pos+16]);
    }
}

/// \param dst_origin
/// \param src_origin
/// \param prf_origin
/// \param src_stride
/// \param dst_stride
void gf127_matrix_transpose_32x32_u256_v2(uint8_t* dst_origin,
                                          const uint8_t* src_origin,
                                          const uint8_t* prf_origin,
                                          const size_t src_stride,
                                          const size_t dst_stride) {
  const __m256i shm_1 = _mm256_load_si256((const __m256i *)SHUFFLE_MASK[0]);
  const __m256i blm_1 = _mm256_load_si256((const __m256i *)BLENDV_MASK[0]);
  __m256i rnd_0_0 = *(const LOAD_TYPE*)(src_origin + 0*src_stride);
  __m256i rnd_0_1 = *(const LOAD_TYPE*)(src_origin + 1*src_stride);
  __m256i shf_1_0 = _mm256_shuffle_epi8(rnd_0_0, shm_1);
  __m256i shf_1_1 = _mm256_shuffle_epi8(rnd_0_1, shm_1);
  __m256i rnd_1_0 = _mm256_blendv_epi8(rnd_0_0, shf_1_1, blm_1);
  __m256i rnd_1_1 = _mm256_blendv_epi8(shf_1_0, rnd_0_1, blm_1);
  __m256i rnd_0_2 = *(const LOAD_TYPE*)(src_origin + 2*src_stride);
  __m256i rnd_0_3 = *(const LOAD_TYPE*)(src_origin + 3*src_stride);
  __m256i shf_1_2 = _mm256_shuffle_epi8(rnd_0_2, shm_1);
  __m256i shf_1_3 = _mm256_shuffle_epi8(rnd_0_3, shm_1);
  __m256i rnd_1_2 = _mm256_blendv_epi8(rnd_0_2, shf_1_3, blm_1);
  __m256i rnd_1_3 = _mm256_blendv_epi8(shf_1_2, rnd_0_3, blm_1);
  __m256i rnd_0_4 = *(const LOAD_TYPE*)(src_origin + 4*src_stride);
  __m256i rnd_0_5 = *(const LOAD_TYPE*)(src_origin + 5*src_stride);
  __m256i shf_1_4 = _mm256_shuffle_epi8(rnd_0_4, shm_1);
  __m256i shf_1_5 = _mm256_shuffle_epi8(rnd_0_5, shm_1);
  __m256i rnd_1_4 = _mm256_blendv_epi8(rnd_0_4, shf_1_5, blm_1);
  __m256i rnd_1_5 = _mm256_blendv_epi8(shf_1_4, rnd_0_5, blm_1);
  __m256i rnd_0_6 = *(const LOAD_TYPE*)(src_origin + 6*src_stride);
  __m256i rnd_0_7 = *(const LOAD_TYPE*)(src_origin + 7*src_stride);
  __m256i shf_1_6 = _mm256_shuffle_epi8(rnd_0_6, shm_1);
  __m256i shf_1_7 = _mm256_shuffle_epi8(rnd_0_7, shm_1);
  _mm_prefetch(prf_origin+0*src_stride, _MM_HINT_NTA);
  __m256i rnd_1_6 = _mm256_blendv_epi8(rnd_0_6, shf_1_7, blm_1);
  __m256i rnd_1_7 = _mm256_blendv_epi8(shf_1_6, rnd_0_7, blm_1);
  __m256i rnd_0_8 = *(const LOAD_TYPE*)(src_origin + 8*src_stride);
  __m256i rnd_0_9 = *(const LOAD_TYPE*)(src_origin + 9*src_stride);
  __m256i shf_1_8 = _mm256_shuffle_epi8(rnd_0_8, shm_1);
  __m256i shf_1_9 = _mm256_shuffle_epi8(rnd_0_9, shm_1);
  __m256i rnd_1_8 = _mm256_blendv_epi8(rnd_0_8, shf_1_9, blm_1);
  __m256i rnd_1_9 = _mm256_blendv_epi8(shf_1_8, rnd_0_9, blm_1);
  __m256i rnd_0_10 = *(const LOAD_TYPE*)(src_origin + 10*src_stride);
  __m256i rnd_0_11 = *(const LOAD_TYPE*)(src_origin + 11*src_stride);
  __m256i shf_1_10 = _mm256_shuffle_epi8(rnd_0_10, shm_1);
  __m256i shf_1_11 = _mm256_shuffle_epi8(rnd_0_11, shm_1);
  __m256i rnd_1_10 = _mm256_blendv_epi8(rnd_0_10, shf_1_11, blm_1);
  __m256i rnd_1_11 = _mm256_blendv_epi8(shf_1_10, rnd_0_11, blm_1);
  __m256i rnd_0_12 = *(const LOAD_TYPE*)(src_origin + 12*src_stride);
  __m256i rnd_0_13 = *(const LOAD_TYPE*)(src_origin + 13*src_stride);
  __m256i shf_1_12 = _mm256_shuffle_epi8(rnd_0_12, shm_1);
  __m256i shf_1_13 = _mm256_shuffle_epi8(rnd_0_13, shm_1);
  __m256i rnd_1_12 = _mm256_blendv_epi8(rnd_0_12, shf_1_13, blm_1);
  __m256i rnd_1_13 = _mm256_blendv_epi8(shf_1_12, rnd_0_13, blm_1);
  __m256i rnd_0_14 = *(const LOAD_TYPE*)(src_origin + 14*src_stride);
  __m256i rnd_0_15 = *(const LOAD_TYPE*)(src_origin + 15*src_stride);
  __m256i shf_1_14 = _mm256_shuffle_epi8(rnd_0_14, shm_1);
  __m256i shf_1_15 = _mm256_shuffle_epi8(rnd_0_15, shm_1);
  _mm_prefetch(prf_origin+1*src_stride, _MM_HINT_NTA);
  __m256i rnd_1_14 = _mm256_blendv_epi8(rnd_0_14, shf_1_15, blm_1);
  __m256i rnd_1_15 = _mm256_blendv_epi8(shf_1_14, rnd_0_15, blm_1);
  __m256i rnd_0_16 = *(const LOAD_TYPE*)(src_origin + 16*src_stride);
  __m256i rnd_0_17 = *(const LOAD_TYPE*)(src_origin + 17*src_stride);
  __m256i shf_1_16 = _mm256_shuffle_epi8(rnd_0_16, shm_1);
  __m256i shf_1_17 = _mm256_shuffle_epi8(rnd_0_17, shm_1);
  __m256i rnd_1_16 = _mm256_blendv_epi8(rnd_0_16, shf_1_17, blm_1);
  __m256i rnd_1_17 = _mm256_blendv_epi8(shf_1_16, rnd_0_17, blm_1);
  __m256i rnd_0_18 = *(const LOAD_TYPE*)(src_origin + 18*src_stride);
  __m256i rnd_0_19 = *(const LOAD_TYPE*)(src_origin + 19*src_stride);
  __m256i shf_1_18 = _mm256_shuffle_epi8(rnd_0_18, shm_1);
  __m256i shf_1_19 = _mm256_shuffle_epi8(rnd_0_19, shm_1);
  __m256i rnd_1_18 = _mm256_blendv_epi8(rnd_0_18, shf_1_19, blm_1);
  __m256i rnd_1_19 = _mm256_blendv_epi8(shf_1_18, rnd_0_19, blm_1);
  __m256i rnd_0_20 = *(const LOAD_TYPE*)(src_origin + 20*src_stride);
  __m256i rnd_0_21 = *(const LOAD_TYPE*)(src_origin + 21*src_stride);
  __m256i shf_1_20 = _mm256_shuffle_epi8(rnd_0_20, shm_1);
  __m256i shf_1_21 = _mm256_shuffle_epi8(rnd_0_21, shm_1);
  __m256i rnd_1_20 = _mm256_blendv_epi8(rnd_0_20, shf_1_21, blm_1);
  __m256i rnd_1_21 = _mm256_blendv_epi8(shf_1_20, rnd_0_21, blm_1);
  __m256i rnd_0_22 = *(const LOAD_TYPE*)(src_origin + 22*src_stride);
  __m256i rnd_0_23 = *(const LOAD_TYPE*)(src_origin + 23*src_stride);
  __m256i shf_1_22 = _mm256_shuffle_epi8(rnd_0_22, shm_1);
  __m256i shf_1_23 = _mm256_shuffle_epi8(rnd_0_23, shm_1);
  _mm_prefetch(prf_origin+2*src_stride, _MM_HINT_NTA);
  __m256i rnd_1_22 = _mm256_blendv_epi8(rnd_0_22, shf_1_23, blm_1);
  __m256i rnd_1_23 = _mm256_blendv_epi8(shf_1_22, rnd_0_23, blm_1);
  __m256i rnd_0_24 = *(const LOAD_TYPE*)(src_origin + 24*src_stride);
  __m256i rnd_0_25 = *(const LOAD_TYPE*)(src_origin + 25*src_stride);
  __m256i shf_1_24 = _mm256_shuffle_epi8(rnd_0_24, shm_1);
  __m256i shf_1_25 = _mm256_shuffle_epi8(rnd_0_25, shm_1);
  __m256i rnd_1_24 = _mm256_blendv_epi8(rnd_0_24, shf_1_25, blm_1);
  __m256i rnd_1_25 = _mm256_blendv_epi8(shf_1_24, rnd_0_25, blm_1);
  __m256i rnd_0_26 = *(const LOAD_TYPE*)(src_origin + 26*src_stride);
  __m256i rnd_0_27 = *(const LOAD_TYPE*)(src_origin + 27*src_stride);
  __m256i shf_1_26 = _mm256_shuffle_epi8(rnd_0_26, shm_1);
  __m256i shf_1_27 = _mm256_shuffle_epi8(rnd_0_27, shm_1);
  __m256i rnd_1_26 = _mm256_blendv_epi8(rnd_0_26, shf_1_27, blm_1);
  __m256i rnd_1_27 = _mm256_blendv_epi8(shf_1_26, rnd_0_27, blm_1);
  __m256i rnd_0_28 = *(const LOAD_TYPE*)(src_origin + 28*src_stride);
  __m256i rnd_0_29 = *(const LOAD_TYPE*)(src_origin + 29*src_stride);
  __m256i shf_1_28 = _mm256_shuffle_epi8(rnd_0_28, shm_1);
  __m256i shf_1_29 = _mm256_shuffle_epi8(rnd_0_29, shm_1);
  __m256i rnd_1_28 = _mm256_blendv_epi8(rnd_0_28, shf_1_29, blm_1);
  __m256i rnd_1_29 = _mm256_blendv_epi8(shf_1_28, rnd_0_29, blm_1);
  __m256i rnd_0_30 = *(const LOAD_TYPE*)(src_origin + 30*src_stride);
  __m256i rnd_0_31 = *(const LOAD_TYPE*)(src_origin + 31*src_stride);
  __m256i shf_1_30 = _mm256_shuffle_epi8(rnd_0_30, shm_1);
  __m256i shf_1_31 = _mm256_shuffle_epi8(rnd_0_31, shm_1);
  _mm_prefetch(prf_origin+3*src_stride, _MM_HINT_NTA);
  __m256i rnd_1_30 = _mm256_blendv_epi8(rnd_0_30, shf_1_31, blm_1);
  __m256i rnd_1_31 = _mm256_blendv_epi8(shf_1_30, rnd_0_31, blm_1);
  const __m256i shm_2 = _mm256_load_si256((const __m256i *)SHUFFLE_MASK[1]);
  const __m256i blm_2 = _mm256_load_si256((const __m256i *)BLENDV_MASK[1]);
  __m256i shf_2_0 = _mm256_shuffle_epi8(rnd_1_0, shm_2);
  __m256i shf_2_2 = _mm256_shuffle_epi8(rnd_1_2, shm_2);
  __m256i rnd_2_0 = _mm256_blendv_epi8(rnd_1_0, shf_2_2, blm_2);
  __m256i rnd_2_2 = _mm256_blendv_epi8(shf_2_0, rnd_1_2, blm_2);
  __m256i shf_2_1 = _mm256_shuffle_epi8(rnd_1_1, shm_2);
  __m256i shf_2_3 = _mm256_shuffle_epi8(rnd_1_3, shm_2);
  __m256i rnd_2_1 = _mm256_blendv_epi8(rnd_1_1, shf_2_3, blm_2);
  __m256i rnd_2_3 = _mm256_blendv_epi8(shf_2_1, rnd_1_3, blm_2);
  __m256i shf_2_4 = _mm256_shuffle_epi8(rnd_1_4, shm_2);
  __m256i shf_2_6 = _mm256_shuffle_epi8(rnd_1_6, shm_2);
  __m256i rnd_2_4 = _mm256_blendv_epi8(rnd_1_4, shf_2_6, blm_2);
  __m256i rnd_2_6 = _mm256_blendv_epi8(shf_2_4, rnd_1_6, blm_2);
  __m256i shf_2_5 = _mm256_shuffle_epi8(rnd_1_5, shm_2);
  __m256i shf_2_7 = _mm256_shuffle_epi8(rnd_1_7, shm_2);
  __m256i rnd_2_5 = _mm256_blendv_epi8(rnd_1_5, shf_2_7, blm_2);
  __m256i rnd_2_7 = _mm256_blendv_epi8(shf_2_5, rnd_1_7, blm_2);
  __m256i shf_2_8 = _mm256_shuffle_epi8(rnd_1_8, shm_2);
  __m256i shf_2_10 = _mm256_shuffle_epi8(rnd_1_10, shm_2);
  __m256i rnd_2_8 = _mm256_blendv_epi8(rnd_1_8, shf_2_10, blm_2);
  __m256i rnd_2_10 = _mm256_blendv_epi8(shf_2_8, rnd_1_10, blm_2);
  _mm_prefetch(prf_origin+4*src_stride, _MM_HINT_NTA);
  __m256i shf_2_9 = _mm256_shuffle_epi8(rnd_1_9, shm_2);
  __m256i shf_2_11 = _mm256_shuffle_epi8(rnd_1_11, shm_2);
  __m256i rnd_2_9 = _mm256_blendv_epi8(rnd_1_9, shf_2_11, blm_2);
  __m256i rnd_2_11 = _mm256_blendv_epi8(shf_2_9, rnd_1_11, blm_2);
  __m256i shf_2_12 = _mm256_shuffle_epi8(rnd_1_12, shm_2);
  __m256i shf_2_14 = _mm256_shuffle_epi8(rnd_1_14, shm_2);
  __m256i rnd_2_12 = _mm256_blendv_epi8(rnd_1_12, shf_2_14, blm_2);
  __m256i rnd_2_14 = _mm256_blendv_epi8(shf_2_12, rnd_1_14, blm_2);
  __m256i shf_2_13 = _mm256_shuffle_epi8(rnd_1_13, shm_2);
  __m256i shf_2_15 = _mm256_shuffle_epi8(rnd_1_15, shm_2);
  __m256i rnd_2_13 = _mm256_blendv_epi8(rnd_1_13, shf_2_15, blm_2);
  __m256i rnd_2_15 = _mm256_blendv_epi8(shf_2_13, rnd_1_15, blm_2);
  __m256i shf_2_16 = _mm256_shuffle_epi8(rnd_1_16, shm_2);
  __m256i shf_2_18 = _mm256_shuffle_epi8(rnd_1_18, shm_2);
  __m256i rnd_2_16 = _mm256_blendv_epi8(rnd_1_16, shf_2_18, blm_2);
  __m256i rnd_2_18 = _mm256_blendv_epi8(shf_2_16, rnd_1_18, blm_2);
  __m256i shf_2_17 = _mm256_shuffle_epi8(rnd_1_17, shm_2);
  __m256i shf_2_19 = _mm256_shuffle_epi8(rnd_1_19, shm_2);
  __m256i rnd_2_17 = _mm256_blendv_epi8(rnd_1_17, shf_2_19, blm_2);
  __m256i rnd_2_19 = _mm256_blendv_epi8(shf_2_17, rnd_1_19, blm_2);
  __m256i shf_2_20 = _mm256_shuffle_epi8(rnd_1_20, shm_2);
  __m256i shf_2_22 = _mm256_shuffle_epi8(rnd_1_22, shm_2);
  __m256i rnd_2_20 = _mm256_blendv_epi8(rnd_1_20, shf_2_22, blm_2);
  __m256i rnd_2_22 = _mm256_blendv_epi8(shf_2_20, rnd_1_22, blm_2);
  _mm_prefetch(prf_origin+5*src_stride, _MM_HINT_NTA);
  __m256i shf_2_21 = _mm256_shuffle_epi8(rnd_1_21, shm_2);
  __m256i shf_2_23 = _mm256_shuffle_epi8(rnd_1_23, shm_2);
  __m256i rnd_2_21 = _mm256_blendv_epi8(rnd_1_21, shf_2_23, blm_2);
  __m256i rnd_2_23 = _mm256_blendv_epi8(shf_2_21, rnd_1_23, blm_2);
  __m256i shf_2_24 = _mm256_shuffle_epi8(rnd_1_24, shm_2);
  __m256i shf_2_26 = _mm256_shuffle_epi8(rnd_1_26, shm_2);
  __m256i rnd_2_24 = _mm256_blendv_epi8(rnd_1_24, shf_2_26, blm_2);
  __m256i rnd_2_26 = _mm256_blendv_epi8(shf_2_24, rnd_1_26, blm_2);
  __m256i shf_2_25 = _mm256_shuffle_epi8(rnd_1_25, shm_2);
  __m256i shf_2_27 = _mm256_shuffle_epi8(rnd_1_27, shm_2);
  __m256i rnd_2_25 = _mm256_blendv_epi8(rnd_1_25, shf_2_27, blm_2);
  __m256i rnd_2_27 = _mm256_blendv_epi8(shf_2_25, rnd_1_27, blm_2);
  __m256i shf_2_28 = _mm256_shuffle_epi8(rnd_1_28, shm_2);
  __m256i shf_2_30 = _mm256_shuffle_epi8(rnd_1_30, shm_2);
  __m256i rnd_2_28 = _mm256_blendv_epi8(rnd_1_28, shf_2_30, blm_2);
  __m256i rnd_2_30 = _mm256_blendv_epi8(shf_2_28, rnd_1_30, blm_2);
  __m256i shf_2_29 = _mm256_shuffle_epi8(rnd_1_29, shm_2);
  __m256i shf_2_31 = _mm256_shuffle_epi8(rnd_1_31, shm_2);
  __m256i rnd_2_29 = _mm256_blendv_epi8(rnd_1_29, shf_2_31, blm_2);
  __m256i rnd_2_31 = _mm256_blendv_epi8(shf_2_29, rnd_1_31, blm_2);
  const __m256i shm_3 = _mm256_load_si256((const __m256i *)SHUFFLE_MASK[2]);
  const __m256i blm_3 = _mm256_load_si256((const __m256i *)BLENDV_MASK[2]);
  __m256i shf_3_0 = _mm256_shuffle_epi8(rnd_2_0, shm_3);
  __m256i shf_3_4 = _mm256_shuffle_epi8(rnd_2_4, shm_3);
  _mm_prefetch(prf_origin+6*src_stride, _MM_HINT_NTA);
  __m256i rnd_3_0 = _mm256_blendv_epi8(rnd_2_0, shf_3_4, blm_3);
  __m256i rnd_3_4 = _mm256_blendv_epi8(shf_3_0, rnd_2_4, blm_3);
  __m256i shf_3_1 = _mm256_shuffle_epi8(rnd_2_1, shm_3);
  __m256i shf_3_5 = _mm256_shuffle_epi8(rnd_2_5, shm_3);
  __m256i rnd_3_1 = _mm256_blendv_epi8(rnd_2_1, shf_3_5, blm_3);
  __m256i rnd_3_5 = _mm256_blendv_epi8(shf_3_1, rnd_2_5, blm_3);
  __m256i shf_3_2 = _mm256_shuffle_epi8(rnd_2_2, shm_3);
  __m256i shf_3_6 = _mm256_shuffle_epi8(rnd_2_6, shm_3);
  __m256i rnd_3_2 = _mm256_blendv_epi8(rnd_2_2, shf_3_6, blm_3);
  __m256i rnd_3_6 = _mm256_blendv_epi8(shf_3_2, rnd_2_6, blm_3);
  __m256i shf_3_3 = _mm256_shuffle_epi8(rnd_2_3, shm_3);
  __m256i shf_3_7 = _mm256_shuffle_epi8(rnd_2_7, shm_3);
  __m256i rnd_3_3 = _mm256_blendv_epi8(rnd_2_3, shf_3_7, blm_3);
  __m256i rnd_3_7 = _mm256_blendv_epi8(shf_3_3, rnd_2_7, blm_3);
  __m256i shf_3_8 = _mm256_shuffle_epi8(rnd_2_8, shm_3);
  __m256i shf_3_12 = _mm256_shuffle_epi8(rnd_2_12, shm_3);
  __m256i rnd_3_8 = _mm256_blendv_epi8(rnd_2_8, shf_3_12, blm_3);
  __m256i rnd_3_12 = _mm256_blendv_epi8(shf_3_8, rnd_2_12, blm_3);
  __m256i shf_3_9 = _mm256_shuffle_epi8(rnd_2_9, shm_3);
  __m256i shf_3_13 = _mm256_shuffle_epi8(rnd_2_13, shm_3);
  __m256i rnd_3_9 = _mm256_blendv_epi8(rnd_2_9, shf_3_13, blm_3);
  __m256i rnd_3_13 = _mm256_blendv_epi8(shf_3_9, rnd_2_13, blm_3);
  __m256i shf_3_10 = _mm256_shuffle_epi8(rnd_2_10, shm_3);
  __m256i shf_3_14 = _mm256_shuffle_epi8(rnd_2_14, shm_3);
  _mm_prefetch(prf_origin+7*src_stride, _MM_HINT_NTA);
  __m256i rnd_3_10 = _mm256_blendv_epi8(rnd_2_10, shf_3_14, blm_3);
  __m256i rnd_3_14 = _mm256_blendv_epi8(shf_3_10, rnd_2_14, blm_3);
  __m256i shf_3_11 = _mm256_shuffle_epi8(rnd_2_11, shm_3);
  __m256i shf_3_15 = _mm256_shuffle_epi8(rnd_2_15, shm_3);
  __m256i rnd_3_11 = _mm256_blendv_epi8(rnd_2_11, shf_3_15, blm_3);
  __m256i rnd_3_15 = _mm256_blendv_epi8(shf_3_11, rnd_2_15, blm_3);
  __m256i shf_3_16 = _mm256_shuffle_epi8(rnd_2_16, shm_3);
  __m256i shf_3_20 = _mm256_shuffle_epi8(rnd_2_20, shm_3);
  __m256i rnd_3_16 = _mm256_blendv_epi8(rnd_2_16, shf_3_20, blm_3);
  __m256i rnd_3_20 = _mm256_blendv_epi8(shf_3_16, rnd_2_20, blm_3);
  __m256i shf_3_17 = _mm256_shuffle_epi8(rnd_2_17, shm_3);
  __m256i shf_3_21 = _mm256_shuffle_epi8(rnd_2_21, shm_3);
  __m256i rnd_3_17 = _mm256_blendv_epi8(rnd_2_17, shf_3_21, blm_3);
  __m256i rnd_3_21 = _mm256_blendv_epi8(shf_3_17, rnd_2_21, blm_3);
  __m256i shf_3_18 = _mm256_shuffle_epi8(rnd_2_18, shm_3);
  __m256i shf_3_22 = _mm256_shuffle_epi8(rnd_2_22, shm_3);
  __m256i rnd_3_18 = _mm256_blendv_epi8(rnd_2_18, shf_3_22, blm_3);
  __m256i rnd_3_22 = _mm256_blendv_epi8(shf_3_18, rnd_2_22, blm_3);
  __m256i shf_3_19 = _mm256_shuffle_epi8(rnd_2_19, shm_3);
  __m256i shf_3_23 = _mm256_shuffle_epi8(rnd_2_23, shm_3);
  __m256i rnd_3_19 = _mm256_blendv_epi8(rnd_2_19, shf_3_23, blm_3);
  __m256i rnd_3_23 = _mm256_blendv_epi8(shf_3_19, rnd_2_23, blm_3);
  __m256i shf_3_24 = _mm256_shuffle_epi8(rnd_2_24, shm_3);
  __m256i shf_3_28 = _mm256_shuffle_epi8(rnd_2_28, shm_3);
  _mm_prefetch(prf_origin+8*src_stride, _MM_HINT_NTA);
  __m256i rnd_3_24 = _mm256_blendv_epi8(rnd_2_24, shf_3_28, blm_3);
  __m256i rnd_3_28 = _mm256_blendv_epi8(shf_3_24, rnd_2_28, blm_3);
  __m256i shf_3_25 = _mm256_shuffle_epi8(rnd_2_25, shm_3);
  __m256i shf_3_29 = _mm256_shuffle_epi8(rnd_2_29, shm_3);
  __m256i rnd_3_25 = _mm256_blendv_epi8(rnd_2_25, shf_3_29, blm_3);
  __m256i rnd_3_29 = _mm256_blendv_epi8(shf_3_25, rnd_2_29, blm_3);
  __m256i shf_3_26 = _mm256_shuffle_epi8(rnd_2_26, shm_3);
  __m256i shf_3_30 = _mm256_shuffle_epi8(rnd_2_30, shm_3);
  __m256i rnd_3_26 = _mm256_blendv_epi8(rnd_2_26, shf_3_30, blm_3);
  __m256i rnd_3_30 = _mm256_blendv_epi8(shf_3_26, rnd_2_30, blm_3);
  __m256i shf_3_27 = _mm256_shuffle_epi8(rnd_2_27, shm_3);
  __m256i shf_3_31 = _mm256_shuffle_epi8(rnd_2_31, shm_3);
  __m256i rnd_3_27 = _mm256_blendv_epi8(rnd_2_27, shf_3_31, blm_3);
  __m256i rnd_3_31 = _mm256_blendv_epi8(shf_3_27, rnd_2_31, blm_3);
  const __m256i shm_4 = _mm256_load_si256((const __m256i *)SHUFFLE_MASK[3]);
  const __m256i blm_4 = _mm256_load_si256((const __m256i *)BLENDV_MASK[3]);
  __m256i shf_4_0 = _mm256_shuffle_epi8(rnd_3_0, shm_4);
  __m256i shf_4_8 = _mm256_shuffle_epi8(rnd_3_8, shm_4);
  __m256i rnd_4_0 = _mm256_blendv_epi8(rnd_3_0, shf_4_8, blm_4);
  __m256i rnd_4_8 = _mm256_blendv_epi8(shf_4_0, rnd_3_8, blm_4);
  __m256i shf_4_1 = _mm256_shuffle_epi8(rnd_3_1, shm_4);
  __m256i shf_4_9 = _mm256_shuffle_epi8(rnd_3_9, shm_4);
  __m256i rnd_4_1 = _mm256_blendv_epi8(rnd_3_1, shf_4_9, blm_4);
  __m256i rnd_4_9 = _mm256_blendv_epi8(shf_4_1, rnd_3_9, blm_4);
  _mm_prefetch(prf_origin+9*src_stride, _MM_HINT_NTA);
  __m256i shf_4_2 = _mm256_shuffle_epi8(rnd_3_2, shm_4);
  __m256i shf_4_10 = _mm256_shuffle_epi8(rnd_3_10, shm_4);
  __m256i rnd_4_2 = _mm256_blendv_epi8(rnd_3_2, shf_4_10, blm_4);
  __m256i rnd_4_10 = _mm256_blendv_epi8(shf_4_2, rnd_3_10, blm_4);
  __m256i shf_4_3 = _mm256_shuffle_epi8(rnd_3_3, shm_4);
  __m256i shf_4_11 = _mm256_shuffle_epi8(rnd_3_11, shm_4);
  __m256i rnd_4_3 = _mm256_blendv_epi8(rnd_3_3, shf_4_11, blm_4);
  __m256i rnd_4_11 = _mm256_blendv_epi8(shf_4_3, rnd_3_11, blm_4);
  __m256i shf_4_4 = _mm256_shuffle_epi8(rnd_3_4, shm_4);
  __m256i shf_4_12 = _mm256_shuffle_epi8(rnd_3_12, shm_4);
  __m256i rnd_4_4 = _mm256_blendv_epi8(rnd_3_4, shf_4_12, blm_4);
  __m256i rnd_4_12 = _mm256_blendv_epi8(shf_4_4, rnd_3_12, blm_4);
  __m256i shf_4_5 = _mm256_shuffle_epi8(rnd_3_5, shm_4);
  __m256i shf_4_13 = _mm256_shuffle_epi8(rnd_3_13, shm_4);
  __m256i rnd_4_5 = _mm256_blendv_epi8(rnd_3_5, shf_4_13, blm_4);
  __m256i rnd_4_13 = _mm256_blendv_epi8(shf_4_5, rnd_3_13, blm_4);
  __m256i shf_4_6 = _mm256_shuffle_epi8(rnd_3_6, shm_4);
  __m256i shf_4_14 = _mm256_shuffle_epi8(rnd_3_14, shm_4);
  __m256i rnd_4_6 = _mm256_blendv_epi8(rnd_3_6, shf_4_14, blm_4);
  __m256i rnd_4_14 = _mm256_blendv_epi8(shf_4_6, rnd_3_14, blm_4);
  __m256i shf_4_7 = _mm256_shuffle_epi8(rnd_3_7, shm_4);
  __m256i shf_4_15 = _mm256_shuffle_epi8(rnd_3_15, shm_4);
  __m256i rnd_4_7 = _mm256_blendv_epi8(rnd_3_7, shf_4_15, blm_4);
  __m256i rnd_4_15 = _mm256_blendv_epi8(shf_4_7, rnd_3_15, blm_4);
  _mm_prefetch(prf_origin+10*src_stride, _MM_HINT_NTA);
  __m256i shf_4_16 = _mm256_shuffle_epi8(rnd_3_16, shm_4);
  __m256i shf_4_24 = _mm256_shuffle_epi8(rnd_3_24, shm_4);
  __m256i rnd_4_16 = _mm256_blendv_epi8(rnd_3_16, shf_4_24, blm_4);
  __m256i rnd_4_24 = _mm256_blendv_epi8(shf_4_16, rnd_3_24, blm_4);
  __m256i shf_4_17 = _mm256_shuffle_epi8(rnd_3_17, shm_4);
  __m256i shf_4_25 = _mm256_shuffle_epi8(rnd_3_25, shm_4);
  __m256i rnd_4_17 = _mm256_blendv_epi8(rnd_3_17, shf_4_25, blm_4);
  __m256i rnd_4_25 = _mm256_blendv_epi8(shf_4_17, rnd_3_25, blm_4);
  __m256i shf_4_18 = _mm256_shuffle_epi8(rnd_3_18, shm_4);
  __m256i shf_4_26 = _mm256_shuffle_epi8(rnd_3_26, shm_4);
  __m256i rnd_4_18 = _mm256_blendv_epi8(rnd_3_18, shf_4_26, blm_4);
  __m256i rnd_4_26 = _mm256_blendv_epi8(shf_4_18, rnd_3_26, blm_4);
  __m256i shf_4_19 = _mm256_shuffle_epi8(rnd_3_19, shm_4);
  __m256i shf_4_27 = _mm256_shuffle_epi8(rnd_3_27, shm_4);
  __m256i rnd_4_19 = _mm256_blendv_epi8(rnd_3_19, shf_4_27, blm_4);
  __m256i rnd_4_27 = _mm256_blendv_epi8(shf_4_19, rnd_3_27, blm_4);
  __m256i shf_4_20 = _mm256_shuffle_epi8(rnd_3_20, shm_4);
  __m256i shf_4_28 = _mm256_shuffle_epi8(rnd_3_28, shm_4);
  __m256i rnd_4_20 = _mm256_blendv_epi8(rnd_3_20, shf_4_28, blm_4);
  __m256i rnd_4_28 = _mm256_blendv_epi8(shf_4_20, rnd_3_28, blm_4);
  __m256i shf_4_21 = _mm256_shuffle_epi8(rnd_3_21, shm_4);
  __m256i shf_4_29 = _mm256_shuffle_epi8(rnd_3_29, shm_4);
  __m256i rnd_4_21 = _mm256_blendv_epi8(rnd_3_21, shf_4_29, blm_4);
  __m256i rnd_4_29 = _mm256_blendv_epi8(shf_4_21, rnd_3_29, blm_4);
  _mm_prefetch(prf_origin+11*src_stride, _MM_HINT_NTA);
  __m256i shf_4_22 = _mm256_shuffle_epi8(rnd_3_22, shm_4);
  __m256i shf_4_30 = _mm256_shuffle_epi8(rnd_3_30, shm_4);
  __m256i rnd_4_22 = _mm256_blendv_epi8(rnd_3_22, shf_4_30, blm_4);
  __m256i rnd_4_30 = _mm256_blendv_epi8(shf_4_22, rnd_3_30, blm_4);
  __m256i shf_4_23 = _mm256_shuffle_epi8(rnd_3_23, shm_4);
  __m256i shf_4_31 = _mm256_shuffle_epi8(rnd_3_31, shm_4);
  __m256i rnd_4_23 = _mm256_blendv_epi8(rnd_3_23, shf_4_31, blm_4);
  __m256i rnd_4_31 = _mm256_blendv_epi8(shf_4_23, rnd_3_31, blm_4);
  const __m256i blm_5 = _mm256_load_si256((const __m256i *)BLENDV_MASK[4]);
  __m256i shf_5_0 = _mm256_permute2x128_si256(rnd_4_0, rnd_4_0, 0x01);
  __m256i shf_5_16 = _mm256_permute2x128_si256(rnd_4_16, rnd_4_16, 0x01);
  __m256i rnd_5_0 = _mm256_blendv_epi8(rnd_4_0, shf_5_16, blm_5);
  __m256i rnd_5_16 = _mm256_blendv_epi8(shf_5_0, rnd_4_16, blm_5);
  *(STORE_TYPE*)(dst_origin + 0*dst_stride) = rnd_5_0;
  *(STORE_TYPE*)(dst_origin + 16*dst_stride) = rnd_5_16;
  __m256i shf_5_1 = _mm256_permute2x128_si256(rnd_4_1, rnd_4_1, 0x01);
  __m256i shf_5_17 = _mm256_permute2x128_si256(rnd_4_17, rnd_4_17, 0x01);
  __m256i rnd_5_1 = _mm256_blendv_epi8(rnd_4_1, shf_5_17, blm_5);
  __m256i rnd_5_17 = _mm256_blendv_epi8(shf_5_1, rnd_4_17, blm_5);
  *(STORE_TYPE*)(dst_origin + 1*dst_stride) = rnd_5_1;
  *(STORE_TYPE*)(dst_origin + 17*dst_stride) = rnd_5_17;
  __m256i shf_5_2 = _mm256_permute2x128_si256(rnd_4_2, rnd_4_2, 0x01);
  __m256i shf_5_18 = _mm256_permute2x128_si256(rnd_4_18, rnd_4_18, 0x01);
  __m256i rnd_5_2 = _mm256_blendv_epi8(rnd_4_2, shf_5_18, blm_5);
  _mm_prefetch(prf_origin+12*src_stride, _MM_HINT_NTA);
  __m256i rnd_5_18 = _mm256_blendv_epi8(shf_5_2, rnd_4_18, blm_5);
  *(STORE_TYPE*)(dst_origin + 2*dst_stride) = rnd_5_2;
  *(STORE_TYPE*)(dst_origin + 18*dst_stride) = rnd_5_18;
  __m256i shf_5_3 = _mm256_permute2x128_si256(rnd_4_3, rnd_4_3, 0x01);
  __m256i shf_5_19 = _mm256_permute2x128_si256(rnd_4_19, rnd_4_19, 0x01);
  __m256i rnd_5_3 = _mm256_blendv_epi8(rnd_4_3, shf_5_19, blm_5);
  __m256i rnd_5_19 = _mm256_blendv_epi8(shf_5_3, rnd_4_19, blm_5);
  *(STORE_TYPE*)(dst_origin + 3*dst_stride) = rnd_5_3;
  *(STORE_TYPE*)(dst_origin + 19*dst_stride) = rnd_5_19;
  __m256i shf_5_4 = _mm256_permute2x128_si256(rnd_4_4, rnd_4_4, 0x01);
  __m256i shf_5_20 = _mm256_permute2x128_si256(rnd_4_20, rnd_4_20, 0x01);
  __m256i rnd_5_4 = _mm256_blendv_epi8(rnd_4_4, shf_5_20, blm_5);
  __m256i rnd_5_20 = _mm256_blendv_epi8(shf_5_4, rnd_4_20, blm_5);
  *(STORE_TYPE*)(dst_origin + 4*dst_stride) = rnd_5_4;
  *(STORE_TYPE*)(dst_origin + 20*dst_stride) = rnd_5_20;
  __m256i shf_5_5 = _mm256_permute2x128_si256(rnd_4_5, rnd_4_5, 0x01);
  __m256i shf_5_21 = _mm256_permute2x128_si256(rnd_4_21, rnd_4_21, 0x01);
  __m256i rnd_5_5 = _mm256_blendv_epi8(rnd_4_5, shf_5_21, blm_5);
  __m256i rnd_5_21 = _mm256_blendv_epi8(shf_5_5, rnd_4_21, blm_5);
  *(STORE_TYPE*)(dst_origin + 5*dst_stride) = rnd_5_5;
  *(STORE_TYPE*)(dst_origin + 21*dst_stride) = rnd_5_21;
  __m256i shf_5_6 = _mm256_permute2x128_si256(rnd_4_6, rnd_4_6, 0x01);
  __m256i shf_5_22 = _mm256_permute2x128_si256(rnd_4_22, rnd_4_22, 0x01);
  __m256i rnd_5_6 = _mm256_blendv_epi8(rnd_4_6, shf_5_22, blm_5);
  _mm_prefetch(prf_origin+13*src_stride, _MM_HINT_NTA);
  __m256i rnd_5_22 = _mm256_blendv_epi8(shf_5_6, rnd_4_22, blm_5);
  *(STORE_TYPE*)(dst_origin + 6*dst_stride) = rnd_5_6;
  *(STORE_TYPE*)(dst_origin + 22*dst_stride) = rnd_5_22;
  __m256i shf_5_7 = _mm256_permute2x128_si256(rnd_4_7, rnd_4_7, 0x01);
  __m256i shf_5_23 = _mm256_permute2x128_si256(rnd_4_23, rnd_4_23, 0x01);
  __m256i rnd_5_7 = _mm256_blendv_epi8(rnd_4_7, shf_5_23, blm_5);
  __m256i rnd_5_23 = _mm256_blendv_epi8(shf_5_7, rnd_4_23, blm_5);
  *(STORE_TYPE*)(dst_origin + 7*dst_stride) = rnd_5_7;
  *(STORE_TYPE*)(dst_origin + 23*dst_stride) = rnd_5_23;
  __m256i shf_5_8 = _mm256_permute2x128_si256(rnd_4_8, rnd_4_8, 0x01);
  __m256i shf_5_24 = _mm256_permute2x128_si256(rnd_4_24, rnd_4_24, 0x01);
  __m256i rnd_5_8 = _mm256_blendv_epi8(rnd_4_8, shf_5_24, blm_5);
  __m256i rnd_5_24 = _mm256_blendv_epi8(shf_5_8, rnd_4_24, blm_5);
  *(STORE_TYPE*)(dst_origin + 8*dst_stride) = rnd_5_8;
  *(STORE_TYPE*)(dst_origin + 24*dst_stride) = rnd_5_24;
  __m256i shf_5_9 = _mm256_permute2x128_si256(rnd_4_9, rnd_4_9, 0x01);
  __m256i shf_5_25 = _mm256_permute2x128_si256(rnd_4_25, rnd_4_25, 0x01);
  __m256i rnd_5_9 = _mm256_blendv_epi8(rnd_4_9, shf_5_25, blm_5);
  __m256i rnd_5_25 = _mm256_blendv_epi8(shf_5_9, rnd_4_25, blm_5);
  *(STORE_TYPE*)(dst_origin + 9*dst_stride) = rnd_5_9;
  *(STORE_TYPE*)(dst_origin + 25*dst_stride) = rnd_5_25;
  __m256i shf_5_10 = _mm256_permute2x128_si256(rnd_4_10, rnd_4_10, 0x01);
  __m256i shf_5_26 = _mm256_permute2x128_si256(rnd_4_26, rnd_4_26, 0x01);
  __m256i rnd_5_10 = _mm256_blendv_epi8(rnd_4_10, shf_5_26, blm_5);
  _mm_prefetch(prf_origin+14*src_stride, _MM_HINT_NTA);
  __m256i rnd_5_26 = _mm256_blendv_epi8(shf_5_10, rnd_4_26, blm_5);
  *(STORE_TYPE*)(dst_origin + 10*dst_stride) = rnd_5_10;
  *(STORE_TYPE*)(dst_origin + 26*dst_stride) = rnd_5_26;
  __m256i shf_5_11 = _mm256_permute2x128_si256(rnd_4_11, rnd_4_11, 0x01);
  __m256i shf_5_27 = _mm256_permute2x128_si256(rnd_4_27, rnd_4_27, 0x01);
  __m256i rnd_5_11 = _mm256_blendv_epi8(rnd_4_11, shf_5_27, blm_5);
  __m256i rnd_5_27 = _mm256_blendv_epi8(shf_5_11, rnd_4_27, blm_5);
  *(STORE_TYPE*)(dst_origin + 11*dst_stride) = rnd_5_11;
  *(STORE_TYPE*)(dst_origin + 27*dst_stride) = rnd_5_27;
  __m256i shf_5_12 = _mm256_permute2x128_si256(rnd_4_12, rnd_4_12, 0x01);
  __m256i shf_5_28 = _mm256_permute2x128_si256(rnd_4_28, rnd_4_28, 0x01);
  __m256i rnd_5_12 = _mm256_blendv_epi8(rnd_4_12, shf_5_28, blm_5);
  __m256i rnd_5_28 = _mm256_blendv_epi8(shf_5_12, rnd_4_28, blm_5);
  *(STORE_TYPE*)(dst_origin + 12*dst_stride) = rnd_5_12;
  *(STORE_TYPE*)(dst_origin + 28*dst_stride) = rnd_5_28;
  __m256i shf_5_13 = _mm256_permute2x128_si256(rnd_4_13, rnd_4_13, 0x01);
  __m256i shf_5_29 = _mm256_permute2x128_si256(rnd_4_29, rnd_4_29, 0x01);
  __m256i rnd_5_13 = _mm256_blendv_epi8(rnd_4_13, shf_5_29, blm_5);
  __m256i rnd_5_29 = _mm256_blendv_epi8(shf_5_13, rnd_4_29, blm_5);
  *(STORE_TYPE*)(dst_origin + 13*dst_stride) = rnd_5_13;
  *(STORE_TYPE*)(dst_origin + 29*dst_stride) = rnd_5_29;
  __m256i shf_5_14 = _mm256_permute2x128_si256(rnd_4_14, rnd_4_14, 0x01);
  __m256i shf_5_30 = _mm256_permute2x128_si256(rnd_4_30, rnd_4_30, 0x01);
  __m256i rnd_5_14 = _mm256_blendv_epi8(rnd_4_14, shf_5_30, blm_5);
  _mm_prefetch(prf_origin+15*src_stride, _MM_HINT_NTA);
  __m256i rnd_5_30 = _mm256_blendv_epi8(shf_5_14, rnd_4_30, blm_5);
  *(STORE_TYPE*)(dst_origin + 14*dst_stride) = rnd_5_14;
  *(STORE_TYPE*)(dst_origin + 30*dst_stride) = rnd_5_30;
  __m256i shf_5_15 = _mm256_permute2x128_si256(rnd_4_15, rnd_4_15, 0x01);
  __m256i shf_5_31 = _mm256_permute2x128_si256(rnd_4_31, rnd_4_31, 0x01);
  __m256i rnd_5_15 = _mm256_blendv_epi8(rnd_4_15, shf_5_31, blm_5);
  __m256i rnd_5_31 = _mm256_blendv_epi8(shf_5_15, rnd_4_31, blm_5);
  *(STORE_TYPE*)(dst_origin + 15*dst_stride) = rnd_5_15;
  *(STORE_TYPE*)(dst_origin + 31*dst_stride) = rnd_5_31;
}

#ifdef USE_AVX512
/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param src_stride[in]:
/// \param dst_stride[in]:
void gf127_matrix_transpose_64x64_u512(uint8_t* dst_origin,
                                       const uint8_t* src_origin,
                                       const size_t src_stride,
                                       const size_t dst_stride) {
    static const uint32_t matrix_transpose_table[] __attribute__((aligned(32))) = {
        0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
    };

    const __m512i m1 = _mm512_setr_epi64(0b0000, 0b0001, 0b1000, 0b1001, 0b0100, 0b0101, 0b1100, 0b1101);
    const __m512i m2 = _mm512_setr_epi64(0b0010, 0b0011, 0b1010, 0b1011, 0b0110, 0b0111, 0b1110, 0b1111);

    const __m512i m3 = _mm512_setr_epi64(0b0000, 0b0001, 0b0010, 0b0011, 0b1000, 0b1001, 0b1010, 0b1011);
    const __m512i m4 = _mm512_setr_epi64(0b0100, 0b0101, 0b0110, 0b0111, 0b1100, 0b1101, 0b1110, 0b1111);

    __m512i t[64];
    for (uint32_t i = 0; i < 64; i++) {
        t[i] = _mm512_loadu_si512((const __m512i *)(src_origin + i*src_stride));
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=2) {
        const __m512i t0 = _mm512_unpacklo_epi8(t[i+0], t[i+1]);
        const __m512i t1 = _mm512_unpackhi_epi8(t[i+0], t[i+1]);
        t[i+0] = t0;
        t[i+1] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=4) {
        const __m512i t0 = _mm512_unpacklo_epi16(t[i+0], t[i+2]);
        const __m512i t1 = _mm512_unpacklo_epi16(t[i+1], t[i+3]);
        const __m512i t2 = _mm512_unpackhi_epi16(t[i+0], t[i+2]);
        const __m512i t3 = _mm512_unpackhi_epi16(t[i+1], t[i+3]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=8) {
        const __m512i t0 = _mm512_unpacklo_epi32(t[i+0], t[i+4]);
        const __m512i t1 = _mm512_unpacklo_epi32(t[i+1], t[i+5]);
        const __m512i t2 = _mm512_unpacklo_epi32(t[i+2], t[i+6]);
        const __m512i t3 = _mm512_unpacklo_epi32(t[i+3], t[i+7]);
        const __m512i t4 = _mm512_unpackhi_epi32(t[i+0], t[i+4]);
        const __m512i t5 = _mm512_unpackhi_epi32(t[i+1], t[i+5]);
        const __m512i t6 = _mm512_unpackhi_epi32(t[i+2], t[i+6]);
        const __m512i t7 = _mm512_unpackhi_epi32(t[i+3], t[i+7]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
        t[i+4] = t4;
        t[i+5] = t5;
        t[i+6] = t6;
        t[i+7] = t7;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        const __m512i t0 = _mm512_unpacklo_epi64(t[i+ 0], t[i+ 8]);
        const __m512i t1 = _mm512_unpackhi_epi64(t[i+ 0], t[i+ 8]);
        const __m512i t2 = _mm512_unpacklo_epi64(t[i+16], t[i+24]);
        const __m512i t3 = _mm512_unpackhi_epi64(t[i+16], t[i+24]);
        const __m512i t4 = _mm512_unpacklo_epi64(t[i+32], t[i+40]);
        const __m512i t5 = _mm512_unpackhi_epi64(t[i+32], t[i+40]);
        const __m512i t6 = _mm512_unpacklo_epi64(t[i+48], t[i+56]);
        const __m512i t7 = _mm512_unpackhi_epi64(t[i+48], t[i+56]);
        t[i+ 0] = t0;
        t[i+ 8] = t1;
        t[i+16] = t2;
        t[i+24] = t3;
        t[i+32] = t4;
        t[i+40] = t5;
        t[i+48] = t6;
        t[i+56] = t7;
    }

    // swap 128 bit limbs
    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const __m512i t0 = _mm512_permutex2var_epi64(t[i+ 0], m1, t[i+16]);
        const __m512i t1 = _mm512_permutex2var_epi64(t[i+ 0], m2, t[i+16]);
        const __m512i t2 = _mm512_permutex2var_epi64(t[i+32], m1, t[i+48]);
        const __m512i t3 = _mm512_permutex2var_epi64(t[i+32], m2, t[i+48]);
        t[i+ 0] = t0;
        t[i+16] = t1;
        t[i+32] = t2;
        t[i+48] = t3;
    }

    // swap 256 bit limbs
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        const __m512i t0 = _mm512_permutex2var_epi64(t[i+0], m3, t[i+32]);
        const __m512i t1 = _mm512_permutex2var_epi64(t[i+0], m4, t[i+32]);
        t[i+ 0] = t0;
        t[i+32] = t1;
    }


    #pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t off = j*16;

        #pragma unroll
        for (uint32_t i = 0; i < 16; i++) {
            const uint32_t pos = matrix_transpose_table[i];
            _mm512_storeu_si512((__m512i *)(dst_origin + (off+i)*dst_stride), t[off+pos]);
        }
    }
}
#endif

/// \param dst
/// \param src
/// \param n must be divisable by 64
void gf127_matrix_transpose_opt_avx2(uint8_t *dst,
                                     const uint8_t *src,
                                     const size_t n) {
    for (uint64_t rb = 0; rb < n/64; rb++) {
        for (uint64_t cb = 0; cb < n/64; cb++) {
            const uint8_t* src_origin = src + (rb*n+cb)*64;
            const uint8_t* prf_origin = gf127_next_block(src, rb, cb, n);
                  uint8_t* dst_origin = dst + (cb*n+rb)*64;

            gf127_matrix_transpose_32x32_u256(dst_origin,         src_origin,         n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32,      src_origin+32*n,    n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32*n,    src_origin+32,      n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32*n+32, src_origin+32*n+32, n, n);
        }
    }
}

/// input: (read row wise)
/// 0           4 (byte)
/// 0  1  2  3 |4
/// 4  5  6  7 |8
/// 8  9  10 11|12
/// 12 13 14 15|16
///
/// output:
/// 0  4  8  12 
/// 1  4  9  13
/// 2  6 10  14 
/// 3  7 11  15 
static inline __m128i gf127_transpose_4x4 (const __m128i m) {
    return _mm_shuffle_epi8 (m, _mm_setr_epi8 (0, 4, 8, 12,
                                               1, 5, 9, 13,
                                               2, 6, 10, 14,
                                               3, 7, 11, 15));
}

#endif

///
static inline
void gf127_matrix_transpose_opt2(uint8_t *dst,
                                const uint8_t *src,
                                const size_t n) {
    /// NOTE: possible values for `bsize = 8,16,32,64`
    /// NOTE: this transpose is not completly randomly choosen.
    /// Floyd: did some benchmarks, see `src/127/README.md`
    const size_t bsize = 32;
    uint64_t rb = 0;
#ifdef USE_AVX2
    const uint32_t avx2_rem = n % 64;
    for (; rb < n/64; rb++) {
        for (uint64_t cb = 0; cb < n/64; cb++) {
            const uint8_t* src_origin = src + (rb*n+cb)*64;
                  uint8_t* dst_origin = dst + (cb*n+rb)*64;

            gf127_matrix_transpose_32x32_u256(dst_origin,         src_origin,         n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32,      src_origin+32*n,    n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32*n,    src_origin+32,      n, n);
            gf127_matrix_transpose_32x32_u256(dst_origin+32*n+32, src_origin+32*n+32, n, n);
        }
    }

    rb *= (64 / bsize);

    if (avx2_rem >= 32) {
        // solve above the last 64x64 block
        for (uint64_t cb = 0; cb < ((n/bsize) - 1); cb++) {
            const uint8_t *srcb_origin = src + (rb * n + cb) * bsize;
                  uint8_t *dstb_origin = dst + (rb * n + cb) * bsize;
            for (size_t rw = 0; rw < bsize / 8; rw++) {
                for (size_t cw = 0; cw < bsize / 8; cw++) {
                    const uint8_t *srcw_origin = srcb_origin + (cw * n + rw) * 8;
                    uint8_t *dstw_origin = dstb_origin + (rw * n + cw) * 8;
                    gf127_matrix_transpose8x8(dstw_origin, srcw_origin, n, n);
                }
            }
        }
    }

#endif

    for (; rb < n / bsize; rb++) {
        for (uint64_t cb = 0; cb < n / bsize; cb++) {
            const uint8_t *srcb_origin = src + (rb * n + cb) * bsize;
                  uint8_t *dstb_origin = dst + (cb * n + rb) * bsize;

            for (size_t rw = 0; rw < bsize / 8; rw++) {
                for (size_t cw = 0; cw < bsize / 8; cw++) {
                    const uint8_t *srcw_origin = srcb_origin + (cw * n + rw) * 8;
                          uint8_t *dstw_origin = dstb_origin + (rw * n + cw) * 8;
                    gf127_matrix_transpose8x8(dstw_origin, srcw_origin, n, n);
                }
            }
        }
    }

    const uint32_t rem = n % bsize;
    if (rem) {
        // solve the last columns
        for (uint32_t i = rb*bsize; i < n; i++) {
            for(uint32_t j = 0; j < n; j++) {
                dst[j*n + i] = src[i*n + j];
            }
        }

        // this if is needed to catch the case `n < 32`, and
        // we do not want to do things twicw.
        if (rb != 0) {
            // solve the last rows
            for (uint32_t i = 0; i < n; i++) {
                for(uint32_t j = rb*bsize; j < n; j++) {
                    dst[j*n + i] = src[i*n + j];
                }
            }
        }
    }
}

/// assumes max 8 rows
/// assumes that the output matrix has n columns
/// \param dst
/// \param src
/// \param n   number of columns
static inline void gf127_matrix_transpose8xN(uint8_t *dst, 
                                             const uint8_t *src,
                                             const uint32_t n) {
    const uint32_t bsize = 8;
    uint64_t rb = 0;
    for (; rb < n / bsize; rb++) {
            const uint8_t *srcb_origin = src + ( 0 * n + rb) * bsize;
                  uint8_t *dstb_origin = dst + (rb * n +  0) * bsize;
        gf127_matrix_transpose8x8(dstb_origin, srcb_origin, n, n);
    }

    rb *= bsize;
    for (; rb < n; rb++) {
        for (uint32_t j = 0; j < 8; j++) {
            const uint8_t t = src[j*n + rb];
            dst[rb*n + j] = t;
        }
    }
}

/// assumes max 8 cols in the input matrix
/// assumes that the output matrix has n columns
/// \param dst
/// \param src
/// \param n   number of columns
static inline void gf127_matrix_transposeNx8(uint8_t *dst, 
                                             const uint8_t *src,
                                             const uint32_t n) {
    const uint32_t bsize = 8;
    uint64_t cb = 0;
    for (; cb < n / bsize; cb++) {
            const uint8_t *srcb_origin = src + (cb * n +  0) * bsize;
                  uint8_t *dstb_origin = dst + ( 0 * n + cb) * bsize;
        gf127_matrix_transpose8x8(dstb_origin, srcb_origin, n, n);
    }

    cb *= bsize;
    for (; cb < n; cb++) {
        for (uint32_t j = 0; j < 8; j++) {
            const uint8_t t = src[cb*n + j];
            dst[j*n + cb] = t;
        }
    }
}

/// 
void gf127_matrix_vector_mul(uint8_t *c, 
                             const uint8_t *a, 
                             const uint8_t *b, 
                             const size_t n,
                             const size_t m) {
    for (uint32_t i = 0; i < m; i++) {
        const uint8_t v = b[i];
        for (uint32_t j = 0; j < n; j++) {
            const uint8_t w = a[i*m + j];
            const uint8_t a = gf127_mul(v, w);
            c[i*m + j] = a;
        }
    }
}

void gf127_matrix_vector_mul_32x32x1_u256(uint8_t *c, 
                                          const uint8_t *a, 
                                          const uint8_t *b) {
    __m256i t0;
    __m256i t1;
    t1 = _mm256_set1_epi8(b[0]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[1]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[2]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[3]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[4]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[5]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[6]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[7]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[8]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[9]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[10]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[11]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[12]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[13]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[14]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[15]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[16]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[17]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[18]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[19]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[20]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[21]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[22]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[23]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[24]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[25]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[26]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[27]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[28]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[29]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[30]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
    t1 = _mm256_set1_epi8(b[31]);
    t0 = _mm256_loadu_si256((__m256i *)(a + 0*32));
    t0 = gf127v_mul_u256(t0, t1);
    _mm256_storeu_si256((__m256i *)(c + 0*32), t0);
    a += 32;
    c += 32;
}

