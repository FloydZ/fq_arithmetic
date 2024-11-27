#pragma once 

#include <stdint.h>
#include <stdio.h>

#include "arith.h"

/// NOTE: everything row major

static inline void matrix_print(const uint8_t *in, 
                                const uint32_t nrows,
                                const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for(uint32_t j = 0; j < ncols; j++) {
            printf("%02d ", in[i*ncols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/// simple transpose, needed for debugging
static inline void matrix_transpose(uint8_t *out, 
                                    uint8_t *in, 
                                    const uint32_t nrows,
                                    const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for(uint32_t j = 0; j < ncols; j++) {
            out[j*ncols + i] = in[i*ncols + j];
        }
    }
}

/// TODO: move cryptanalyslib
void matrix_transpose4x4(uint8_t* dst, 
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
void matrix_transpose8x8(uint8_t* dst,
                         const uint8_t* src, 
                         const size_t src_stride, 
                         const size_t dst_stride) {
    // load rows of src matrix
    uint64_t a0 = *((uint64_t*)(src+0*src_stride));
    uint64_t a1 = *((uint64_t*)(src+1*src_stride));
    uint64_t a2 = *((uint64_t*)(src+2*src_stride));
    uint64_t a3 = *((uint64_t*)(src+3*src_stride));
    uint64_t a4 = *((uint64_t*)(src+4*src_stride));
    uint64_t a5 = *((uint64_t*)(src+5*src_stride));
    uint64_t a6 = *((uint64_t*)(src+6*src_stride));
    uint64_t a7 = *((uint64_t*)(src+7*src_stride));
    
    // 2x2 block matrices
    uint64_t b0 = (a0 & 0x00ff00ff00ff00ffULL) | ((a1 << 8) & 0xff00ff00ff00ff00ULL);
    uint64_t b1 = (a1 & 0xff00ff00ff00ff00ULL) | ((a0 >> 8) & 0x00ff00ff00ff00ffULL);
    uint64_t b2 = (a2 & 0x00ff00ff00ff00ffULL) | ((a3 << 8) & 0xff00ff00ff00ff00ULL);
    uint64_t b3 = (a3 & 0xff00ff00ff00ff00ULL) | ((a2 >> 8) & 0x00ff00ff00ff00ffULL);
    uint64_t b4 = (a4 & 0x00ff00ff00ff00ffULL) | ((a5 << 8) & 0xff00ff00ff00ff00ULL);
    uint64_t b5 = (a5 & 0xff00ff00ff00ff00ULL) | ((a4 >> 8) & 0x00ff00ff00ff00ffULL);
    uint64_t b6 = (a6 & 0x00ff00ff00ff00ffULL) | ((a7 << 8) & 0xff00ff00ff00ff00ULL);
    uint64_t b7 = (a7 & 0xff00ff00ff00ff00ULL) | ((a6 >> 8) & 0x00ff00ff00ff00ffULL);
    
    // 4x4 block matrices
    uint64_t c0 = (b0 & 0x0000ffff0000ffffULL) | ((b2 << 16) & 0xffff0000ffff0000ULL);
    uint64_t c1 = (b1 & 0x0000ffff0000ffffULL) | ((b3 << 16) & 0xffff0000ffff0000ULL);
    uint64_t c2 = (b2 & 0xffff0000ffff0000ULL) | ((b0 >> 16) & 0x0000ffff0000ffffULL);
    uint64_t c3 = (b3 & 0xffff0000ffff0000ULL) | ((b1 >> 16) & 0x0000ffff0000ffffULL);
    uint64_t c4 = (b4 & 0x0000ffff0000ffffULL) | ((b6 << 16) & 0xffff0000ffff0000ULL);
    uint64_t c5 = (b5 & 0x0000ffff0000ffffULL) | ((b7 << 16) & 0xffff0000ffff0000ULL);
    uint64_t c6 = (b6 & 0xffff0000ffff0000ULL) | ((b4 >> 16) & 0x0000ffff0000ffffULL);
    uint64_t c7 = (b7 & 0xffff0000ffff0000ULL) | ((b5 >> 16) & 0x0000ffff0000ffffULL);
    
    // 8x8 block matrix
    uint64_t d0 = (c0 & 0x00000000ffffffffULL) | ((c4 << 32) & 0xffffffff00000000ULL);
    uint64_t d1 = (c1 & 0x00000000ffffffffULL) | ((c5 << 32) & 0xffffffff00000000ULL);
    uint64_t d2 = (c2 & 0x00000000ffffffffULL) | ((c6 << 32) & 0xffffffff00000000ULL);
    uint64_t d3 = (c3 & 0x00000000ffffffffULL) | ((c7 << 32) & 0xffffffff00000000ULL);
    uint64_t d4 = (c4 & 0xffffffff00000000ULL) | ((c0 >> 32) & 0x00000000ffffffffULL);
    uint64_t d5 = (c5 & 0xffffffff00000000ULL) | ((c1 >> 32) & 0x00000000ffffffffULL);
    uint64_t d6 = (c6 & 0xffffffff00000000ULL) | ((c2 >> 32) & 0x00000000ffffffffULL);
    uint64_t d7 = (c7 & 0xffffffff00000000ULL) | ((c3 >> 32) & 0x00000000ffffffffULL);
    
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
inline const uint8_t* next_block(const uint8_t *src,
                                 uint64_t rb,
                                 uint64_t cb,
                                 const size_t n) {
    uint64_t cb1 = cb + 1;
    uint64_t rb1 = rb;
    if (cb1 == n/64) {
        rb1 += 1;
        cb1 = 0;
    }

    return src + (rb1*n + cb1) * 64;
}

void matrix_transpose_opt(uint8_t *dst,
                          const uint8_t *src,
                          const size_t n) {
    const size_t bsize = 64;
    
    // iterate over 64x64 block matrices
    for (uint64_t rb = 0; rb < n / bsize; rb++) {
        for (uint64_t cb = 0; cb < n / bsize; cb++) {
            const uint8_t *srcb_origin = src + (rb * n + cb) * bsize;
                  uint8_t *dstb_origin = dst + (cb * n + rb) * bsize;
            const uint8_t *prfb_origin = next_block(src, rb, cb, n);
            
            // iterate over sizeof(W)xsizeof(W) block matrices inside 64x64 block
            for (size_t rw = 0; rw < 64 / 8; rw++) {
                // preload sizeof(W) rows of the next 64x64 block
                for (size_t i = rw * 8; i < (rw + 1) * 8; i++) {
                    _mm_prefetch(prfb_origin + i * n, _MM_HINT_NTA);
                }

                for (size_t cw = 0; cw < 64 / 8; cw++) {
                    const uint8_t *srcw_origin = srcb_origin + (cw * n + rw) * 8;
                          uint8_t *dstw_origin = dstb_origin + (rw * n + cw) * 8;
                    // use vector kernel to transpose sizeof(W)xsizeof(W) matrix
                    matrix_transpose8x8(dstw_origin, srcw_origin, n, n);
                }
            }
        }
    }
}

#ifdef USE_AVX2 
#include <immintrin.h>
/// input: (read row wise)
/// 0           4 (byte)
/// 0  1  2  3 |4
/// 4  5  6  7 |8
/// 8  9  10 11|12
/// 12 13 14 15|16
///
/// output:
/// 0  4  8  12 
inline __m128i transpose_4x4 (__m128i m) {
    return _mm_shuffle_epi8 (m, _mm_setr_epi8 (0, 4, 8, 12,
                                               1, 5, 9, 13,
                                               2, 6, 10, 14,
                                               3, 7, 11, 15));
}

#endif
