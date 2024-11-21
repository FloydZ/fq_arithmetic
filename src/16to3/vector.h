#pragma once
#include "arith.h"

/// 
/// @param n 
/// @return 
static inline gf16to3* gf16to3_vector_alloc(const uint32_t n) {
    return (gf16to3 *)malloc(n*sizeof(gf16to3));
}

/// 
/// @param v 
/// @param n 
static inline void gf16to3_vector_zero(gf16to3 *v,
                                       const uint32_t n) {
    memset(v, 0, n*sizeof(gf16to3));
}

/// 
/// @param v 
/// @param n 
static inline void gf16to3_vector_rand(gf16to3 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFFF;
    }
}

/// 
/// @param v1 
/// @param v2 
/// @param n 
static inline void gf16to3_vector_copy(gf16to3 *v1,
                                       gf16to3 *v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// 
/// @param out += in
/// @param in 
/// @param n 
static inline void gf16to3_vector_add(gf16to3 *out,
                                      const gf16to3 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf16to3_vector_scalar_add(gf16to3 *out,
                                             const gf16to3 c,
                                             const gf16to3 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(c, in[i]);
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf16to3_vector_scalar_add_gf16(gf16to3 *out,
                                                  const gf16 c,
                                                  const gf16to3 *in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(in[i], c);
    }
}

/// out += in1
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    uint32_t i = d;
    // avx2 code
    while (i >= 16u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)out) ^
                            _mm256_loadu_si256((__m256i *)in1));
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)out) ^
                         _mm_loadu_si128((__m128i *)in1));
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++;
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_u256(gf16to3 *out,
                                                  const gf16to3 a,
                                                  const gf16to3 *in1,
                                                  const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);
    // const __m128i a128 = _mm_set1_epi16(a);
    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf16to3v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    // while(i >= 8u) {
    //     const __m128i t1 = _mm_loadu_si128((__m128i *)out);
    //     const __m128i t2 = _mm_loadu_si128((__m128i *)in1);
    //     const __m128i t3 = gf16to3v_mul_u128(t2, a256);
    //     const __m128i t = t1 ^ t3;

    //     _mm_storeu_si128((__m128i *)out, t);
    //     i   -= 8u;
    //     in1 += 8u;
    //     out += 8u;
    // }

    for (; i > 0; --i) {
        *out++ ^= gf16to3_mul(a, *in1++);
    }
}

static inline void gf16to3_vector_scalar_add_gf16_u256(gf16to3 *out,
                                                        const gf16 a,
                                                        const gf16to3 *in1,
                                                        const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);
    const __m128i a128 = _mm_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf16v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        const __m128i t1 = _mm_loadu_si128((__m128i *)out);
        const __m128i t2 = _mm_loadu_si128((__m128i *)in1);
        const __m128i t3 = gf16v_mul_u128(t2, a128);
        const __m128i t = t1 ^ t3;

        _mm_storeu_si128((__m128i *)out, t);
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf16to3_mul_gf16(*in1++, a);
    }
}
