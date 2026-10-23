#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "../16/vector.h"

/// \param n[in]: number of elements
/// \return zero initialized vector
static inline
gf16to3* gf16to3_vector_alloc(const uint32_t n) {
    return (gf16to3 *)calloc(n, sizeof(gf16to3));
}

/// \return v[i]
static inline
gf16to3 gf16to3_vector_get(const gf16to3 *v,
                           const uint32_t i) {
    return v[i];
}

/// v[i] = a
static inline
void gf16to3_vector_set(gf16to3 *v,
                        const uint32_t i,
                        const gf16to3 a) {
    v[i] = a;
}

/// \param v
/// \param n
static inline
void gf16to3_vector_print(const gf16to3 *v,
                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

/// \param v
/// \param n
static inline
void gf16to3_vector_zero(gf16to3 *v,
                         const uint32_t n) {
    memset(v, 0, n*sizeof(gf16to3));
}

/// \param v
/// \param n
static inline void gf16to3_vector_random(gf16to3 *v,
                                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFFF;
    }
}

/// \param v1
/// \param v2
/// \param n
static inline void gf16to3_vector_copy(gf16to3 *__restrict__ v1,
                                       const gf16to3 *__restrict__ v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add(gf16to3 *out,
                                      const gf16to3 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add_gf16(gf16to3 *__restrict__ out,
                                           const gf16 *__restrict__ in,
                                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const ff_t t = gf16_vector_get(in, i);
        out[i] ^= t;
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_add(gf16to3 *out,
                                             const gf16to3 c,
                                             const gf16to3 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(c, in[i]);
    }
}

/// \param out = in1 + c*in2
/// \param in1
/// \param c
/// \param in2
/// \param n
static inline void gf16to3_vector_scalar_add_v2(gf16to3 *out,
                                                const gf16to3 *in1,
                                                const gf16to3 c,
                                                const gf16to3 *in2,
                                                const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf16to3_mul(c, in2[i]);
    }
}

/// \return sum_i a_i*b_i
static inline gf16to3 gf16to3_vector_mul_acc(const gf16to3 *a,
                                             const gf16to3 *b,
                                             const uint32_t n) {
    gf16to3 acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc ^= gf16to3_mul(a[i], b[i]);
    }
    return acc;
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_add_gf16(gf16to3 *__restrict__ out,
                                                  const gf16 c,
                                                  const gf16to3 *__restrict__ in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(in[i], c);
    }
}

/// \brief vector1 = vector2 * scalar
///
/// \param[out] vector1 Vector over ff_mu
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf16to3_vector_scalar_gf16(gf16to3 *vector1,
                                              const gf16to3 scalar,
                                              const gf16 *vector2,
                                              const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const gf16to3 t = gf16_vector_get(vector2, i);
        vector1[i] = gf16to3_mul(scalar, t);
    }
}

/// \brief vector1 += vector2 * scalar
///
/// \param[out] vector1 Vector over ff_mu
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf16to3_vector_add_scalar_gf16(gf16to3 *vector1,
                                                  const gf16to3 scalar,
                                                  const gf16 *vector2,
                                                  const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const gf16to3 t = gf16_vector_get(vector2, i);
        vector1[i] ^= gf16to3_mul(scalar, t);
    }
}

/// internal helper: copies the elements [i, n) of the packed gf16 vector
/// `in` into a zero padded 8 byte buffer `tmp` (16 elements)
static inline void gf16to3_vector_load_gf16_tail(uint8_t tmp[8],
                                                 const gf16 *in,
                                                 const uint32_t rem) {
    memset(tmp, 0, 8);
    for (uint32_t j = 0; j < (rem+1)/2; ++j) { tmp[j] = in[j]; }
}

#ifdef USE_AVX2
/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add_gf16_u256(gf16to3 *__restrict__ out,
                                                const gf16 *__restrict__ in,
                                                const uint32_t n) {
    uint32_t i = n;
    // avx2 code
    while (i >= 16u) {
        const __m256i m2 = gf16to3_vector_extend_gf16_x16(in);
        const __m256i m1 = _mm256_loadu_si256((const __m256i *)out);

        _mm256_storeu_si256((__m256i *)out, m1 ^ m2);
        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16] __attribute__((aligned(32))) = {0};
        gf16to3_vector_load_gf16_tail(tmp, in, i);

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);

        for (uint32_t j = 0; j < i; ++j) { tmp2[j] = out[j]; }
        const __m256i m1 = _mm256_load_si256((const __m256i *)tmp2);

        _mm256_store_si256((__m256i *)tmp2, t1^m1);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }
}

/// out = in1 + in2
/// \param out
/// \param in1
/// \param in2
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256_v2(gf16to3 *out,
                                              const gf16to3 *in1,
                                              const gf16to3 *in2,
                                              const uint32_t d) {
    uint32_t i = d;
    // avx2 code
    while (i >= 16u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((const __m256i *)in1) ^
                            _mm256_loadu_si256((const __m256i *)in2));
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((const __m128i *)in1) ^
                         _mm_loadu_si128((const __m128i *)in2));
        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ *in2++;
    }
}

/// out += in1
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    gf16to3_vector_add_u256_v2(out, out, in1, d);
}

/// out = in1 + a*in2
static inline void gf16to3_vector_scalar_add_u256_v2(gf16to3 *out,
                                                     const gf16to3 *in1,
                                                     const gf16to3 a,
                                                     const gf16to3 *in2,
                                                     const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)in2);
        const __m256i t3 = gf16to3v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ gf16to3_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_u256(gf16to3 *out,
                                                  const gf16to3 a,
                                                  const gf16to3 *in1,
                                                  const uint32_t d) {
    gf16to3_vector_scalar_add_u256_v2(out, out, a, in1, d);
}

/// \return sum_i a_i*b_i
static inline gf16to3 gf16to3_vector_mul_acc_u256(const gf16to3 *a,
                                                  const gf16to3 *b,
                                                  const uint32_t d) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i+16) <= d; i += 16) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf16to3v_mul_u256(t1, t2);
    }

    gf16to3 r = gf16to3_hadd_u256(acc);
    for (; i < d; i++) {
        r ^= gf16to3_mul(a[i], b[i]);
    }
    return r;
}

/// out += a*in1
/// \param a[in]: scalar in gf16
static inline void gf16to3_vector_scalar_add_gf16_u256(gf16to3 *out,
                                                        const gf16 a,
                                                        const gf16to3 *in1,
                                                        const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi8(a & 0xF);
    const __m128i a128 = _mm_set1_epi8(a & 0xF);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t3 = gf16v_mul_lo_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)out);
        const __m128i t2 = _mm_loadu_si128((const __m128i *)in1);
        const __m128i t3 = gf16v_mul_lo_u128(t2, a128);
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

/// \brief vector1 += vector2 * scalar
/// \param[out] vector1 Vector over gf16^3
/// \param[in] scalar Scalar over gf16^3
/// \param[in] vector2 Vector over gf16
/// \param[in] ncols number of columns
static inline void gf16to3_vector_add_scalar_gf16_u256(gf16to3 *vector1,
                                                       const gf16to3 scalar,
                                                       const gf16 *vector2,
                                                       const uint32_t ncols) {
    uint32_t i = ncols;
    const __m256i s = _mm256_set1_epi16(scalar);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = gf16to3_vector_extend_gf16_x16(vector2);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)vector1);

        _mm256_storeu_si256((__m256i *)vector1, t2^t3);
        i       -= 16u;
        vector2 += 8u;
        vector1 += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16] __attribute__((aligned(32)));
        gf16to3_vector_load_gf16_tail(tmp, vector2, i);

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_store_si256((__m256i *)tmp2, t2);

        for (uint32_t j = 0; j < i; ++j) { vector1[j] ^= tmp2[j]; }
    }
}

/// \param out = scalar*in
/// \param scalar[in]: over gf16to3
/// \param in[in]: over
/// \param n
static inline void gf16to3_vector_scalar_gf16_u256(gf16to3 *__restrict__ out,
                                                   gf16to3 scalar,
                                                   const gf16 *__restrict__ in,
                                                   const uint32_t n) {
    uint32_t i = n;
    const __m256i s = _mm256_set1_epi16(scalar);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = gf16to3_vector_extend_gf16_x16(in);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_storeu_si256((__m256i *)out, t2);
        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16] __attribute__((aligned(32)));
        gf16to3_vector_load_gf16_tail(tmp, in, i);

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_store_si256((__m256i *)tmp2, t2);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }
}
#endif

#ifdef USE_NEON

/// \param out += in
/// \param in
/// \param n
static inline
void gf16to3_vector_add_gf16_u128(gf16to3 *__restrict__ out,
                                  const gf16 *__restrict__ in,
                                  const uint32_t n) {
    uint32_t i = n;
    while (i >= 16u) {
        const uint16x8x2_t m2 = gf16to3_vector_extend_gf16_x16(in);
        const uint16x8x2_t m1 = vld1q_u16_x2(out);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(out, m3);

        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16] = {0};
        gf16to3_vector_load_gf16_tail(tmp, in, i);

        const uint16x8x2_t m2 = gf16to3_vector_extend_gf16_x16(tmp);

        for (uint32_t j = 0; j < i; ++j) { tmp2[j] = out[j]; }
        const uint16x8x2_t m1 = vld1q_u16_x2(tmp2);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(tmp2, m3);
        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }
}

/// out = in1 + in2
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u128_v2(gf16to3 *out,
                                              const gf16to3 *in1,
                                              const gf16to3 *in2,
                                              const uint32_t d) {
    uint32_t i = d;
    while(i >= 8u) {
        const uint16x8_t m1 = vld1q_u16(in1);
        const uint16x8_t m2 = vld1q_u16(in2);
        vst1q_u16(out, gf16to3v_add_u128(m1, m2));
        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ *in2++;
    }
}

/// out += in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u128(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    gf16to3_vector_add_u128_v2(out, out, in1, d);
}

/// out = in1 + a*in2
static inline void gf16to3_vector_scalar_add_u128_v2(gf16to3 *out,
                                                     const gf16to3 *in1,
                                                     const gf16to3 a,
                                                     const gf16to3 *in2,
                                                     const uint32_t d) {
    uint32_t i = d;
    const uint16x8_t a128 = vdupq_n_u16(a);

    while (i >= 8u) {
        const uint16x8_t t1 = vld1q_u16(in1);
        const uint16x8_t t2 = vld1q_u16(in2);
        const uint16x8_t t3 = gf16to3v_mul_u128(t2, a128);
        vst1q_u16(out, veorq_u16(t1, t3));

        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ gf16to3_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_u128(gf16to3 *out,
                                                  const gf16to3 a,
                                                  const gf16to3 *in1,
                                                  const uint32_t d) {
    gf16to3_vector_scalar_add_u128_v2(out, out, a, in1, d);
}

/// \return sum_i a_i*b_i
static inline gf16to3 gf16to3_vector_mul_acc_u128(const gf16to3 *a,
                                                  const gf16to3 *b,
                                                  const uint32_t d) {
    uint16x8_t acc = vdupq_n_u16(0);
    uint32_t i = 0;
    for (; (i+8) <= d; i += 8) {
        const uint16x8_t t1 = vld1q_u16(a + i);
        const uint16x8_t t2 = vld1q_u16(b + i);
        acc = veorq_u16(acc, gf16to3v_mul_u128(t1, t2));
    }

    gf16to3 r = gf16to3_hadd_u128(acc);
    for (; i < d; i++) {
        r ^= gf16to3_mul(a[i], b[i]);
    }
    return r;
}

/// out += a*in1
/// \param a[in]: scalar in gf16
static inline void gf16to3_vector_scalar_add_gf16_u128(gf16to3 *out,
                                                       const gf16 a,
                                                       const gf16to3 *in1,
                                                       const uint32_t d) {
    uint32_t i = d;
    while (i >= 8u) {
        const uint16x8_t t1 = vld1q_u16(out);
        const uint16x8_t t2 = vld1q_u16(in1);
        const uint16x8_t t3 = gf16to3v_mul_gf16_u128(t2, a);
        vst1q_u16(out, veorq_u16(t1, t3));

        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf16to3_mul_gf16(*in1++, a);
    }
}

/// \brief vector1 += vector2 * scalar
/// \param[out] vector1 Vector over gf16^3
/// \param[in] scalar Scalar over gf16^3
/// \param[in] vector2 Vector over gf16
/// \param[in] ncols number of columns
static inline void gf16to3_vector_add_scalar_gf16_u128(gf16to3 *vector1,
                                                       const gf16to3 scalar,
                                                       const gf16 *vector2,
                                                       const uint32_t ncols) {
    uint32_t i = ncols;
    const uint16x8x2_t s = vdupq_n_u16_x2(scalar);

    while (i >= 16u) {
        const uint16x8x2_t t1 = gf16to3_vector_extend_gf16_x16(vector2);
        const uint16x8x2_t t2 = gf16to3v_mul_u256(t1, s);
        const uint16x8x2_t t3 = vld1q_u16_x2(vector1);

        vst1q_u16_x2(vector1, veorq_u16_x2(t2, t3));
        i       -= 16u;
        vector2 += 8u;
        vector1 += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16];
        gf16to3_vector_load_gf16_tail(tmp, vector2, i);

        const uint16x8x2_t t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const uint16x8x2_t t2 = gf16to3v_mul_u256(t1, s);

        vst1q_u16_x2(tmp2, t2);
        for (uint32_t j = 0; j < i; ++j) { vector1[j] ^= tmp2[j]; }
    }
}

/// \param out = scalar*in
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_gf16_u128(gf16to3 *__restrict__ out,
                                                   gf16to3 scalar,
                                                   const gf16 *__restrict__ in,
                                                   const uint32_t n) {
    uint32_t i = n;
    const uint16x8x2_t s = vdupq_n_u16_x2(scalar);

    while (i >= 16u) {
        const uint16x8x2_t t1 = gf16to3_vector_extend_gf16_x16(in);
        const uint16x8x2_t t2 = gf16to3v_mul_u256(t1, s);

        vst1q_u16_x2(out, t2);
        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[8];
        uint16_t tmp2[16];
        gf16to3_vector_load_gf16_tail(tmp, in, i);

        const uint16x8x2_t t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const uint16x8x2_t t2 = gf16to3v_mul_u256(t1, s);

        vst1q_u16_x2(tmp2, t2);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }
}
#endif

#ifdef USE_AVX512

/// out = in1 + in2
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u512_v2(gf16to3 *out,
                                              const gf16to3 *in1,
                                              const gf16to3 *in2,
                                              const uint32_t d) {
    uint32_t i = 0;
    for (; (i+32) <= d; i += 32) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf16to3v_add_u512(a, b));
    }

    if (i < d) {
        const __mmask32 m = (__mmask32)((1ull << (d - i)) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi16(m, (const void *)(in1 + i));
        const __m512i b = _mm512_maskz_loadu_epi16(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi16((void *)(out + i), m, gf16to3v_add_u512(a, b));
    }
}

/// out += in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u512(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    gf16to3_vector_add_u512_v2(out, out, in1, d);
}

/// out = in1 + a*in2
static inline void gf16to3_vector_scalar_add_u512_v2(gf16to3 *out,
                                                     const gf16to3 *in1,
                                                     const gf16to3 a,
                                                     const gf16to3 *in2,
                                                     const uint32_t d) {
    const __m512i aa = _mm512_set1_epi16(a);
    uint32_t i = 0;
    for (; (i+32) <= d; i += 32) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), t1 ^ gf16to3v_mul_u512(t2, aa));
    }

    if (i < d) {
        const __mmask32 m = (__mmask32)((1ull << (d - i)) - 1ull);
        const __m512i t1 = _mm512_maskz_loadu_epi16(m, (const void *)(in1 + i));
        const __m512i t2 = _mm512_maskz_loadu_epi16(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi16((void *)(out + i), m, t1 ^ gf16to3v_mul_u512(t2, aa));
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_u512(gf16to3 *out,
                                                  const gf16to3 a,
                                                  const gf16to3 *in1,
                                                  const uint32_t d) {
    gf16to3_vector_scalar_add_u512_v2(out, out, a, in1, d);
}

/// \return sum_i a_i*b_i
static inline gf16to3 gf16to3_vector_mul_acc_u512(const gf16to3 *a,
                                                  const gf16to3 *b,
                                                  const uint32_t d) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i+32) <= d; i += 32) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc ^= gf16to3v_mul_u512(t1, t2);
    }

    if (i < d) {
        const __mmask32 m = (__mmask32)((1ull << (d - i)) - 1ull);
        const __m512i t1 = _mm512_maskz_loadu_epi16(m, (const void *)(a + i));
        const __m512i t2 = _mm512_maskz_loadu_epi16(m, (const void *)(b + i));
        acc ^= gf16to3v_mul_u512(t1, t2);
    }

    return gf16to3_hadd_u512(acc);
}
#endif
