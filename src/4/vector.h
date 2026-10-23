#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"

/// NOTE: vectors are packed: element i is stored in the bits
/// [2*(i%4), 2*(i%4)+1] of byte i/4. All functions taking a length `n` count
/// elements, not bytes. Bits beyond element n-1 in the last byte are never
/// modified by the arithmetic functions below (except `zero` and `random`).
#define gf4_vector_size(n) (((n)+3)/4)

/// \param n[in]: number of elements
/// \return mask of the valid bits in the last byte, if n%4 != 0
static inline uint8_t gf4_vector_tail_mask(const uint32_t n) {
    return (uint8_t)((1u << (2u*(n % 4u))) - 1u);
}

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline gf4* gf4_vector_alloc(const uint32_t n) {
    return (gf4 *)calloc(gf4_vector_size(n) + 1, sizeof(gf4));
}

/// \param v[in]: vector
/// \param i[in]: index
/// \return v[i]
static inline gf4 gf4_vector_get(const gf4 *v,
                                 const uint32_t i) {
    return (v[i/4] >> (2u*(i%4))) & 3u;
}

/// v[i] = a
/// \param v[in/out]: vector
/// \param i[in]: index
/// \param a[in]: value
static inline void gf4_vector_set(gf4 *v,
                                  const uint32_t i,
                                  const gf4 a) {
    const uint32_t s = 2u*(i%4);
    const uint8_t mask = (uint8_t)~(3u << s);
    v[i/4] = (v[i/4] & mask) | ((a & 3u) << s);
}

/// \param v[in]: vector to print
/// \param n[in]: number of elements
static inline void gf4_vector_print(const gf4 *v,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%u ", (unsigned)gf4_vector_get(v, i));
    }
    printf("\n");
}

/// \param v[out]: v = 0
/// \param n[in]: number of elements
static inline void gf4_vector_zero(gf4 *v,
                                   const uint32_t n) {
    memset(v, 0, gf4_vector_size(n));
}

/// \param v[out]: random vector, bits beyond element n-1 are zeroed
/// \param n[in]: number of elements
static inline void gf4_vector_random(gf4 *v,
                                     const uint32_t n) {
    const uint32_t nb = gf4_vector_size(n);
    for (uint32_t i = 0; i < nb; i++) {
        v[i] = rand();
    }

    if (n % 4) {
        v[nb - 1] &= gf4_vector_tail_mask(n);
    }
}

/// \param out[out]: out = in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline void gf4_vector_copy(gf4 *out,
                                   const gf4 *in,
                                   const uint32_t n) {
    const uint32_t nb = n / 4;
    memcpy(out, in, nb);
    if (n % 4) {
        const uint8_t m = gf4_vector_tail_mask(n);
        out[nb] = (out[nb] & (uint8_t)~m) | (in[nb] & m);
    }
}

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add(gf4 *out,
                                  const gf4 *in,
                                  const uint32_t n) {
    const uint32_t nb = n / 4;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] ^= in[i];
    }

    if (n % 4) {
        out[nb] ^= in[nb] & gf4_vector_tail_mask(n);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_v2(gf4 *out,
                                     const gf4 *in1,
                                     const gf4 *in2,
                                     const uint32_t n) {
    const uint32_t nb = n / 4;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] = in1[i] ^ in2[i];
    }

    if (n % 4) {
        const uint8_t m = gf4_vector_tail_mask(n);
        out[nb] = (out[nb] & (uint8_t)~m) | ((in1[nb] ^ in2[nb]) & m);
    }
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add(gf4 *out,
                                         const gf4 c,
                                         const gf4 *in,
                                         const uint32_t n) {
    const uint8_t bc = gf4_broadcast_u8(c);
    const uint32_t nb = n / 4;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] ^= gf4v_mul_u8(in[i], bc);
    }

    if (n % 4) {
        out[nb] ^= gf4v_mul_u8(in[nb], bc) & gf4_vector_tail_mask(n);
    }
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_v2(gf4 *out,
                                            const gf4 *in1,
                                            const gf4 c,
                                            const gf4 *in2,
                                            const uint32_t n) {
    const uint8_t bc = gf4_broadcast_u8(c);
    const uint32_t nb = n / 4;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] = in1[i] ^ gf4v_mul_u8(in2[i], bc);
    }

    if (n % 4) {
        const uint8_t m = gf4_vector_tail_mask(n);
        const uint8_t t = in1[nb] ^ gf4v_mul_u8(in2[nb], bc);
        out[nb] = (out[nb] & (uint8_t)~m) | (t & m);
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf4 gf4_vector_mul_acc(const gf4 *a,
                                     const gf4 *b,
                                     const uint32_t n) {
    const uint32_t nb = n / 4;
    uint8_t acc = 0;
    for (uint32_t i = 0; i < nb; i++) {
        acc ^= gf4v_mul_u8(a[i], b[i]);
    }

    if (n % 4) {
        acc ^= gf4v_mul_u8(a[nb], b[nb]) & gf4_vector_tail_mask(n);
    }

    return gf4_hadd_u64(acc);
}

#ifdef USE_AVX2

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u256(gf4 *out,
                                       const gf4 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in  + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf4v_add_u256(a, b));
    }

    gf4_vector_add(out + i, in + i, n - 4*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u256_v2(gf4 *out,
                                          const gf4 *in1,
                                          const gf4 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf4v_add_u256(a, b));
    }

    gf4_vector_add_v2(out + i, in1 + i, in2 + i, n - 4*i);
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u256(gf4 *out,
                                              const gf4 c,
                                              const gf4 *in,
                                              const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in  + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf4v_add_u256(a, gf4v_mul_u256(b, s)));
    }

    gf4_vector_scalar_add(out + i, c, in + i, n - 4*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u256_v2(gf4 *out,
                                                 const gf4 *in1,
                                                 const gf4 c,
                                                 const gf4 *in2,
                                                 const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf4v_add_u256(a, gf4v_mul_u256(b, s)));
    }

    gf4_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 4*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf4 gf4_vector_mul_acc_u256(const gf4 *a,
                                          const gf4 *b,
                                          const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = gf4v_add_u256(acc, gf4v_mul_u256(t1, t2));
    }

    return gf4_hadd_u256(acc) ^ gf4_vector_mul_acc(a + i, b + i, n - 4*i);
}

#endif

#ifdef USE_NEON

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u128(gf4 *out,
                                       const gf4 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in  + i);
        vst1q_u8(out + i, gf4v_add_u128(a, b));
    }

    gf4_vector_add(out + i, in + i, n - 4*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u128_v2(gf4 *out,
                                          const gf4 *in1,
                                          const gf4 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf4v_add_u128(a, b));
    }

    gf4_vector_add_v2(out + i, in1 + i, in2 + i, n - 4*i);
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u128(gf4 *out,
                                              const gf4 c,
                                              const gf4 *in,
                                              const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in  + i);
        vst1q_u8(out + i, gf4v_add_u128(a, gf4v_mul_u128(b, s)));
    }

    gf4_vector_scalar_add(out + i, c, in + i, n - 4*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u128_v2(gf4 *out,
                                                 const gf4 *in1,
                                                 const gf4 c,
                                                 const gf4 *in2,
                                                 const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf4v_add_u128(a, gf4v_mul_u128(b, s)));
    }

    gf4_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 4*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf4 gf4_vector_mul_acc_u128(const gf4 *a,
                                          const gf4 *b,
                                          const uint32_t n) {
    uint8x16_t acc = vdupq_n_u8(0);
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t t1 = vld1q_u8(a + i);
        const uint8x16_t t2 = vld1q_u8(b + i);
        acc = gf4v_add_u128(acc, gf4v_mul_u128(t1, t2));
    }

    return gf4_hadd_u128(acc) ^ gf4_vector_mul_acc(a + i, b + i, n - 4*i);
}

#endif

#ifdef USE_AVX512

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u512(gf4 *out,
                                       const gf4 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(out + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in  + i));
        _mm512_storeu_si512((void *)(out + i), gf4v_add_u512(a, b));
    }

    gf4_vector_add(out + i, in + i, n - 4*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_add_u512_v2(gf4 *out,
                                          const gf4 *in1,
                                          const gf4 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf4v_add_u512(a, b));
    }

    gf4_vector_add_v2(out + i, in1 + i, in2 + i, n - 4*i);
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u512(gf4 *out,
                                              const gf4 c,
                                              const gf4 *in,
                                              const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(out + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in  + i));
        _mm512_storeu_si512((void *)(out + i), gf4v_add_u512(a, gf4v_mul_u512(b, s)));
    }

    gf4_vector_scalar_add(out + i, c, in + i, n - 4*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf4_vector_scalar_add_u512_v2(gf4 *out,
                                                 const gf4 *in1,
                                                 const gf4 c,
                                                 const gf4 *in2,
                                                 const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(gf4_broadcast_u8(c));
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf4v_add_u512(a, gf4v_mul_u512(b, s)));
    }

    gf4_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 4*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf4 gf4_vector_mul_acc_u512(const gf4 *a,
                                          const gf4 *b,
                                          const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    const uint32_t nb = n / 4;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc = gf4v_add_u512(acc, gf4v_mul_u512(t1, t2));
    }

    return gf4_hadd_u512(acc) ^ gf4_vector_mul_acc(a + i, b + i, n - 4*i);
}

#endif
