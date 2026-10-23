#pragma once

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "arith.h"
#include "../2/vector.h"


/// \return v[i]
static inline
gf2to64 gf2to64_vector_get(const gf2to64 *v,
                           const uint32_t i) {
    gf2to64 b = v[i];
    return b;
}

/// \param v[out]: v[i] = a
/// \param i[in]: index position
/// \param a[in]: value to set the array index to
static inline
void gf2to64_vector_set(gf2to64 *v,
                        const uint32_t i,
                        const gf2to64 a) {
    v[i] = a;
}

/// \param v[in]: vector to print
/// \param n[in]: length of the vector
static inline
void gf2to64_vector_print(const gf2to64 *v,
                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2to64 t = gf2to64_vector_get(v, i);
        printf("%llx ", (unsigned long long)t);
    }
    printf("\n");
}

/// \param n size of the vector (number of elements)
/// \return zero initialized vector with n elements
static inline
gf2to64* gf2to64_vector_alloc(const uint32_t n) {
    return (gf2to64 *)calloc(n, sizeof(gf2to64));
}

/// \param v[out]: vector to set to zero
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_zero(gf2to64 *v,
                         const uint32_t n) {
    memset(v, 0, n * sizeof(gf2to64));
}

/// \param v[out]: random vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_random(gf2to64 *v,
                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = ((gf2to64)rand()) ^ ((gf2to64)rand() << 31u) ^ ((gf2to64)rand() << 62u);
    }
}

/// \param v1[out]: out vector
/// \param v2[in]: in vector
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_copy(gf2to64 *__restrict__ v1,
                                       const gf2to64 *__restrict__ v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_add(gf2to64 *out,
                                      const gf2to64 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}


/// \param out[in/out] += in
/// \param in[in]
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_add_gf2(gf2to64 *__restrict__ out,
                                          const gf2 *__restrict__ in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 t = gf2_vector_get(in, i);
        out[i] ^= t;
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add(gf2to64 *out,
                               const gf2to64 c,
                               const gf2to64 *in,
                               const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to64_mul(c, in[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2(gf2to64 *__restrict__ out,
                                   const gf2 c,
                                   const gf2to64 *__restrict__ in,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to64_mul_gf2(in[i], c);
    }
}

/// \param out += t*in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_v2(gf2to64 *__restrict__ out,
                                      const gf2to64 t,
                                      const gf2 *__restrict__ in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        const gf2to64 b = gf2to64_mul_gf2(t, a);
        out[i] ^= b;
    }
}

/// out = in1 + a*in2
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_v2(gf2to64 *__restrict__ out,
                                  const gf2to64 *__restrict__ in1,
                                  const gf2to64 a,
                                  const gf2to64 *__restrict__ in2,
                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf2to64_mul(a, in2[i]);
    }
}

/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
/// \return sum(a[i] * b[i])
static inline
gf2to64 gf2to64_vector_mul_acc(const gf2to64 *a,
                               const gf2to64 *b,
                               const uint32_t n) {
    gf2to64 acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc ^= gf2to64_mul(a[i], b[i]);
    }

    return acc;
}


#ifdef USE_AVX2

/// \param out[out]: vector of length n: = += in
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf2to64_vector_add_u256(gf2to64 *out,
                             const gf2to64 *in,
                             const uint32_t n) {
    uint32_t i = 0;
    for (; (i+4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2to64v_add_u256(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(out[i], in[i]);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf2to64_vector_add_u256_v2(gf2to64 *__restrict__ out,
                                const gf2to64 *__restrict__ in1,
                                const gf2to64 *__restrict__ in2,
                                const uint32_t n) {
    uint32_t i = 0;
    for (; (i+4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2to64v_add_u256(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(in1[i], in2[i]);
    }
}

/// \param out += in
/// \param in: packed GF(2) vector
/// \param n[in] number of elements NOT bytes
static inline
void gf2to64_vector_add_gf2_u256(gf2to64 *__restrict__ out,
                                 const gf2 *__restrict__ in,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = in[i/8];
        const __m256i t0 = gf2to64v_expand_gf2_x4_u256(b);
        const __m256i t1 = gf2to64v_expand_gf2_x4_u256(b >> 4);
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(out + i + 0));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(out + i + 4));
        _mm256_storeu_si256((__m256i *)(out + i + 0), t0 ^ a0);
        _mm256_storeu_si256((__m256i *)(out + i + 4), t1 ^ a1);
    }

    for (; i < n; i++) {
        out[i] ^= gf2_vector_get(in, i);
    }
}

/// \param out = in1 + in2
/// \param in1[in]:
/// \param in2[in]: packed GF(2) vector
/// \param n[in]: number of elements NOT bytes
static inline
void gf2to64_vector_add_gf2_u256_v2(gf2to64 *__restrict__ out,
                                    const gf2to64 *__restrict__ in1,
                                    const gf2 *__restrict__ in2,
                                    const uint32_t n) {
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = in2[i/8];
        const __m256i t0 = gf2to64v_expand_gf2_x4_u256(b);
        const __m256i t1 = gf2to64v_expand_gf2_x4_u256(b >> 4);
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(in1 + i + 0));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(in1 + i + 4));
        _mm256_storeu_si256((__m256i *)(out + i + 0), t0 ^ a0);
        _mm256_storeu_si256((__m256i *)(out + i + 4), t1 ^ a1);
    }

    for (; i < n; i++) {
        out[i] = in1[i] ^ gf2_vector_get(in2, i);
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param d[in]: length of the vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_u256_v2(gf2to64 *__restrict__ out,
                                       const gf2to64 *__restrict__ in1,
                                       const gf2to64 a,
                                       const gf2to64 *__restrict__ in2,
                                       const uint32_t d) {
    const __m256i aa = _mm256_set1_epi64x((long long)a);
    uint32_t i = 0;
    for (; (i+4) <= d; i += 4) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));
        const __m256i u = gf2to64v_mul_u256(aa, t2);
        _mm256_storeu_si256((__m256i *)(out + i), u ^ t1);
    }

    for (; i < d; i++) {
        out[i] = in1[i] ^ gf2to64_mul(a, in2[i]);
    }
}

/// poly1 += scalar * poly2
/// \param poly1[out]
/// \param scalar[in]
/// \param poly2[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_u256(gf2to64 *poly1,
                                    const gf2to64 scalar,
                                    const gf2to64 *poly2,
                                    const uint32_t n) {
    const __m256i aa = _mm256_set1_epi64x((long long)scalar);
    uint32_t i = 0;
    for (; (i+4) <= n; i += 4) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(poly1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(poly2 + i));
        const __m256i u = gf2to64v_mul_u256(aa, t2);
        _mm256_storeu_si256((__m256i *)(poly1 + i), u ^ t1);
    }

    for (; i < n; i++) {
        poly1[i] ^= gf2to64_mul(scalar, poly2[i]);
    }
}

/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vectors (number of elements)
/// \return sum(a[i] * b[i])
static inline
gf2to64 gf2to64_vector_mul_acc_u256(const gf2to64 *a,
                                    const gf2to64 *b,
                                    const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i+4) <= n; i += 4) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf2to64v_mul_u256(t1, t2);
    }

    uint64_t tmp[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, acc);
    gf2to64 ret = tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];
    for (; i < n; i++) {
        ret ^= gf2to64_mul(a[i], b[i]);
    }

    return ret;
}

/// out += scalar * in
/// \param out[out]: over gf2to64
/// \param scalar[in]: over gf2to64
/// \param in[in]: over gf2, packed
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_u256_v2(gf2to64 *out,
                                           const gf2to64 scalar,
                                           const gf2 *in,
                                           const uint32_t n) {
    const __m256i s = _mm256_set1_epi64x((long long)scalar);
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = in[i/8];
        const __m256i t0 = gf2to64v_mul_gf2_u256(s, gf2to64v_expand_gf2_x4_ext_u256(b));
        const __m256i t1 = gf2to64v_mul_gf2_u256(s, gf2to64v_expand_gf2_x4_ext_u256(b >> 4));
        const __m256i a0 = _mm256_loadu_si256((const __m256i *)(out + i + 0));
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(out + i + 4));
        _mm256_storeu_si256((__m256i *)(out + i + 0), t0 ^ a0);
        _mm256_storeu_si256((__m256i *)(out + i + 4), t1 ^ a1);
    }

    for (; i < n; i++) {
        out[i] ^= gf2to64_mul_gf2(scalar, gf2_vector_get(in, i));
    }
}

/// out += scalar * in
/// \param out[out]: over gf2to64
/// \param scalar[in]: over gf2
/// \param in[in]: over gf2to64
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_u256(gf2to64 *__restrict__ out,
                                        const gf2 scalar,
                                        const gf2to64 *__restrict__ in,
                                        const uint32_t n) {
    const __m256i s = _mm256_set1_epi64x(-(long long)scalar);
    uint32_t i = 0;
    for (; (i+4) <= n; i += 4) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i t2 = gf2to64v_mul_gf2_u256(t1, s);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), t2 ^ t3);
    }

    for (; i < n; i++) {
        out[i] ^= gf2to64_mul_gf2(in[i], scalar);
    }
}

#endif

#ifdef USE_AVX512

/// \param out[in/out]: out += in
/// \param in[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_add_u512(gf2to64 *out,
                             const gf2to64 *in,
                             const uint32_t n) {
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in + i));
        const __m512i b = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), gf2to64v_add_u512(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(out[i], in[i]);
    }
}

/// \param out[out]: out = in1 + in2
/// \param in1[in]: vector of length n
/// \param in2[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_add_u512_v2(gf2to64 *__restrict__ out,
                                const gf2to64 *__restrict__ in1,
                                const gf2to64 *__restrict__ in2,
                                const uint32_t n) {
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf2to64v_add_u512(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(in1[i], in2[i]);
    }
}

/// \param out[in/out]: out += c*in
/// \param c[in]: scalar
/// \param in[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_scalar_add_u512(gf2to64 *out,
                                    const gf2to64 c,
                                    const gf2to64 *in,
                                    const uint32_t n) {
    const __m512i cc = _mm512_set1_epi64((long long)c);
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in + i));
        const __m512i b = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), gf2to64v_add_u512(b, gf2to64v_mul_u512(cc, a)));
    }

    for (; i < n; i++) {
        out[i] ^= gf2to64_mul(c, in[i]);
    }
}

/// \param out[out]: out = in1 + c*in2
/// \param in1[in]: vector of length n
/// \param c[in]: scalar
/// \param in2[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_scalar_add_u512_v2(gf2to64 *__restrict__ out,
                                       const gf2to64 *__restrict__ in1,
                                       const gf2to64 c,
                                       const gf2to64 *__restrict__ in2,
                                       const uint32_t n) {
    const __m512i cc = _mm512_set1_epi64((long long)c);
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf2to64v_add_u512(a, gf2to64v_mul_u512(cc, b)));
    }

    for (; i < n; i++) {
        out[i] = in1[i] ^ gf2to64_mul(c, in2[i]);
    }
}

/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vectors (number of elements)
/// \return sum(a[i] * b[i])
static inline
gf2to64 gf2to64_vector_mul_acc_u512(const gf2to64 *a,
                                    const gf2to64 *b,
                                    const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc = gf2to64v_add_u512(acc, gf2to64v_mul_u512(t1, t2));
    }

    uint64_t tmp[8] __attribute__((aligned(64)));
    _mm512_store_si512((void *)tmp, acc);
    gf2to64 ret = 0;
    for (uint32_t j = 0; j < 8; j++) { ret ^= tmp[j]; }
    for (; i < n; i++) {
        ret ^= gf2to64_mul(a[i], b[i]);
    }

    return ret;
}

#endif

#ifdef USE_NEON

/// \param out[in/out]: out += in
/// \param in[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_add_u128(gf2to64 *out,
                             const gf2to64 *in,
                             const uint32_t n) {
    uint32_t i = 0;
    for (; (i+2) <= n; i += 2) {
        const uint64x2_t a = vld1q_u64(in + i);
        const uint64x2_t b = vld1q_u64(out + i);
        vst1q_u64(out + i, gf2to64v_add_u128(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(out[i], in[i]);
    }
}

/// \param out[out]: out = in1 + in2
/// \param in1[in]: vector of length n
/// \param in2[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_add_u128_v2(gf2to64 *__restrict__ out,
                                const gf2to64 *__restrict__ in1,
                                const gf2to64 *__restrict__ in2,
                                const uint32_t n) {
    uint32_t i = 0;
    for (; (i+2) <= n; i += 2) {
        const uint64x2_t a = vld1q_u64(in1 + i);
        const uint64x2_t b = vld1q_u64(in2 + i);
        vst1q_u64(out + i, gf2to64v_add_u128(a, b));
    }

    for (; i < n; i++) {
        out[i] = gf2to64_add(in1[i], in2[i]);
    }
}

/// \param out[in/out]: out += c*in
/// \param c[in]: scalar
/// \param in[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_scalar_add_u128(gf2to64 *out,
                                    const gf2to64 c,
                                    const gf2to64 *in,
                                    const uint32_t n) {
    const uint64x2_t cc = vdupq_n_u64(c);
    uint32_t i = 0;
    for (; (i+2) <= n; i += 2) {
        const uint64x2_t a = vld1q_u64(in + i);
        const uint64x2_t b = vld1q_u64(out + i);
        vst1q_u64(out + i, gf2to64v_add_u128(b, gf2to64v_mul_u128(cc, a)));
    }

    for (; i < n; i++) {
        out[i] ^= gf2to64_mul(c, in[i]);
    }
}

/// \param out[out]: out = in1 + c*in2
/// \param in1[in]: vector of length n
/// \param c[in]: scalar
/// \param in2[in]: vector of length n
/// \param n[in]: number of elements
static inline
void gf2to64_vector_scalar_add_u128_v2(gf2to64 *__restrict__ out,
                                       const gf2to64 *__restrict__ in1,
                                       const gf2to64 c,
                                       const gf2to64 *__restrict__ in2,
                                       const uint32_t n) {
    const uint64x2_t cc = vdupq_n_u64(c);
    uint32_t i = 0;
    for (; (i+2) <= n; i += 2) {
        const uint64x2_t a = vld1q_u64(in1 + i);
        const uint64x2_t b = vld1q_u64(in2 + i);
        vst1q_u64(out + i, gf2to64v_add_u128(a, gf2to64v_mul_u128(cc, b)));
    }

    for (; i < n; i++) {
        out[i] = in1[i] ^ gf2to64_mul(c, in2[i]);
    }
}

/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vectors (number of elements)
/// \return sum(a[i] * b[i])
static inline
gf2to64 gf2to64_vector_mul_acc_u128(const gf2to64 *a,
                                    const gf2to64 *b,
                                    const uint32_t n) {
    uint64x2_t acc = vdupq_n_u64(0);
    uint32_t i = 0;
    for (; (i+2) <= n; i += 2) {
        const uint64x2_t t1 = vld1q_u64(a + i);
        const uint64x2_t t2 = vld1q_u64(b + i);
        acc = gf2to64v_add_u128(acc, gf2to64v_mul_u128(t1, t2));
    }

    gf2to64 ret = vgetq_lane_u64(acc, 0) ^ vgetq_lane_u64(acc, 1);
    for (; i < n; i++) {
        ret ^= gf2to64_mul(a[i], b[i]);
    }

    return ret;
}

#endif
