#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline
gf256to8* gf256to8_vector_alloc(const uint32_t n) {
    return (gf256to8 *)calloc(n, sizeof(gf256to8));
}

/// \return v[i]
static inline
gf256to8 gf256to8_vector_get(const gf256to8 *v,
                             const uint32_t i) {
    return v[i];
}

/// v[i] = a
static inline
void gf256to8_vector_set(gf256to8 *v,
                         const uint32_t i,
                         const gf256to8 a) {
    v[i] = a;
}

/// \param v[in]: vector to print
/// \param n[in]: number of elements
static inline
void gf256to8_vector_print(const gf256to8 *v,
                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%016llx ", (unsigned long long)v[i]);
    }
    printf("\n");
}

/// v = 0
static inline
void gf256to8_vector_zero(gf256to8 *v,
                          const uint32_t n) {
    memset(v, 0, (size_t)n * sizeof(gf256to8));
}

/// v = random
static inline
void gf256to8_vector_random(gf256to8 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = ((uint64_t)rand() << 42u) ^ ((uint64_t)rand() << 21u) ^ (uint64_t)rand();
    }
}

/// out = in
static inline
void gf256to8_vector_copy(gf256to8 *__restrict__ out,
                          const gf256to8 *__restrict__ in,
                          const uint32_t n) {
    memcpy(out, in, (size_t)n * sizeof(gf256to8));
}

/// out += in
static inline
void gf256to8_vector_add(gf256to8 *out,
                         const gf256to8 *in,
                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = gf256to8_add(out[i], in[i]);
    }
}

/// out += c*in
static inline
void gf256to8_vector_scalar_add(gf256to8 *out,
                                const gf256to8 c,
                                const gf256to8 *in,
                                const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf256to8_mul(c, in[i]);
    }
}

/// out = in1 + c*in2
static inline
void gf256to8_vector_scalar_add_v2(gf256to8 *out,
                                   const gf256to8 *in1,
                                   const gf256to8 c,
                                   const gf256to8 *in2,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf256to8_mul(c, in2[i]);
    }
}

/// \return sum_i a[i]*b[i]
static inline
gf256to8 gf256to8_vector_mul_acc(const gf256to8 *a,
                                 const gf256to8 *b,
                                 const uint32_t n) {
    gf256to8 acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc ^= gf256to8_mul(a[i], b[i]);
    }
    return acc;
}

#ifdef USE_AVX2
/// out += in
static inline
void gf256to8_vector_add_u256(gf256to8 *out,
                              const gf256to8 *in,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf256to8v_add_u256(a, b));
    }
    for (; i < n; i++) { out[i] ^= in[i]; }
}

/// out = in1 + in2
static inline
void gf256to8_vector_add_u256_v2(gf256to8 *out,
                                 const gf256to8 *in1,
                                 const gf256to8 *in2,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf256to8v_add_u256(a, b));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ in2[i]; }
}

/// out += c*in
static inline
void gf256to8_vector_scalar_add_u256(gf256to8 *out,
                                     const gf256to8 c,
                                     const gf256to8 *in,
                                     const uint32_t n) {
    const __m256i s = _mm256_set1_epi64x((long long)c);
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i o = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), o ^ gf256to8v_mul_u256(a, s));
    }
    for (; i < n; i++) { out[i] ^= gf256to8_mul(c, in[i]); }
}

/// out = in1 + c*in2
static inline
void gf256to8_vector_scalar_add_u256_v2(gf256to8 *out,
                                        const gf256to8 *in1,
                                        const gf256to8 c,
                                        const gf256to8 *in2,
                                        const uint32_t n) {
    const __m256i s = _mm256_set1_epi64x((long long)c);
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), a ^ gf256to8v_mul_u256(b, s));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ gf256to8_mul(c, in2[i]); }
}

/// \return sum_i a[i]*b[i]
static inline
gf256to8 gf256to8_vector_mul_acc_u256(const gf256to8 *a,
                                      const gf256to8 *b,
                                      const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf256to8v_mul_u256(t1, t2);
    }

    gf256to8 tmp[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, acc);
    gf256to8 r = tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];
    for (; i < n; i++) { r ^= gf256to8_mul(a[i], b[i]); }
    return r;
}
#endif // USE_AVX2

#ifdef USE_NEON
/// out += in
static inline
void gf256to8_vector_add_u128(gf256to8 *out,
                              const gf256to8 *in,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        vst1q_u64(out + i, gf256to8v_add_u128(vld1q_u64(out + i), vld1q_u64(in + i)));
    }
    for (; i < n; i++) { out[i] ^= in[i]; }
}

/// out = in1 + in2
static inline
void gf256to8_vector_add_u128_v2(gf256to8 *out,
                                 const gf256to8 *in1,
                                 const gf256to8 *in2,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        vst1q_u64(out + i, gf256to8v_add_u128(vld1q_u64(in1 + i), vld1q_u64(in2 + i)));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ in2[i]; }
}

/// out += c*in
static inline
void gf256to8_vector_scalar_add_u128(gf256to8 *out,
                                     const gf256to8 c,
                                     const gf256to8 *in,
                                     const uint32_t n) {
    const uint64x2_t s = vdupq_n_u64(c);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const uint64x2_t t = gf256to8v_mul_u128(vld1q_u64(in + i), s);
        vst1q_u64(out + i, veorq_u64(vld1q_u64(out + i), t));
    }
    for (; i < n; i++) { out[i] ^= gf256to8_mul(c, in[i]); }
}

/// out = in1 + c*in2
static inline
void gf256to8_vector_scalar_add_u128_v2(gf256to8 *out,
                                        const gf256to8 *in1,
                                        const gf256to8 c,
                                        const gf256to8 *in2,
                                        const uint32_t n) {
    const uint64x2_t s = vdupq_n_u64(c);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const uint64x2_t t = gf256to8v_mul_u128(vld1q_u64(in2 + i), s);
        vst1q_u64(out + i, veorq_u64(vld1q_u64(in1 + i), t));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ gf256to8_mul(c, in2[i]); }
}

/// \return sum_i a[i]*b[i]
static inline
gf256to8 gf256to8_vector_mul_acc_u128(const gf256to8 *a,
                                      const gf256to8 *b,
                                      const uint32_t n) {
    uint64x2_t acc = vdupq_n_u64(0);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        acc = veorq_u64(acc, gf256to8v_mul_u128(vld1q_u64(a + i), vld1q_u64(b + i)));
    }

    gf256to8 r = vgetq_lane_u64(acc, 0) ^ vgetq_lane_u64(acc, 1);
    for (; i < n; i++) { r ^= gf256to8_mul(a[i], b[i]); }
    return r;
}
#endif // USE_NEON

#ifdef USE_AVX512
/// out += in
static inline
void gf256to8_vector_add_u512(gf256to8 *out,
                              const gf256to8 *in,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in + i));
        const __m512i b = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), gf256to8v_add_u512(a, b));
    }
    for (; i < n; i++) { out[i] ^= in[i]; }
}

/// out = in1 + in2
static inline
void gf256to8_vector_add_u512_v2(gf256to8 *out,
                                 const gf256to8 *in1,
                                 const gf256to8 *in2,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf256to8v_add_u512(a, b));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ in2[i]; }
}

/// out += c*in
static inline
void gf256to8_vector_scalar_add_u512(gf256to8 *out,
                                     const gf256to8 c,
                                     const gf256to8 *in,
                                     const uint32_t n) {
    const __m512i s = _mm512_set1_epi64((long long)c);
    uint32_t i = 0;
    for (; (i + 8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in + i));
        const __m512i o = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), o ^ gf256to8v_mul_u512(a, s));
    }
    for (; i < n; i++) { out[i] ^= gf256to8_mul(c, in[i]); }
}

/// out = in1 + c*in2
static inline
void gf256to8_vector_scalar_add_u512_v2(gf256to8 *out,
                                        const gf256to8 *in1,
                                        const gf256to8 c,
                                        const gf256to8 *in2,
                                        const uint32_t n) {
    const __m512i s = _mm512_set1_epi64((long long)c);
    uint32_t i = 0;
    for (; (i + 8) <= n; i += 8) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), a ^ gf256to8v_mul_u512(b, s));
    }
    for (; i < n; i++) { out[i] = in1[i] ^ gf256to8_mul(c, in2[i]); }
}

/// \return sum_i a[i]*b[i]
static inline
gf256to8 gf256to8_vector_mul_acc_u512(const gf256to8 *a,
                                      const gf256to8 *b,
                                      const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i + 8) <= n; i += 8) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc ^= gf256to8v_mul_u512(t1, t2);
    }

    gf256to8 tmp[8] __attribute__((aligned(64)));
    _mm512_store_si512((void *)tmp, acc);
    gf256to8 r = 0;
    for (uint32_t j = 0; j < 8; j++) { r ^= tmp[j]; }
    for (; i < n; i++) { r ^= gf256to8_mul(a[i], b[i]); }
    return r;
}
#endif // USE_AVX512
