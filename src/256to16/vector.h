#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

// a vector of n elements is an array `gf256to16 v[n]` (16*n bytes)

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline
gf256to16* gf256to16_vector_alloc(const uint32_t n) {
    return (gf256to16 *)calloc(n, sizeof(gf256to16));
}

/// out = v[i]
static inline
void gf256to16_vector_get(gf256to16 out,
                          const gf256to16 *v,
                          const uint32_t i) {
    memcpy(out, v[i], sizeof(gf256to16));
}

/// v[i] = a
static inline
void gf256to16_vector_set(gf256to16 *v,
                          const uint32_t i,
                          const gf256to16 a) {
    memcpy(v[i], a, sizeof(gf256to16));
}

/// prints each element as its 16 bytes (byte 0 first)
static inline
void gf256to16_vector_print(const gf256to16 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < 16; j++) {
            printf("%02x", v[i][j]);
        }
        printf(" ");
    }
    printf("\n");
}

/// v = 0
static inline
void gf256to16_vector_zero(gf256to16 *v,
                           const uint32_t n) {
    memset(v, 0, (size_t)n * sizeof(gf256to16));
}

/// v = random
static inline
void gf256to16_vector_random(gf256to16 *v,
                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < 16; j++) {
            v[i][j] = (uint8_t)rand();
        }
    }
}

/// out = in
static inline
void gf256to16_vector_copy(gf256to16 *__restrict__ out,
                           const gf256to16 *__restrict__ in,
                           const uint32_t n) {
    memcpy(out, in, (size_t)n * sizeof(gf256to16));
}

/// out += in
static inline
void gf256to16_vector_add(gf256to16 *out,
                          const gf256to16 *in,
                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf256to16_add(out[i], out[i], in[i]);
    }
}

/// out += c*in
static inline
void gf256to16_vector_scalar_add(gf256to16 *out,
                                 const gf256to16 c,
                                 const gf256to16 *in,
                                 const uint32_t n) {
    gf256to16 t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to16_mul(t, c, in[i]);
        gf256to16_add(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256to16_vector_scalar_add_v2(gf256to16 *out,
                                    const gf256to16 *in1,
                                    const gf256to16 c,
                                    const gf256to16 *in2,
                                    const uint32_t n) {
    gf256to16 t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to16_mul(t, c, in2[i]);
        gf256to16_add(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to16_vector_mul_acc(gf256to16 out,
                              const gf256to16 *a,
                              const gf256to16 *b,
                              const uint32_t n) {
    gf256to16 acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to16_mul(t, a[i], b[i]);
        gf256to16_add(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256to16));
}

#ifdef USE_AVX2
/// out += in
static inline
void gf256to16_vector_add_u256(gf256to16 *out,
                               const gf256to16 *in,
                               const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)in[i]);
        const __m256i b = _mm256_loadu_si256((const __m256i *)out[i]);
        _mm256_storeu_si256((__m256i *)out[i], gf256to16v_add_u256(a, b));
    }
    if (i < n) {
        const __m128i a = _mm_loadu_si128((const __m128i *)in[i]);
        const __m128i b = _mm_loadu_si128((const __m128i *)out[i]);
        _mm_storeu_si128((__m128i *)out[i], a ^ b);
    }
}

/// out = in1 + in2
static inline
void gf256to16_vector_add_u256_v2(gf256to16 *out,
                                  const gf256to16 *in1,
                                  const gf256to16 *in2,
                                  const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)in1[i]);
        const __m256i b = _mm256_loadu_si256((const __m256i *)in2[i]);
        _mm256_storeu_si256((__m256i *)out[i], gf256to16v_add_u256(a, b));
    }
    if (i < n) {
        const __m128i a = _mm_loadu_si128((const __m128i *)in1[i]);
        const __m128i b = _mm_loadu_si128((const __m128i *)in2[i]);
        _mm_storeu_si128((__m128i *)out[i], a ^ b);
    }
}

/// out += c*in
static inline
void gf256to16_vector_scalar_add_u256(gf256to16 *out,
                                      const gf256to16 c,
                                      const gf256to16 *in,
                                      const uint32_t n) {
    const __m128i s128 = _mm_loadu_si128((const __m128i *)c);
    const __m256i s256 = _mm256_broadcastsi128_si256(s128);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)in[i]);
        const __m256i o = _mm256_loadu_si256((const __m256i *)out[i]);
        _mm256_storeu_si256((__m256i *)out[i], o ^ gf256to16v_mul_u256(a, s256));
    }
    if (i < n) {
        const __m128i a = _mm_loadu_si128((const __m128i *)in[i]);
        const __m128i o = _mm_loadu_si128((const __m128i *)out[i]);
        _mm_storeu_si128((__m128i *)out[i], o ^ gf256to16v_mul_u128(a, s128));
    }
}

/// out = in1 + c*in2
static inline
void gf256to16_vector_scalar_add_u256_v2(gf256to16 *out,
                                         const gf256to16 *in1,
                                         const gf256to16 c,
                                         const gf256to16 *in2,
                                         const uint32_t n) {
    const __m128i s128 = _mm_loadu_si128((const __m128i *)c);
    const __m256i s256 = _mm256_broadcastsi128_si256(s128);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)in1[i]);
        const __m256i b = _mm256_loadu_si256((const __m256i *)in2[i]);
        _mm256_storeu_si256((__m256i *)out[i], a ^ gf256to16v_mul_u256(b, s256));
    }
    if (i < n) {
        const __m128i a = _mm_loadu_si128((const __m128i *)in1[i]);
        const __m128i b = _mm_loadu_si128((const __m128i *)in2[i]);
        _mm_storeu_si128((__m128i *)out[i], a ^ gf256to16v_mul_u128(b, s128));
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to16_vector_mul_acc_u256(gf256to16 out,
                                   const gf256to16 *a,
                                   const gf256to16 *b,
                                   const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)a[i]);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)b[i]);
        acc ^= gf256to16v_mul_u256(t1, t2);
    }

    __m128i r = _mm256_castsi256_si128(acc) ^ _mm256_extracti128_si256(acc, 1);
    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)a[i]);
        const __m128i t2 = _mm_loadu_si128((const __m128i *)b[i]);
        r ^= gf256to16v_mul_u128(t1, t2);
    }
    _mm_storeu_si128((__m128i *)out, r);
}
#endif // USE_AVX2

#ifdef USE_NEON
/// loads a single element
static inline uint64x2_t gf256to16v_load_u128(const uint8_t *a) {
    return vreinterpretq_u64_u8(vld1q_u8(a));
}

/// stores a single element
static inline void gf256to16v_store_u128(uint8_t *a, const uint64x2_t b) {
    vst1q_u8(a, vreinterpretq_u8_u64(b));
}

/// out += in
static inline
void gf256to16_vector_add_u128(gf256to16 *out,
                               const gf256to16 *in,
                               const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf256to16v_store_u128(out[i], gf256to16v_add_u128(gf256to16v_load_u128(out[i]),
                                                          gf256to16v_load_u128(in[i])));
    }
}

/// out = in1 + in2
static inline
void gf256to16_vector_add_u128_v2(gf256to16 *out,
                                  const gf256to16 *in1,
                                  const gf256to16 *in2,
                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf256to16v_store_u128(out[i], gf256to16v_add_u128(gf256to16v_load_u128(in1[i]),
                                                          gf256to16v_load_u128(in2[i])));
    }
}

/// out += c*in
static inline
void gf256to16_vector_scalar_add_u128(gf256to16 *out,
                                      const gf256to16 c,
                                      const gf256to16 *in,
                                      const uint32_t n) {
    const uint64x2_t s = gf256to16v_load_u128(c);
    for (uint32_t i = 0; i < n; i++) {
        const uint64x2_t t = gf256to16v_mul_u128(gf256to16v_load_u128(in[i]), s);
        gf256to16v_store_u128(out[i], veorq_u64(gf256to16v_load_u128(out[i]), t));
    }
}

/// out = in1 + c*in2
static inline
void gf256to16_vector_scalar_add_u128_v2(gf256to16 *out,
                                         const gf256to16 *in1,
                                         const gf256to16 c,
                                         const gf256to16 *in2,
                                         const uint32_t n) {
    const uint64x2_t s = gf256to16v_load_u128(c);
    for (uint32_t i = 0; i < n; i++) {
        const uint64x2_t t = gf256to16v_mul_u128(gf256to16v_load_u128(in2[i]), s);
        gf256to16v_store_u128(out[i], veorq_u64(gf256to16v_load_u128(in1[i]), t));
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to16_vector_mul_acc_u128(gf256to16 out,
                                   const gf256to16 *a,
                                   const gf256to16 *b,
                                   const uint32_t n) {
    uint64x2_t acc = vdupq_n_u64(0);
    for (uint32_t i = 0; i < n; i++) {
        acc = veorq_u64(acc, gf256to16v_mul_u128(gf256to16v_load_u128(a[i]),
                                                 gf256to16v_load_u128(b[i])));
    }
    gf256to16v_store_u128(out, acc);
}
#endif // USE_NEON

#ifdef USE_AVX512
/// \return mask selecting the 64-bit limbs of the first `rem` < 4 elements
static inline __mmask8 gf256to16v_tail_mask_u512(const uint32_t rem) {
    return (__mmask8)((1u << (2u*rem)) - 1u);
}

/// out += in
static inline
void gf256to16_vector_add_u512(gf256to16 *out,
                               const gf256to16 *in,
                               const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i a = _mm512_loadu_si512((const void *)in[i]);
        const __m512i b = _mm512_loadu_si512((const void *)out[i]);
        _mm512_storeu_si512((void *)out[i], gf256to16v_add_u512(a, b));
    }
    if (i < n) {
        const __mmask8 k = gf256to16v_tail_mask_u512(n - i);
        const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)in[i]);
        const __m512i b = _mm512_maskz_loadu_epi64(k, (const void *)out[i]);
        _mm512_mask_storeu_epi64((void *)out[i], k, gf256to16v_add_u512(a, b));
    }
}

/// out = in1 + in2
static inline
void gf256to16_vector_add_u512_v2(gf256to16 *out,
                                  const gf256to16 *in1,
                                  const gf256to16 *in2,
                                  const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i a = _mm512_loadu_si512((const void *)in1[i]);
        const __m512i b = _mm512_loadu_si512((const void *)in2[i]);
        _mm512_storeu_si512((void *)out[i], gf256to16v_add_u512(a, b));
    }
    if (i < n) {
        const __mmask8 k = gf256to16v_tail_mask_u512(n - i);
        const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)in1[i]);
        const __m512i b = _mm512_maskz_loadu_epi64(k, (const void *)in2[i]);
        _mm512_mask_storeu_epi64((void *)out[i], k, gf256to16v_add_u512(a, b));
    }
}

/// out += c*in
static inline
void gf256to16_vector_scalar_add_u512(gf256to16 *out,
                                      const gf256to16 c,
                                      const gf256to16 *in,
                                      const uint32_t n) {
    const __m512i s = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)c));
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i a = _mm512_loadu_si512((const void *)in[i]);
        const __m512i o = _mm512_loadu_si512((const void *)out[i]);
        _mm512_storeu_si512((void *)out[i], o ^ gf256to16v_mul_u512(a, s));
    }
    if (i < n) {
        const __mmask8 k = gf256to16v_tail_mask_u512(n - i);
        const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)in[i]);
        const __m512i o = _mm512_maskz_loadu_epi64(k, (const void *)out[i]);
        _mm512_mask_storeu_epi64((void *)out[i], k, o ^ gf256to16v_mul_u512(a, s));
    }
}

/// out = in1 + c*in2
static inline
void gf256to16_vector_scalar_add_u512_v2(gf256to16 *out,
                                         const gf256to16 *in1,
                                         const gf256to16 c,
                                         const gf256to16 *in2,
                                         const uint32_t n) {
    const __m512i s = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)c));
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i a = _mm512_loadu_si512((const void *)in1[i]);
        const __m512i b = _mm512_loadu_si512((const void *)in2[i]);
        _mm512_storeu_si512((void *)out[i], a ^ gf256to16v_mul_u512(b, s));
    }
    if (i < n) {
        const __mmask8 k = gf256to16v_tail_mask_u512(n - i);
        const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)in1[i]);
        const __m512i b = _mm512_maskz_loadu_epi64(k, (const void *)in2[i]);
        _mm512_mask_storeu_epi64((void *)out[i], k, a ^ gf256to16v_mul_u512(b, s));
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to16_vector_mul_acc_u512(gf256to16 out,
                                   const gf256to16 *a,
                                   const gf256to16 *b,
                                   const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)a[i]);
        const __m512i t2 = _mm512_loadu_si512((const void *)b[i]);
        acc ^= gf256to16v_mul_u512(t1, t2);
    }
    if (i < n) {
        // the masked out elements are zero, hence their product is zero
        const __mmask8 k = gf256to16v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(k, (const void *)a[i]);
        const __m512i t2 = _mm512_maskz_loadu_epi64(k, (const void *)b[i]);
        acc ^= gf256to16v_mul_u512(t1, t2);
    }

    const __m128i r = _mm512_extracti32x4_epi32(acc, 0) ^ _mm512_extracti32x4_epi32(acc, 1) ^
                      _mm512_extracti32x4_epi32(acc, 2) ^ _mm512_extracti32x4_epi32(acc, 3);
    _mm_storeu_si128((__m128i *)out, r);
}
#endif // USE_AVX512
