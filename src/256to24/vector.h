#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

// a vector of n elements is an array `gf256to24 v[n]` (24*n bytes)

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline
gf256to24* gf256to24_vector_alloc(const uint32_t n) {
    return (gf256to24 *)calloc(n, sizeof(gf256to24));
}

/// out = v[i]
static inline
void gf256to24_vector_get(gf256to24 out,
                          const gf256to24 *v,
                          const uint32_t i) {
    memcpy(out, v[i], sizeof(gf256to24));
}

/// v[i] = a
static inline
void gf256to24_vector_set(gf256to24 *v,
                          const uint32_t i,
                          const gf256to24 a) {
    memcpy(v[i], a, sizeof(gf256to24));
}

/// prints each element as its 24 bytes (byte 0 first)
static inline
void gf256to24_vector_print(const gf256to24 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < 24; j++) {
            printf("%02x", v[i][j]);
        }
        printf(" ");
    }
    printf("\n");
}

/// v = 0
static inline
void gf256to24_vector_zero(gf256to24 *v,
                           const uint32_t n) {
    memset(v, 0, (size_t)n * sizeof(gf256to24));
}

/// v = random
static inline
void gf256to24_vector_random(gf256to24 *v,
                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < 24; j++) {
            v[i][j] = (uint8_t)rand();
        }
    }
}

/// out = in
static inline
void gf256to24_vector_copy(gf256to24 *__restrict__ out,
                           const gf256to24 *__restrict__ in,
                           const uint32_t n) {
    memcpy(out, in, (size_t)n * sizeof(gf256to24));
}

/// out += in
static inline
void gf256to24_vector_add(gf256to24 *out,
                          const gf256to24 *in,
                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf256to24_add(out[i], out[i], in[i]);
    }
}

/// out += c*in
static inline
void gf256to24_vector_scalar_add(gf256to24 *out,
                                 const gf256to24 c,
                                 const gf256to24 *in,
                                 const uint32_t n) {
    gf256to24 t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to24_mul(t, c, in[i]);
        gf256to24_add(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256to24_vector_scalar_add_v2(gf256to24 *out,
                                    const gf256to24 *in1,
                                    const gf256to24 c,
                                    const gf256to24 *in2,
                                    const uint32_t n) {
    gf256to24 t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to24_mul(t, c, in2[i]);
        gf256to24_add(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to24_vector_mul_acc(gf256to24 out,
                              const gf256to24 *a,
                              const gf256to24 *b,
                              const uint32_t n) {
    gf256to24 acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256to24_mul(t, a[i], b[i]);
        gf256to24_add(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256to24));
}

#ifdef USE_AVX2
/// \return mask for loading/storing a single element: [-1, -1, -1, 0]
static inline __m256i gf256to24v_mask_u256(void) {
    return _mm256_setr_epi64x(-1, -1, -1, 0);
}

/// loads a single element: [a_0, a_1, a_2, 0]
static inline __m256i gf256to24v_load_u256(const uint8_t *a) {
    return _mm256_maskload_epi64((const long long *)a, gf256to24v_mask_u256());
}

/// stores a single element
static inline void gf256to24v_store_u256(uint8_t *a, const __m256i b) {
    _mm256_maskstore_epi64((long long *)a, gf256to24v_mask_u256(), b);
}

/// out = in1 ^ in2 over `bytes` bytes, `bytes` must be a multiple of 8
static inline void gf256to24_xor_bytes_u256(uint8_t *out,
                                            const uint8_t *in1,
                                            const uint8_t *in2,
                                            const size_t bytes) {
    size_t i = 0;
    for (; (i + 32) <= bytes; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), a ^ b);
    }
    for (; i < bytes; i += 8) {
        uint64_t a, b;
        memcpy(&a, in1 + i, 8);
        memcpy(&b, in2 + i, 8);
        a ^= b;
        memcpy(out + i, &a, 8);
    }
}

/// out += in
static inline
void gf256to24_vector_add_u256(gf256to24 *out,
                               const gf256to24 *in,
                               const uint32_t n) {
    gf256to24_xor_bytes_u256((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                             (size_t)n * sizeof(gf256to24));
}

/// out = in1 + in2
static inline
void gf256to24_vector_add_u256_v2(gf256to24 *out,
                                  const gf256to24 *in1,
                                  const gf256to24 *in2,
                                  const uint32_t n) {
    gf256to24_xor_bytes_u256((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                             (size_t)n * sizeof(gf256to24));
}

/// out += c*in
static inline
void gf256to24_vector_scalar_add_u256(gf256to24 *out,
                                      const gf256to24 c,
                                      const gf256to24 *in,
                                      const uint32_t n) {
    const __m256i s = gf256to24v_load_u256(c);
    for (uint32_t i = 0; i < n; i++) {
        const __m256i a = gf256to24v_load_u256(in[i]);
        const __m256i o = gf256to24v_load_u256(out[i]);
        gf256to24v_store_u256(out[i], o ^ gf256to24v_mul_u256(a, s));
    }
}

/// out = in1 + c*in2
static inline
void gf256to24_vector_scalar_add_u256_v2(gf256to24 *out,
                                         const gf256to24 *in1,
                                         const gf256to24 c,
                                         const gf256to24 *in2,
                                         const uint32_t n) {
    const __m256i s = gf256to24v_load_u256(c);
    for (uint32_t i = 0; i < n; i++) {
        const __m256i a = gf256to24v_load_u256(in1[i]);
        const __m256i b = gf256to24v_load_u256(in2[i]);
        gf256to24v_store_u256(out[i], a ^ gf256to24v_mul_u256(b, s));
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to24_vector_mul_acc_u256(gf256to24 out,
                                   const gf256to24 *a,
                                   const gf256to24 *b,
                                   const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    for (uint32_t i = 0; i < n; i++) {
        acc ^= gf256to24v_mul_u256(gf256to24v_load_u256(a[i]), gf256to24v_load_u256(b[i]));
    }
    gf256to24v_store_u256(out, acc);
}
#endif // USE_AVX2

#ifdef USE_NEON
/// loads a single element: val[0] = [a_0, a_1], val[1] = [a_2, 0]
static inline uint64x2x2_t gf256to24v_load_u128(const uint8_t *a) {
    uint64x2x2_t r;
    r.val[0] = vreinterpretq_u64_u8(vld1q_u8(a));
    r.val[1] = vcombine_u64(vreinterpret_u64_u8(vld1_u8(a + 16)), vdup_n_u64(0));
    return r;
}

/// stores a single element
static inline void gf256to24v_store_u128(uint8_t *a, const uint64x2x2_t b) {
    vst1q_u8(a, vreinterpretq_u8_u64(b.val[0]));
    vst1_u8(a + 16, vreinterpret_u8_u64(vget_low_u64(b.val[1])));
}

/// out = in1 ^ in2 over `bytes` bytes, `bytes` must be a multiple of 8
static inline void gf256to24_xor_bytes_u128(uint8_t *out,
                                            const uint8_t *in1,
                                            const uint8_t *in2,
                                            const size_t bytes) {
    size_t i = 0;
    for (; (i + 16) <= bytes; i += 16) {
        vst1q_u8(out + i, veorq_u8(vld1q_u8(in1 + i), vld1q_u8(in2 + i)));
    }
    for (; i < bytes; i += 8) {
        vst1_u8(out + i, veor_u8(vld1_u8(in1 + i), vld1_u8(in2 + i)));
    }
}

/// out += in
static inline
void gf256to24_vector_add_u128(gf256to24 *out,
                               const gf256to24 *in,
                               const uint32_t n) {
    gf256to24_xor_bytes_u128((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                             (size_t)n * sizeof(gf256to24));
}

/// out = in1 + in2
static inline
void gf256to24_vector_add_u128_v2(gf256to24 *out,
                                  const gf256to24 *in1,
                                  const gf256to24 *in2,
                                  const uint32_t n) {
    gf256to24_xor_bytes_u128((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                             (size_t)n * sizeof(gf256to24));
}

/// out += c*in
static inline
void gf256to24_vector_scalar_add_u128(gf256to24 *out,
                                      const gf256to24 c,
                                      const gf256to24 *in,
                                      const uint32_t n) {
    const uint64x2x2_t s = gf256to24v_load_u128(c);
    for (uint32_t i = 0; i < n; i++) {
        const uint64x2x2_t t = gf256to24v_mul_u128(gf256to24v_load_u128(in[i]), s);
        gf256to24v_store_u128(out[i], gf256to24v_add_u128(gf256to24v_load_u128(out[i]), t));
    }
}

/// out = in1 + c*in2
static inline
void gf256to24_vector_scalar_add_u128_v2(gf256to24 *out,
                                         const gf256to24 *in1,
                                         const gf256to24 c,
                                         const gf256to24 *in2,
                                         const uint32_t n) {
    const uint64x2x2_t s = gf256to24v_load_u128(c);
    for (uint32_t i = 0; i < n; i++) {
        const uint64x2x2_t t = gf256to24v_mul_u128(gf256to24v_load_u128(in2[i]), s);
        gf256to24v_store_u128(out[i], gf256to24v_add_u128(gf256to24v_load_u128(in1[i]), t));
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to24_vector_mul_acc_u128(gf256to24 out,
                                   const gf256to24 *a,
                                   const gf256to24 *b,
                                   const uint32_t n) {
    uint64x2x2_t acc;
    acc.val[0] = vdupq_n_u64(0);
    acc.val[1] = vdupq_n_u64(0);
    for (uint32_t i = 0; i < n; i++) {
        const uint64x2x2_t t = gf256to24v_mul_u128(gf256to24v_load_u128(a[i]),
                                                   gf256to24v_load_u128(b[i]));
        acc = gf256to24v_add_u128(acc, t);
    }
    gf256to24v_store_u128(out, acc);
}
#endif // USE_NEON

#ifdef USE_AVX512
/// loads `k` (1 or 2) consecutive elements:
///     [a_0, a_1, a_2, 0, a'_0, a'_1, a'_2, 0]
static inline __m512i gf256to24v_load_u512(const uint8_t *a, const uint32_t k) {
    return _mm512_maskz_expandloadu_epi64(k == 2 ? 0x77 : 0x07, (const void *)a);
}

/// stores `k` (1 or 2) consecutive elements
static inline void gf256to24v_store_u512(uint8_t *a, const __m512i b, const uint32_t k) {
    _mm512_mask_compressstoreu_epi64((void *)a, k == 2 ? 0x77 : 0x07, b);
}

/// out = in1 ^ in2 over `bytes` bytes, `bytes` must be a multiple of 8
static inline void gf256to24_xor_bytes_u512(uint8_t *out,
                                            const uint8_t *in1,
                                            const uint8_t *in2,
                                            const size_t bytes) {
    size_t i = 0;
    for (; (i + 64) <= bytes; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), a ^ b);
    }
    if (i < bytes) {
        const __mmask8 k = (__mmask8)((1u << ((bytes - i) / 8u)) - 1u);
        const __m512i a = _mm512_maskz_loadu_epi64(k, (const void *)(in1 + i));
        const __m512i b = _mm512_maskz_loadu_epi64(k, (const void *)(in2 + i));
        _mm512_mask_storeu_epi64((void *)(out + i), k, a ^ b);
    }
}

/// out += in
static inline
void gf256to24_vector_add_u512(gf256to24 *out,
                               const gf256to24 *in,
                               const uint32_t n) {
    gf256to24_xor_bytes_u512((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                             (size_t)n * sizeof(gf256to24));
}

/// out = in1 + in2
static inline
void gf256to24_vector_add_u512_v2(gf256to24 *out,
                                  const gf256to24 *in1,
                                  const gf256to24 *in2,
                                  const uint32_t n) {
    gf256to24_xor_bytes_u512((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                             (size_t)n * sizeof(gf256to24));
}

/// out += c*in
static inline
void gf256to24_vector_scalar_add_u512(gf256to24 *out,
                                      const gf256to24 c,
                                      const gf256to24 *in,
                                      const uint32_t n) {
    const __m512i s = _mm512_broadcast_i64x4(_mm512_castsi512_si256(gf256to24v_load_u512(c, 1)));
    for (uint32_t i = 0; i < n; i += 2) {
        const uint32_t k = (i + 2) <= n ? 2 : 1;
        const __m512i a = gf256to24v_load_u512(in[i], k);
        const __m512i o = gf256to24v_load_u512(out[i], k);
        gf256to24v_store_u512(out[i], o ^ gf256to24v_mul_u512(a, s), k);
    }
}

/// out = in1 + c*in2
static inline
void gf256to24_vector_scalar_add_u512_v2(gf256to24 *out,
                                         const gf256to24 *in1,
                                         const gf256to24 c,
                                         const gf256to24 *in2,
                                         const uint32_t n) {
    const __m512i s = _mm512_broadcast_i64x4(_mm512_castsi512_si256(gf256to24v_load_u512(c, 1)));
    for (uint32_t i = 0; i < n; i += 2) {
        const uint32_t k = (i + 2) <= n ? 2 : 1;
        const __m512i a = gf256to24v_load_u512(in1[i], k);
        const __m512i b = gf256to24v_load_u512(in2[i], k);
        gf256to24v_store_u512(out[i], a ^ gf256to24v_mul_u512(b, s), k);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256to24_vector_mul_acc_u512(gf256to24 out,
                                   const gf256to24 *a,
                                   const gf256to24 *b,
                                   const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    for (uint32_t i = 0; i < n; i += 2) {
        // if only a single element is loaded, the upper lane is zero
        const uint32_t k = (i + 2) <= n ? 2 : 1;
        acc ^= gf256to24v_mul_u512(gf256to24v_load_u512(a[i], k), gf256to24v_load_u512(b[i], k));
    }

    const __m512i r = acc ^ _mm512_shuffle_i64x2(acc, acc, 0x4E);
    gf256to24v_store_u512(out, r, 1);
}
#endif // USE_AVX512
