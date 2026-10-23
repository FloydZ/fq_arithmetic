#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

// a vector of n elements is an array `gf256toX v[n]` (EXTENSION_DEGREE*n bytes)

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline
gf256toX* gf256toX_vector_alloc(const uint32_t n) {
    return (gf256toX *)calloc(n, sizeof(gf256toX));
}

/// out = v[i]
static inline
void gf256toX_vector_get(gf256toX out,
                         const gf256toX *v,
                         const uint32_t i) {
    memcpy(out, v[i], sizeof(gf256toX));
}

/// v[i] = a
static inline
void gf256toX_vector_set(gf256toX *v,
                         const uint32_t i,
                         const gf256toX a) {
    memcpy(v[i], a, sizeof(gf256toX));
}

/// prints each element as its EXTENSION_DEGREE bytes (byte 0 first)
static inline
void gf256toX_vector_print(const gf256toX *v,
                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < EXTENSION_DEGREE; j++) {
            printf("%02x", v[i][j]);
        }
        printf(" ");
    }
    printf("\n");
}

/// v = 0
static inline
void gf256toX_vector_zero(gf256toX *v,
                          const uint32_t n) {
    memset(v, 0, (size_t)n * sizeof(gf256toX));
}

/// v = random
static inline
void gf256toX_vector_random(gf256toX *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < EXTENSION_DEGREE; j++) {
            v[i][j] = (uint8_t)rand();
        }
    }
}

/// out = in
static inline
void gf256toX_vector_copy(gf256toX *__restrict__ out,
                          const gf256toX *__restrict__ in,
                          const uint32_t n) {
    memcpy(out, in, (size_t)n * sizeof(gf256toX));
}

/// out += in
static inline
void gf256toX_vector_add(gf256toX *out,
                         const gf256toX *in,
                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf256toX_add(out[i], out[i], in[i]);
    }
}

/// out += c*in
static inline
void gf256toX_vector_scalar_add(gf256toX *out,
                                const gf256toX c,
                                const gf256toX *in,
                                const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toX_mul(t, c, in[i]);
        gf256toX_add(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256toX_vector_scalar_add_v2(gf256toX *out,
                                   const gf256toX *in1,
                                   const gf256toX c,
                                   const gf256toX *in2,
                                   const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toX_mul(t, c, in2[i]);
        gf256toX_add(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256toX_vector_mul_acc(gf256toX out,
                             const gf256toX *a,
                             const gf256toX *b,
                             const uint32_t n) {
    gf256toX acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toX_mul(t, a[i], b[i]);
        gf256toX_add(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256toX));
}

#ifdef USE_AVX2
/// out = in1 ^ in2 over `bytes` bytes
static inline void gf256toX_xor_bytes_u256(uint8_t *out,
                                           const uint8_t *in1,
                                           const uint8_t *in2,
                                           const size_t bytes) {
    size_t i = 0;
    for (; (i + 32) <= bytes; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), a ^ b);
    }
    for (; i < bytes; i++) {
        out[i] = in1[i] ^ in2[i];
    }
}

/// out += in
static inline
void gf256toX_vector_add_u256(gf256toX *out,
                              const gf256toX *in,
                              const uint32_t n) {
    gf256toX_xor_bytes_u256((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                            (size_t)n * sizeof(gf256toX));
}

/// out = in1 + in2
static inline
void gf256toX_vector_add_u256_v2(gf256toX *out,
                                 const gf256toX *in1,
                                 const gf256toX *in2,
                                 const uint32_t n) {
    gf256toX_xor_bytes_u256((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                            (size_t)n * sizeof(gf256toX));
}

/// out += c*in
static inline
void gf256toX_vector_scalar_add_u256(gf256toX *out,
                                     const gf256toX c,
                                     const gf256toX *in,
                                     const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u256(t, c, in[i]);
        gf256toXv_add_u256(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256toX_vector_scalar_add_u256_v2(gf256toX *out,
                                        const gf256toX *in1,
                                        const gf256toX c,
                                        const gf256toX *in2,
                                        const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u256(t, c, in2[i]);
        gf256toXv_add_u256(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256toX_vector_mul_acc_u256(gf256toX out,
                                  const gf256toX *a,
                                  const gf256toX *b,
                                  const uint32_t n) {
    gf256toX acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u256(t, a[i], b[i]);
        gf256toXv_add_u256(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256toX));
}
#endif // USE_AVX2

#ifdef USE_NEON
/// out = in1 ^ in2 over `bytes` bytes
static inline void gf256toX_xor_bytes_u128(uint8_t *out,
                                           const uint8_t *in1,
                                           const uint8_t *in2,
                                           const size_t bytes) {
    size_t i = 0;
    for (; (i + 16) <= bytes; i += 16) {
        vst1q_u8(out + i, veorq_u8(vld1q_u8(in1 + i), vld1q_u8(in2 + i)));
    }
    for (; i < bytes; i++) {
        out[i] = in1[i] ^ in2[i];
    }
}

/// out += in
static inline
void gf256toX_vector_add_u128(gf256toX *out,
                              const gf256toX *in,
                              const uint32_t n) {
    gf256toX_xor_bytes_u128((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                            (size_t)n * sizeof(gf256toX));
}

/// out = in1 + in2
static inline
void gf256toX_vector_add_u128_v2(gf256toX *out,
                                 const gf256toX *in1,
                                 const gf256toX *in2,
                                 const uint32_t n) {
    gf256toX_xor_bytes_u128((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                            (size_t)n * sizeof(gf256toX));
}

/// out += c*in
static inline
void gf256toX_vector_scalar_add_u128(gf256toX *out,
                                     const gf256toX c,
                                     const gf256toX *in,
                                     const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u128(t, c, in[i]);
        gf256toXv_add_u128(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256toX_vector_scalar_add_u128_v2(gf256toX *out,
                                        const gf256toX *in1,
                                        const gf256toX c,
                                        const gf256toX *in2,
                                        const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u128(t, c, in2[i]);
        gf256toXv_add_u128(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256toX_vector_mul_acc_u128(gf256toX out,
                                  const gf256toX *a,
                                  const gf256toX *b,
                                  const uint32_t n) {
    gf256toX acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u128(t, a[i], b[i]);
        gf256toXv_add_u128(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256toX));
}
#endif // USE_NEON

#ifdef USE_AVX512
/// out = in1 ^ in2 over `bytes` bytes
static inline void gf256toX_xor_bytes_u512(uint8_t *out,
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
        const __mmask64 k = (~0ull) >> (64u - (bytes - i));
        const __m512i a = _mm512_maskz_loadu_epi8(k, (const void *)(in1 + i));
        const __m512i b = _mm512_maskz_loadu_epi8(k, (const void *)(in2 + i));
        _mm512_mask_storeu_epi8((void *)(out + i), k, a ^ b);
    }
}

/// out += in
static inline
void gf256toX_vector_add_u512(gf256toX *out,
                              const gf256toX *in,
                              const uint32_t n) {
    gf256toX_xor_bytes_u512((uint8_t *)out, (const uint8_t *)out, (const uint8_t *)in,
                            (size_t)n * sizeof(gf256toX));
}

/// out = in1 + in2
static inline
void gf256toX_vector_add_u512_v2(gf256toX *out,
                                 const gf256toX *in1,
                                 const gf256toX *in2,
                                 const uint32_t n) {
    gf256toX_xor_bytes_u512((uint8_t *)out, (const uint8_t *)in1, (const uint8_t *)in2,
                            (size_t)n * sizeof(gf256toX));
}

/// out += c*in
static inline
void gf256toX_vector_scalar_add_u512(gf256toX *out,
                                     const gf256toX c,
                                     const gf256toX *in,
                                     const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u512(t, c, in[i]);
        gf256toXv_add_u512(out[i], out[i], t);
    }
}

/// out = in1 + c*in2
static inline
void gf256toX_vector_scalar_add_u512_v2(gf256toX *out,
                                        const gf256toX *in1,
                                        const gf256toX c,
                                        const gf256toX *in2,
                                        const uint32_t n) {
    gf256toX t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u512(t, c, in2[i]);
        gf256toXv_add_u512(out[i], in1[i], t);
    }
}

/// out = sum_i a[i]*b[i]
static inline
void gf256toX_vector_mul_acc_u512(gf256toX out,
                                  const gf256toX *a,
                                  const gf256toX *b,
                                  const uint32_t n) {
    gf256toX acc = {0}, t;
    for (uint32_t i = 0; i < n; i++) {
        gf256toXv_mul_u512(t, a[i], b[i]);
        gf256toXv_add_u512(acc, acc, t);
    }
    memcpy(out, acc, sizeof(gf256toX));
}
#endif // USE_AVX512
