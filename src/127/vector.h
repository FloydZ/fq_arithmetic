#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

/// NOTE: every gf127 element is stored in a single byte

/// \param n[in]: number of elements
/// \return zero initialized vector with n elements
static inline
gf127* gf127_vector_alloc(const uint32_t n) {
    return (gf127 *)calloc(n, sizeof(gf127));
}

/// \param v[in]: vector
/// \param i[in]: index
/// \return v[i]
static inline
gf127 gf127_vector_get(const gf127 *v,
                       const uint32_t i) {
    return v[i];
}

/// \param v[out]: v[i] = a
/// \param i[in]: index
/// \param a[in]: value
static inline
void gf127_vector_set(gf127 *v,
                      const uint32_t i,
                      const gf127 a) {
    v[i] = a;
}

/// \param v[in]: vector to print
/// \param n[in]: number of elements
static inline
void gf127_vector_print(const gf127 *v,
                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%3d ", v[i]);
    }
    printf("\n");
}

/// \param v[out]: = 0
/// \param n[in]: number of elements
static inline
void gf127_vector_zero(gf127 *v,
                       const uint32_t n) {
    memset(v, 0, n * sizeof(gf127));
}

/// \param v[out]: random vector
/// \param n[in]: number of elements
static inline
void gf127_vector_random(gf127 *v,
                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() % PRIME;
    }
}

/// \param out[out]: = in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_copy(gf127 *out,
                       const gf127 *in,
                       const uint32_t n) {
    memmove(out, in, n * sizeof(gf127));
}

/// \param out[in/out]: += in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add(gf127 *out,
                      const gf127 *in,
                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = gf127_add(out[i], in[i]);
    }
}

/// \param out[in/out]: += c*in
/// \param c[in]: scalar
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add(gf127 *out,
                             const gf127 c,
                             const gf127 *in,
                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = gf127_add(out[i], gf127_mul(c, in[i]));
    }
}

/// \param out[out]: = in1 + c*in2
/// \param in1[in]: input vector
/// \param c[in]: scalar
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_v2(gf127 *out,
                                const gf127 *in1,
                                const gf127 c,
                                const gf127 *in2,
                                const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = gf127_add(in1[i], gf127_mul(c, in2[i]));
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline
gf127 gf127_vector_mul_acc(const gf127 *a,
                           const gf127 *b,
                           const uint32_t n) {
    gf127 acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc = gf127_add(acc, gf127_mul(a[i], b[i]));
    }
    return acc;
}


#ifdef USE_AVX2

/// loads `len` < 32 elements, the remaining bytes are zero
static inline
__m256i gf127v_load_tail_u256(const gf127 *in,
                              const uint32_t len) {
    uint8_t tmp[32] __attribute__((aligned(32))) = {0};
    memcpy(tmp, in, len);
    return _mm256_load_si256((const __m256i *)tmp);
}

/// stores the first `len` < 32 elements of `a`
static inline
void gf127v_store_tail_u256(gf127 *out,
                            const __m256i a,
                            const uint32_t len) {
    uint8_t tmp[32] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, a);
    memcpy(out, tmp, len);
}

/// \param out[in/out]: += in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u256(gf127 *out,
                           const gf127 *in,
                           const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 32) <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf127v_add_u256(a, b));
    }

    if (i < n) {
        const __m256i a = gf127v_load_tail_u256(out + i, n - i);
        const __m256i b = gf127v_load_tail_u256(in + i, n - i);
        gf127v_store_tail_u256(out + i, gf127v_add_u256(a, b), n - i);
    }
}

/// \param out[out]: = in1 + in2
/// \param in1[in]: input vector
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u256_v2(gf127 *out,
                              const gf127 *in1,
                              const gf127 *in2,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 32) <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf127v_add_u256(a, b));
    }

    if (i < n) {
        const __m256i a = gf127v_load_tail_u256(in1 + i, n - i);
        const __m256i b = gf127v_load_tail_u256(in2 + i, n - i);
        gf127v_store_tail_u256(out + i, gf127v_add_u256(a, b), n - i);
    }
}

/// \param out[in/out]: += c*in
/// \param c[in]: scalar
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u256(gf127 *out,
                                  const gf127 c,
                                  const gf127 *in,
                                  const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(c);
    uint32_t i = 0;
    for (; (i + 32) <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf127v_add_u256(a, gf127v_mul_u256(s, b)));
    }

    if (i < n) {
        const __m256i a = gf127v_load_tail_u256(out + i, n - i);
        const __m256i b = gf127v_load_tail_u256(in + i, n - i);
        gf127v_store_tail_u256(out + i, gf127v_add_u256(a, gf127v_mul_u256(s, b)), n - i);
    }
}

/// \param out[out]: = in1 + c*in2
/// \param in1[in]: input vector
/// \param c[in]: scalar
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u256_v2(gf127 *out,
                                     const gf127 *in1,
                                     const gf127 c,
                                     const gf127 *in2,
                                     const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(c);
    uint32_t i = 0;
    for (; (i + 32) <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf127v_add_u256(a, gf127v_mul_u256(s, b)));
    }

    if (i < n) {
        const __m256i a = gf127v_load_tail_u256(in1 + i, n - i);
        const __m256i b = gf127v_load_tail_u256(in2 + i, n - i);
        gf127v_store_tail_u256(out + i, gf127v_add_u256(a, gf127v_mul_u256(s, b)), n - i);
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline
gf127 gf127_vector_mul_acc_u256(const gf127 *a,
                                const gf127 *b,
                                const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i + 32) <= n; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = gf127v_add_u256(acc, gf127v_mul_u256(t1, t2));
    }

    if (i < n) {
        const __m256i t1 = gf127v_load_tail_u256(a + i, n - i);
        const __m256i t2 = gf127v_load_tail_u256(b + i, n - i);
        acc = gf127v_add_u256(acc, gf127v_mul_u256(t1, t2));
    }

    uint8_t tmp[32] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, acc);
    gf127 r = 0;
    for (uint32_t k = 0; k < 32; k++) {
        r = gf127_add(r, tmp[k]);
    }
    return r;
}
#endif


#ifdef USE_AVX512

/// \param out[in/out]: += in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u512(gf127 *out,
                           const gf127 *in,
                           const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const __m512i *)(out + i));
        const __m512i b = _mm512_loadu_si512((const __m512i *)(in + i));
        _mm512_storeu_si512((__m512i *)(out + i), gf127v_add_u512(a, b));
    }

    if (i < n) {
        const __mmask64 m = (1ull << (n - i)) - 1ull;
        const __m512i a = _mm512_maskz_loadu_epi8(m, out + i);
        const __m512i b = _mm512_maskz_loadu_epi8(m, in + i);
        _mm512_mask_storeu_epi8(out + i, m, gf127v_add_u512(a, b));
    }
}

/// \param out[out]: = in1 + in2
/// \param in1[in]: input vector
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u512_v2(gf127 *out,
                              const gf127 *in1,
                              const gf127 *in2,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const __m512i *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const __m512i *)(in2 + i));
        _mm512_storeu_si512((__m512i *)(out + i), gf127v_add_u512(a, b));
    }

    if (i < n) {
        const __mmask64 m = (1ull << (n - i)) - 1ull;
        const __m512i a = _mm512_maskz_loadu_epi8(m, in1 + i);
        const __m512i b = _mm512_maskz_loadu_epi8(m, in2 + i);
        _mm512_mask_storeu_epi8(out + i, m, gf127v_add_u512(a, b));
    }
}

/// \param out[in/out]: += c*in
/// \param c[in]: scalar
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u512(gf127 *out,
                                  const gf127 c,
                                  const gf127 *in,
                                  const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(c);
    uint32_t i = 0;
    for (; (i + 64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const __m512i *)(out + i));
        const __m512i b = _mm512_loadu_si512((const __m512i *)(in + i));
        _mm512_storeu_si512((__m512i *)(out + i), gf127v_add_u512(a, gf127v_mul_u512(s, b)));
    }

    if (i < n) {
        const __mmask64 m = (1ull << (n - i)) - 1ull;
        const __m512i a = _mm512_maskz_loadu_epi8(m, out + i);
        const __m512i b = _mm512_maskz_loadu_epi8(m, in + i);
        _mm512_mask_storeu_epi8(out + i, m, gf127v_add_u512(a, gf127v_mul_u512(s, b)));
    }
}

/// \param out[out]: = in1 + c*in2
/// \param in1[in]: input vector
/// \param c[in]: scalar
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u512_v2(gf127 *out,
                                     const gf127 *in1,
                                     const gf127 c,
                                     const gf127 *in2,
                                     const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(c);
    uint32_t i = 0;
    for (; (i + 64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const __m512i *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const __m512i *)(in2 + i));
        _mm512_storeu_si512((__m512i *)(out + i), gf127v_add_u512(a, gf127v_mul_u512(s, b)));
    }

    if (i < n) {
        const __mmask64 m = (1ull << (n - i)) - 1ull;
        const __m512i a = _mm512_maskz_loadu_epi8(m, in1 + i);
        const __m512i b = _mm512_maskz_loadu_epi8(m, in2 + i);
        _mm512_mask_storeu_epi8(out + i, m, gf127v_add_u512(a, gf127v_mul_u512(s, b)));
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline
gf127 gf127_vector_mul_acc_u512(const gf127 *a,
                                const gf127 *b,
                                const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i + 64) <= n; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const __m512i *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const __m512i *)(b + i));
        acc = gf127v_add_u512(acc, gf127v_mul_u512(t1, t2));
    }

    if (i < n) {
        const __mmask64 m = (1ull << (n - i)) - 1ull;
        const __m512i t1 = _mm512_maskz_loadu_epi8(m, a + i);
        const __m512i t2 = _mm512_maskz_loadu_epi8(m, b + i);
        acc = gf127v_add_u512(acc, gf127v_mul_u512(t1, t2));
    }

    uint8_t tmp[64] __attribute__((aligned(64)));
    _mm512_store_si512((__m512i *)tmp, acc);
    gf127 r = 0;
    for (uint32_t k = 0; k < 64; k++) {
        r = gf127_add(r, tmp[k]);
    }
    return r;
}
#endif


#ifdef USE_NEON

/// loads `len` < 16 elements, the remaining bytes are zero
static inline
uint8x16_t gf127v_load_tail_u128(const gf127 *in,
                                 const uint32_t len) {
    uint8_t tmp[16] = {0};
    memcpy(tmp, in, len);
    return vld1q_u8(tmp);
}

/// stores the first `len` < 16 elements of `a`
static inline
void gf127v_store_tail_u128(gf127 *out,
                            const uint8x16_t a,
                            const uint32_t len) {
    uint8_t tmp[16];
    vst1q_u8(tmp, a);
    memcpy(out, tmp, len);
}

/// \param out[in/out]: += in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u128(gf127 *out,
                           const gf127 *in,
                           const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 16) <= n; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in + i);
        vst1q_u8(out + i, gf127v_add_u128(a, b));
    }

    if (i < n) {
        const uint8x16_t a = gf127v_load_tail_u128(out + i, n - i);
        const uint8x16_t b = gf127v_load_tail_u128(in + i, n - i);
        gf127v_store_tail_u128(out + i, gf127v_add_u128(a, b), n - i);
    }
}

/// \param out[out]: = in1 + in2
/// \param in1[in]: input vector
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_add_u128_v2(gf127 *out,
                              const gf127 *in1,
                              const gf127 *in2,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 16) <= n; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf127v_add_u128(a, b));
    }

    if (i < n) {
        const uint8x16_t a = gf127v_load_tail_u128(in1 + i, n - i);
        const uint8x16_t b = gf127v_load_tail_u128(in2 + i, n - i);
        gf127v_store_tail_u128(out + i, gf127v_add_u128(a, b), n - i);
    }
}

/// \param out[in/out]: += c*in
/// \param c[in]: scalar
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u128(gf127 *out,
                                  const gf127 c,
                                  const gf127 *in,
                                  const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(c);
    uint32_t i = 0;
    for (; (i + 16) <= n; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in + i);
        vst1q_u8(out + i, gf127v_add_u128(a, gf127v_mul_u128(s, b)));
    }

    if (i < n) {
        const uint8x16_t a = gf127v_load_tail_u128(out + i, n - i);
        const uint8x16_t b = gf127v_load_tail_u128(in + i, n - i);
        gf127v_store_tail_u128(out + i, gf127v_add_u128(a, gf127v_mul_u128(s, b)), n - i);
    }
}

/// \param out[out]: = in1 + c*in2
/// \param in1[in]: input vector
/// \param c[in]: scalar
/// \param in2[in]: input vector
/// \param n[in]: number of elements
static inline
void gf127_vector_scalar_add_u128_v2(gf127 *out,
                                     const gf127 *in1,
                                     const gf127 c,
                                     const gf127 *in2,
                                     const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(c);
    uint32_t i = 0;
    for (; (i + 16) <= n; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf127v_add_u128(a, gf127v_mul_u128(s, b)));
    }

    if (i < n) {
        const uint8x16_t a = gf127v_load_tail_u128(in1 + i, n - i);
        const uint8x16_t b = gf127v_load_tail_u128(in2 + i, n - i);
        gf127v_store_tail_u128(out + i, gf127v_add_u128(a, gf127v_mul_u128(s, b)), n - i);
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline
gf127 gf127_vector_mul_acc_u128(const gf127 *a,
                                const gf127 *b,
                                const uint32_t n) {
    uint8x16_t acc = vdupq_n_u8(0);
    uint32_t i = 0;
    for (; (i + 16) <= n; i += 16) {
        const uint8x16_t t1 = vld1q_u8(a + i);
        const uint8x16_t t2 = vld1q_u8(b + i);
        acc = gf127v_add_u128(acc, gf127v_mul_u128(t1, t2));
    }

    if (i < n) {
        const uint8x16_t t1 = gf127v_load_tail_u128(a + i, n - i);
        const uint8x16_t t2 = gf127v_load_tail_u128(b + i, n - i);
        acc = gf127v_add_u128(acc, gf127v_mul_u128(t1, t2));
    }

    uint8_t tmp[16];
    vst1q_u8(tmp, acc);
    gf127 r = 0;
    for (uint32_t k = 0; k < 16; k++) {
        r = gf127_add(r, tmp[k]);
    }
    return r;
}
#endif
