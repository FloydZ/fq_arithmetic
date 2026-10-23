#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"

/// NOTE: vectors are bit packed: element i is bit (i%8) of byte i/8.
/// All functions taking a length `n` count elements (bits), not bytes.
/// Bits beyond element n-1 in the last byte are never modified by the
/// arithmetic functions below (except `zero`, `random`, `scalar`).
#define gf2_vector_size(N) (((N)+7)/8)

/// \param n[in]: number of elements
/// \return mask of the valid bits in the last byte, if n%8 != 0
static inline uint8_t gf2_vector_tail_mask(const uint32_t n) {
    return (uint8_t)((1u << (n % 8u)) - 1u);
}

/// \param n number of elements
/// \return zero initialized vector of n elements
static inline gf2* gf2_vector_alloc(const uint32_t n) {
    return (gf2 *)calloc(gf2_vector_size(n) + 1, sizeof(gf2));
}

/// \param vector
/// \param i
/// \return vector[i]
static inline gf2 gf2_vector_get(const gf2 *vector,
                                 const uint32_t i) {
    return (vector[i / (8)] >> (i % 8)) & 0x01;
}

/// vector[i] = scalar
/// \param vector
/// \param i
/// \param scalar
static inline void gf2_vector_set(gf2 *vector,
                                  const uint32_t i,
                                  const gf2 scalar) {
    const uint8_t mask = 0xff ^ (1 << (i % 8));
    vector[i/8] = (vector[i/8] & mask) ^ ((scalar & 1u) << (i % 8));
}

/// \param v[in]: vector to print
/// \param n[in]: number of elements
static inline void gf2_vector_print(const gf2 *v,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%u", (unsigned)gf2_vector_get(v, i));
    }
    printf("\n");
}

/// \param[in/out] arg1 = random, bits beyond `d` are zeroed
static inline void gf2_vector_random(gf2 *arg1,
                                     const uint32_t d) {
    const uint32_t limbs = (d+7)/8;
    for (uint32_t i = 0; i < limbs; i++) {
        arg1[i] = rand();
    }

    const uint32_t l = d%8;
    if (l) {
        const uint8_t mask = (1u << l) - 1u;
        arg1[limbs - 1] &= mask;
    }
}

/// \param[in/out] arg1 = 0
static inline void gf2_vector_zero(gf2 *arg1,
                                  const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = 0;
    }
}

/// \param out[out]: out = in
/// \param in[in]: input vector
/// \param n[in]: number of elements
static inline void gf2_vector_copy(gf2 *out,
                                   const gf2 *in,
                                   const uint32_t n) {
    const uint32_t nb = n / 8;
    memcpy(out, in, nb);
    if (n % 8) {
        const uint8_t m = gf2_vector_tail_mask(n);
        out[nb] = (out[nb] & (uint8_t)~m) | (in[nb] & m);
    }
}

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add(gf2 *out,
                                  const gf2 *in,
                                  const uint32_t n) {
    const uint32_t nb = n / 8;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] ^= in[i];
    }

    if (n % 8) {
        out[nb] ^= in[nb] & gf2_vector_tail_mask(n);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_v2(gf2 *out,
                                     const gf2 *in1,
                                     const gf2 *in2,
                                     const uint32_t n) {
    const uint32_t nb = n / 8;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] = in1[i] ^ in2[i];
    }

    if (n % 8) {
        const uint8_t m = gf2_vector_tail_mask(n);
        out[nb] = (out[nb] & (uint8_t)~m) | ((in1[nb] ^ in2[nb]) & m);
    }
}

/// arg1 = scalar*arg2
/// \param arg1
/// \param arg2
/// \param value
/// \param d
static inline void gf2_vector_scalar(gf2 *arg1,
                                     const gf2 *arg2,
                                     const gf2 value,
                                     const uint32_t d) {
    const uint8_t m = -(value & 1u);
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] & m;
    }
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add(gf2 *out,
                                         const gf2 c,
                                         const gf2 *in,
                                         const uint32_t n) {
    const uint8_t m = -(c & 1u);
    const uint32_t nb = n / 8;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] ^= in[i] & m;
    }

    if (n % 8) {
        out[nb] ^= in[nb] & m & gf2_vector_tail_mask(n);
    }
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_v2(gf2 *out,
                                            const gf2 *in1,
                                            const gf2 c,
                                            const gf2 *in2,
                                            const uint32_t n) {
    const uint8_t m = -(c & 1u);
    const uint32_t nb = n / 8;
    for (uint32_t i = 0; i < nb; i++) {
        out[i] = in1[i] ^ (in2[i] & m);
    }

    if (n % 8) {
        const uint8_t t = gf2_vector_tail_mask(n);
        out[nb] = (out[nb] & (uint8_t)~t) | ((in1[nb] ^ (in2[nb] & m)) & t);
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf2 gf2_vector_mul_acc(const gf2 *a,
                                     const gf2 *b,
                                     const uint32_t n) {
    const uint32_t nb = n / 8;
    uint8_t acc = 0;
    for (uint32_t i = 0; i < nb; i++) {
        acc ^= a[i] & b[i];
    }

    if (n % 8) {
        acc ^= a[nb] & b[nb] & gf2_vector_tail_mask(n);
    }

    return __builtin_popcount(acc) & 1u;
}

/// evaluates the polynomial arg1[0] + arg1[1]*x + ... + arg1[d-1]*x^{d-1}
/// \return arg1(value)
static inline gf2 gf2_vector_eval(const gf2 *arg1,
                                  const gf2 value,
                                  const uint32_t d) {
	gf2 ret = 0;
	for (int64_t i = (int64_t)d - 1; i >= 0; i--) {
        const gf2 t = gf2_vector_get(arg1, i);
		ret = gf2_add(gf2_mul(ret, value), t);
	}

	return ret;
}

#ifdef USE_AVX2

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u256(gf2 *out,
                                       const gf2 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in  + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2v_add_u256(a, b));
    }

    gf2_vector_add(out + i, in + i, n - 8*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u256_v2(gf2 *out,
                                          const gf2 *in1,
                                          const gf2 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2v_add_u256(a, b));
    }

    gf2_vector_add_v2(out + i, in1 + i, in2 + i, n - 8*i);
}

/// out = scalar*in1
/// \param out
/// \param in1
/// \param scalar
/// \param d
static inline void gf2_vector_scalar_u256(gf2 *out,
                                          const gf2 *in1,
                                          const gf2 scalar,
                                          const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;
    const uint8_t m = -(scalar & 1u);
    const __m256i s256 = _mm256_set1_epi8(m);
    const __m128i s128 = _mm_set1_epi8(m);

    // avx2 code
    while (nbytes >= 32u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = t1&s256;
        _mm256_storeu_si256((__m256i *)out, t2);

        nbytes -= 32u;
        in1 += 32u;
        out += 32u;
    }

    // sse code
    while(nbytes >= 16u) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)in1);
        const __m128i t2 = t1&s128;
        _mm_storeu_si128((__m128i *)out, t2);

        nbytes -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; nbytes > 0; --nbytes) {
        *out++ = *in1++ & m;
    }
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u256(gf2 *out,
                                              const gf2 c,
                                              const gf2 *in,
                                              const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in  + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2v_add_u256(a, gf2v_mul_u256(b, s)));
    }

    gf2_vector_scalar_add(out + i, c, in + i, n - 8*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u256_v2(gf2 *out,
                                                 const gf2 *in1,
                                                 const gf2 c,
                                                 const gf2 *in2,
                                                 const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf2v_add_u256(a, gf2v_mul_u256(b, s)));
    }

    gf2_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 8*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf2 gf2_vector_mul_acc_u256(const gf2 *a,
                                          const gf2 *b,
                                          const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = gf2v_add_u256(acc, gf2v_mul_u256(t1, t2));
    }

    return gf2_hadd_u256(acc) ^ gf2_vector_mul_acc(a + i, b + i, n - 8*i);
}

/// evaluates the polynomial in1[0] + in1[1]*x + ... + in1[d-1]*x^{d-1}
/// \return in1(scalar)
static inline gf2 gf2_vector_eval_u256(const gf2 *in1,
                                       const gf2 scalar,
                                       const uint32_t d) {
    if (d == 0) { return 0; }

    // x = 1: parity of all coefficients, x = 0: constant coefficient
    __m256i acc = _mm256_setzero_si256();
    const uint32_t nb = d / 8;
    uint32_t i = 0;
    for (; (i + 32) <= nb; i += 32) {
        acc ^= _mm256_loadu_si256((const __m256i *)(in1 + i));
    }

    const gf2 parity = gf2_hadd_u256(acc) ^ gf2_vector_eval(in1 + i, 1, d - 8*i);
    const gf2 v = scalar & 1u;
    return (v & parity) | ((v ^ 1u) & gf2_vector_get(in1, 0));
}

#endif

#ifdef USE_NEON

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u128(gf2 *out,
                                       const gf2 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in  + i);
        vst1q_u8(out + i, gf2v_add_u128(a, b));
    }

    gf2_vector_add(out + i, in + i, n - 8*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u128_v2(gf2 *out,
                                          const gf2 *in1,
                                          const gf2 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf2v_add_u128(a, b));
    }

    gf2_vector_add_v2(out + i, in1 + i, in2 + i, n - 8*i);
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u128(gf2 *out,
                                              const gf2 c,
                                              const gf2 *in,
                                              const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(out + i);
        const uint8x16_t b = vld1q_u8(in  + i);
        vst1q_u8(out + i, gf2v_add_u128(a, gf2v_mul_u128(b, s)));
    }

    gf2_vector_scalar_add(out + i, c, in + i, n - 8*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u128_v2(gf2 *out,
                                                 const gf2 *in1,
                                                 const gf2 c,
                                                 const gf2 *in2,
                                                 const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf2v_add_u128(a, gf2v_mul_u128(b, s)));
    }

    gf2_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 8*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf2 gf2_vector_mul_acc_u128(const gf2 *a,
                                          const gf2 *b,
                                          const uint32_t n) {
    uint8x16_t acc = vdupq_n_u8(0);
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 16) <= nb; i += 16) {
        const uint8x16_t t1 = vld1q_u8(a + i);
        const uint8x16_t t2 = vld1q_u8(b + i);
        acc = gf2v_add_u128(acc, gf2v_mul_u128(t1, t2));
    }

    return gf2_hadd_u128(acc) ^ gf2_vector_mul_acc(a + i, b + i, n - 8*i);
}

#endif

#ifdef USE_AVX512

/// out += in
/// \param out[in/out]
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u512(gf2 *out,
                                       const gf2 *in,
                                       const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(out + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in  + i));
        _mm512_storeu_si512((void *)(out + i), gf2v_add_u512(a, b));
    }

    gf2_vector_add(out + i, in + i, n - 8*i);
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_add_u512_v2(gf2 *out,
                                          const gf2 *in1,
                                          const gf2 *in2,
                                          const uint32_t n) {
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf2v_add_u512(a, b));
    }

    gf2_vector_add_v2(out + i, in1 + i, in2 + i, n - 8*i);
}

/// out += c*in
/// \param out[in/out]
/// \param c[in]: scalar
/// \param in[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u512(gf2 *out,
                                              const gf2 c,
                                              const gf2 *in,
                                              const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(out + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in  + i));
        _mm512_storeu_si512((void *)(out + i), gf2v_add_u512(a, gf2v_mul_u512(b, s)));
    }

    gf2_vector_scalar_add(out + i, c, in + i, n - 8*i);
}

/// out = in1 + c*in2
/// \param out[out]
/// \param in1[in]
/// \param c[in]: scalar
/// \param in2[in]
/// \param n[in]: number of elements
static inline void gf2_vector_scalar_add_u512_v2(gf2 *out,
                                                 const gf2 *in1,
                                                 const gf2 c,
                                                 const gf2 *in2,
                                                 const uint32_t n) {
    const __m512i s = _mm512_set1_epi8(-(c & 1u));
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf2v_add_u512(a, gf2v_mul_u512(b, s)));
    }

    gf2_vector_scalar_add_v2(out + i, in1 + i, c, in2 + i, n - 8*i);
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a[i]*b[i]
static inline gf2 gf2_vector_mul_acc_u512(const gf2 *a,
                                          const gf2 *b,
                                          const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    const uint32_t nb = n / 8;
    uint32_t i = 0;
    for (; (i + 64) <= nb; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc = gf2v_add_u512(acc, gf2v_mul_u512(t1, t2));
    }

    return gf2_hadd_u512(acc) ^ gf2_vector_mul_acc(a + i, b + i, n - 8*i);
}

#endif
