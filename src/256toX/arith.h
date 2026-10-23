#ifndef FQ_ARITHMETIC_GF256TOX_ARITH_H
#define FQ_ARITHMETIC_GF256TOX_ARITH_H

#include <stdint.h>
#include <string.h>
#include "../256/arith.h"
#include "../16/arith.h"

#define EXTENSION_DEGREE 48
#define NR_REDUCTION_ENTRIES 4

// describes the reduction polynomial:
// 	[0,1,2,5] with the coefficients [9,1,1,1], becomes:
//      X^(EXTENSION_DEGREE) + X^5 + X^2 + X + 9
// NOTE: a polynomial with only 0/1 coefficients (e.g. the former constant
//  term 1) is never irreducible over GF(256) for EXTENSION_DEGREE = 48, as it
//  splits into gcd(48, 8) factors.
static const uint8_t __gf256toX_red_pol[] __attribute__((aligned(32))) = {0,1,2,5};
static const uint8_t __gf256toX_red_coef[] __attribute__((aligned(32))) = {9,1,1,1};
static const uint8_t __gf256toX_red_pol_full[][EXTENSION_DEGREE] __attribute__((aligned(32))) = {
        {0,0,0,0,0,0,0,0,0,0,1}
};
typedef uint8_t  gf256toX[EXTENSION_DEGREE];


static inline void gf256toX_set(gf256toX res,
                                const gf256toX a) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i];
    }
}

static inline void gf256toX_zero(gf256toX res) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = 0;
    }
}

static inline void gf256toX_add(gf256toX res,
                                const gf256toX a,
                                const gf256toX b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i] ^ b[i];
    }
}

/// res = a - b
static inline void gf256toX_sub(gf256toX res,
                                const gf256toX a,
                                const gf256toX b) {
    gf256toX_add(res, a, b);
}

/// res = -a = a (characteristic 2)
static inline void gf256toX_neg(gf256toX res,
                                const gf256toX a) {
    gf256toX_set(res, a);
}

static inline void gf256toX_scalar_mul(gf256toX res,
                                       const gf256toX a,
                                       const gf256 b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = gf256_mul(a[i], b);
    }
}

static inline void gf256toX_add_scalar_mul(gf256toX res,
                                           const gf256toX a,
                                           const gf256 b) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] ^= gf256_mul(a[i], b);
    }
}

static inline void gf256toX_mul_gf256(gf256toX res,
                                      const gf256toX a,
                                      const gf256 b) {
    gf256toX_scalar_mul(res, a, b);
}

static inline void gf256toX_mul_gf16(gf256toX res,
                                     const gf256toX a,
                                     const gf16 b) {
    gf256toX_scalar_mul(res, a, gf256_expand_tab[b]);
}

static inline void gf256toX_mul(gf256toX res,
                                const gf256toX a,
                                const gf256toX b) {
    gf256 product[2*EXTENSION_DEGREE - 1u] = {0};
    gf256toX tmp;
    gf256toX_scalar_mul(product, a, b[0]);

    // mul
    for(uint32_t i = 1 ; i < EXTENSION_DEGREE ; i++) {
        gf256toX_scalar_mul(tmp, a, b[i]);
        gf256toX_add(product + i, product + i, tmp);
    }

    // Red
    for(uint32_t i = 2*EXTENSION_DEGREE - 2; i>=EXTENSION_DEGREE ; i--) {
        for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
            product[i - (EXTENSION_DEGREE - __gf256toX_red_pol[j])] ^= gf256_mul(product[i], __gf256toX_red_coef[j]);
        }
    }

    // copy back
    for(uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        res[i] = product[i];
    }
}

/// Itoh-Tsujii inversion over GF(256):
///     u = a^(256 + 256^2 + ... + 256^(EXTENSION_DEGREE-1)),
///     N(a) = u*a \in GF(256), a^-1 = u * N(a)^-1
/// NOTE: inv(0) = 0
/// \param res = a^-1
static inline void gf256toX_inv(gf256toX res,
                                const gf256toX a) {
    gf256toX t, u, n;
    gf256toX_set(t, a);
    gf256toX_zero(u);
    u[0] = 1;
    for (uint32_t i = 1; i < EXTENSION_DEGREE; i++) {
        // t = t^256
        for (uint32_t j = 0; j < 8; j++) {
            gf256toX_mul(t, t, t);
        }
        gf256toX_mul(u, u, t);
    }

    gf256toX_mul(n, u, a);
    gf256toX_scalar_mul(res, u, gf256_inv(n[0]));
}

/// \return the largest exponent (< EXTENSION_DEGREE) of the reduction polynomial
static inline uint32_t gf256toX_red_pol_max(void) {
    uint32_t r = 0;
    for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
        r = __gf256toX_red_pol[j] > r ? __gf256toX_red_pol[j] : r;
    }
    return r;
}

/// \return the number of coefficients >= EXTENSION_DEGREE which remain after
///         reducing `len` such coefficients once
static inline uint32_t gf256toX_red_next_len(const uint32_t len) {
    const uint32_t m = gf256toX_red_pol_max();
    return (len + m > EXTENSION_DEGREE) ? (len + m - EXTENSION_DEGREE) : 0;
}

// NOTE: the SIMD functions of this field operate on a single element given by
// a pointer, as `EXTENSION_DEGREE` is a compile time parameter and an element
// does not fit a fixed register type. The suffix names the instruction set used.

#ifdef USE_AVX2
/// EXTENSION_DEGREE rounded up to a multiple of 32
#define GF256TOX_PAD_U256 (((EXTENSION_DEGREE + 31u) / 32u) * 32u)

/// res = a + b
static inline void gf256toXv_add_u256(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint32_t i = 0;
    for (; (i + 32) <= EXTENSION_DEGREE; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        _mm256_storeu_si256((__m256i *)(res + i), t1 ^ t2);
    }
    for (; (i + 16) <= EXTENSION_DEGREE; i += 16) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(a + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(b + i));
        _mm_storeu_si128((__m128i *)(res + i), t1 ^ t2);
    }
    for (; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i] ^ b[i];
    }
}

/// res = a - b
static inline void gf256toXv_sub_u256(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    gf256toXv_add_u256(res, a, b);
}

/// res = a*b: schoolbook multiplication, followed by a vectorized reduction
static inline void gf256toXv_mul_u256(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint8_t pa[GF256TOX_PAD_U256] __attribute__((aligned(32))) = {0};
    uint8_t prod[2*GF256TOX_PAD_U256 + 32] __attribute__((aligned(32))) = {0};
    uint8_t h[GF256TOX_PAD_U256] __attribute__((aligned(32)));
    memcpy(pa, a, EXTENSION_DEGREE);

    // prod[i + j] += a[i]*b[j]
    for (uint32_t j = 0; j < EXTENSION_DEGREE; j++) {
        const __m256i bj = _mm256_set1_epi8((char)b[j]);
        for (uint32_t k = 0; k < GF256TOX_PAD_U256; k += 32) {
            const __m256i t = gf256v_mul_u256(_mm256_load_si256((const __m256i *)(pa + k)), bj);
            const __m256i p = _mm256_loadu_si256((const __m256i *)(prod + j + k));
            _mm256_storeu_si256((__m256i *)(prod + j + k), p ^ t);
        }
    }

    // X^EXTENSION_DEGREE = sum_j coef[j] * X^pol[j]
    uint32_t len = EXTENSION_DEGREE - 1;
    while (len) {
        memset(h, 0, GF256TOX_PAD_U256);
        memcpy(h, prod + EXTENSION_DEGREE, len);
        memset(prod + EXTENSION_DEGREE, 0, len);
        for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
            const __m256i c = _mm256_set1_epi8((char)__gf256toX_red_coef[j]);
            uint8_t *p = prod + __gf256toX_red_pol[j];
            for (uint32_t k = 0; k < GF256TOX_PAD_U256; k += 32) {
                const __m256i t = gf256v_mul_u256(_mm256_load_si256((const __m256i *)(h + k)), c);
                _mm256_storeu_si256((__m256i *)(p + k), _mm256_loadu_si256((const __m256i *)(p + k)) ^ t);
            }
        }
        len = gf256toX_red_next_len(len);
    }

    memcpy(res, prod, EXTENSION_DEGREE);
}
#endif // USE_AVX2

#ifdef USE_NEON
/// EXTENSION_DEGREE rounded up to a multiple of 16
#define GF256TOX_PAD_U128 (((EXTENSION_DEGREE + 15u) / 16u) * 16u)

/// res = a + b
static inline void gf256toXv_add_u128(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint32_t i = 0;
    for (; (i + 16) <= EXTENSION_DEGREE; i += 16) {
        vst1q_u8(res + i, veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i)));
    }
    for (; i < EXTENSION_DEGREE; i++) {
        res[i] = a[i] ^ b[i];
    }
}

/// res = a - b
static inline void gf256toXv_sub_u128(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    gf256toXv_add_u128(res, a, b);
}

/// res = a*b: schoolbook multiplication, followed by a vectorized reduction
static inline void gf256toXv_mul_u128(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint8_t pa[GF256TOX_PAD_U128] = {0};
    uint8_t prod[2*GF256TOX_PAD_U128 + 16] = {0};
    uint8_t h[GF256TOX_PAD_U128];
    memcpy(pa, a, EXTENSION_DEGREE);

    // prod[i + j] += a[i]*b[j]
    for (uint32_t j = 0; j < EXTENSION_DEGREE; j++) {
        const uint8x16_t bj = vdupq_n_u8(b[j]);
        for (uint32_t k = 0; k < GF256TOX_PAD_U128; k += 16) {
            const uint8x16_t t = gf256v_mul_u128(vld1q_u8(pa + k), bj);
            vst1q_u8(prod + j + k, veorq_u8(vld1q_u8(prod + j + k), t));
        }
    }

    // X^EXTENSION_DEGREE = sum_j coef[j] * X^pol[j]
    uint32_t len = EXTENSION_DEGREE - 1;
    while (len) {
        memset(h, 0, GF256TOX_PAD_U128);
        memcpy(h, prod + EXTENSION_DEGREE, len);
        memset(prod + EXTENSION_DEGREE, 0, len);
        for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
            const uint8x16_t c = vdupq_n_u8(__gf256toX_red_coef[j]);
            uint8_t *p = prod + __gf256toX_red_pol[j];
            for (uint32_t k = 0; k < GF256TOX_PAD_U128; k += 16) {
                const uint8x16_t t = gf256v_mul_u128(vld1q_u8(h + k), c);
                vst1q_u8(p + k, veorq_u8(vld1q_u8(p + k), t));
            }
        }
        len = gf256toX_red_next_len(len);
    }

    memcpy(res, prod, EXTENSION_DEGREE);
}
#endif // USE_NEON

#ifdef USE_AVX512
/// EXTENSION_DEGREE rounded up to a multiple of 64
#define GF256TOX_PAD_U512 (((EXTENSION_DEGREE + 63u) / 64u) * 64u)

/// res = a + b
static inline void gf256toXv_add_u512(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint32_t i = 0;
    for (; (i + 64) <= EXTENSION_DEGREE; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        _mm512_storeu_si512((void *)(res + i), t1 ^ t2);
    }
    if (i < EXTENSION_DEGREE) {
        const __mmask64 k = (~0ull) >> (64u - (EXTENSION_DEGREE - i));
        const __m512i t1 = _mm512_maskz_loadu_epi8(k, (const void *)(a + i));
        const __m512i t2 = _mm512_maskz_loadu_epi8(k, (const void *)(b + i));
        _mm512_mask_storeu_epi8((void *)(res + i), k, t1 ^ t2);
    }
}

/// res = a - b
static inline void gf256toXv_sub_u512(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    gf256toXv_add_u512(res, a, b);
}

/// res = a*b: schoolbook multiplication, followed by a vectorized reduction
static inline void gf256toXv_mul_u512(gf256toX res,
                                      const gf256toX a,
                                      const gf256toX b) {
    uint8_t pa[GF256TOX_PAD_U512] __attribute__((aligned(64))) = {0};
    uint8_t prod[2*GF256TOX_PAD_U512 + 64] __attribute__((aligned(64))) = {0};
    uint8_t h[GF256TOX_PAD_U512] __attribute__((aligned(64)));
    memcpy(pa, a, EXTENSION_DEGREE);

    // prod[i + j] += a[i]*b[j]
    for (uint32_t j = 0; j < EXTENSION_DEGREE; j++) {
        const __m512i bj = _mm512_set1_epi8((char)b[j]);
        for (uint32_t k = 0; k < GF256TOX_PAD_U512; k += 64) {
            const __m512i t = gf256v_mul_u512(_mm512_load_si512((const void *)(pa + k)), bj);
            const __m512i p = _mm512_loadu_si512((const void *)(prod + j + k));
            _mm512_storeu_si512((void *)(prod + j + k), p ^ t);
        }
    }

    // X^EXTENSION_DEGREE = sum_j coef[j] * X^pol[j]
    uint32_t len = EXTENSION_DEGREE - 1;
    while (len) {
        memset(h, 0, GF256TOX_PAD_U512);
        memcpy(h, prod + EXTENSION_DEGREE, len);
        memset(prod + EXTENSION_DEGREE, 0, len);
        for (uint32_t j = 0; j < NR_REDUCTION_ENTRIES; j++) {
            const __m512i c = _mm512_set1_epi8((char)__gf256toX_red_coef[j]);
            uint8_t *p = prod + __gf256toX_red_pol[j];
            for (uint32_t k = 0; k < GF256TOX_PAD_U512; k += 64) {
                const __m512i t = gf256v_mul_u512(_mm512_load_si512((const void *)(h + k)), c);
                _mm512_storeu_si512((void *)(p + k), _mm512_loadu_si512((const void *)(p + k)) ^ t);
            }
        }
        len = gf256toX_red_next_len(len);
    }

    memcpy(res, prod, EXTENSION_DEGREE);
}
#endif // USE_AVX512

#endif
