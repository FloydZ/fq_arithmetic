#ifndef FQ_ARITHMETIC_GF256TO8_ARITH_H
#define FQ_ARITHMETIC_GF256TO8_ARITH_H

#include <stdint.h>
#include "../256to4/arith.h"

typedef uint64_t gf256to8;

/// X^2 + X + (65536*256*32) = 0
/// \param a
/// \param b
/// \return
static inline gf256to8 gf256to8_mul(const gf256to8 a,
                                    const gf256to8 b) {
    gf256to8 res;
    const gf256to8 upper = gf256to4_mul(a>>32, b>>32);
    const gf256to8 lower = gf256to4_mul(a, b);

    // Compute lower
    const uint32_t factor = 0x20000000;
    res = gf256to4_mul(upper, factor);
    res ^= lower;

    // Compute upper
    const gf256to4 suma = gf256to4_add(a, a>>32);
    const gf256to4 sumb = gf256to4_add(b, b>>32);
    res ^= ((gf256to8)gf256to4_mul(suma, sumb))<<32;
    res ^= lower << 32;
    return res;
}

/// \param c
/// \param a
/// \param b
/// \return c += a*b
static inline gf256to8 gf256to8_addmul(gf256to8 c,
                                       const gf256to8 a,
                                       const gf256to8 b) {
    gf256to8 upper, lower;
    upper = gf256to4_mul(a>>32, b>>32);
    lower = gf256to4_mul(a, b);

    // Compute lower
    c ^= gf256to4_mul(upper, 0x20000000);
    c ^= lower;

    // Compute upper
    c ^= ((gf256to8)gf256to4_mul(gf256to4_add(a, a>>32u), gf256to4_add(b, b>>32u)))<<32u;
    c ^= lower << 32u;
    return c;

}

static inline gf256to8 gf256to8_add(const gf256to8 a,
                                    const gf256to8 b) {
    return a ^ b;
}

static inline gf256to8 gf256to8_sub(const gf256to8 a,
                                    const gf256to8 b) {
    return a ^ b;
}

#ifdef USE_AVX2
#include <immintrin.h>

/// \param a
/// \param b
/// \return
static inline __m256i gf256to8v_mul_u256(const __m256i a,
                                         const __m256i b) {
    __m256i c;
    const __m256i mask1 = _mm256_set1_epi64x(0x00000000FFFFFFFF);
    const __m256i mask2 = _mm256_set1_epi64x(0xFFFFFFFF00000000);
    const __m256i mulm  = _mm256_set1_epi32 (0x20000000);
    const __m256i ab = gf256to4v_mul_u256(a, b);
    const __m256i sa = _mm256_slli_epi64(a, 32u) ^ a;
    const __m256i sb = _mm256_slli_epi64(b, 32u) ^ b;

    // compute the higher
    c = (gf256to4v_mul_u256(sa, sb) ^ _mm256_slli_epi64(ab, 32u)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to4v_mul_u256(_mm256_srli_epi64(ab, 32u), mulm)) & mask1;
    return c;
}

/// 3 gf(256) mul + 11 inst = 119 instr
/// \param a
/// \param b
/// \return
static inline __m128i gf256to8v_mul_u128(const __m128i a,
                                         const __m128i b) {
    __m128i c;
    const __m128i mask1 = _mm_set1_epi64x(0x00000000FFFFFFFF);
    const __m128i mask2 = _mm_set1_epi64x(0xFFFFFFFF00000000);
    const __m128i mulm  = _mm_set1_epi32 (0x20000000);
    const __m128i ab = gf256to4v_mul_u128(a, b);
    const __m128i sa = _mm_slli_epi64(a, 32u) ^ a;
    const __m128i sb = _mm_slli_epi64(b, 32u) ^ b;

    // compute the higher
    c = (gf256to4v_mul_u128(sa, sb) ^ _mm_slli_epi64(ab, 32u)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to4v_mul_u128 (_mm_srli_epi64(ab, 32u), mulm)) & mask1;
    return c;
}
#endif // USE_AVX2

#endif
