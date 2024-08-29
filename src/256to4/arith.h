#ifndef FQ_ARITHMETIC_GF256TO4_ARITH_H
#define FQ_ARITHMETIC_GF256TO4_ARITH_H

#include <stdint.h>
#include "../256to2/arith.h"
#include "../256/arith.h"


//  GF256to4: uint32_t: X^2 + x + (256*32) = 0
typedef uint32_t gf256to4;


/// X^2 + X + (256*32) = 0
/// \param a
/// \param b
/// \return a*b
static inline gf256to4 gf256to4_mul(const gf256to4 a, const gf256to4 b) {
    gf256to4 upper, lower, res;
    upper = gf256to2_mul(a >> 16, b >> 16);
    lower = gf256to2_mul(a, b);

    // Compute lower
    res =  gf256_mul(gf256_mul(upper>>8, 0x20), 0x20);
    res ^= (gf256_mul(gf256_add(upper, upper>>8), 0x20) << 8u);
    res ^= lower;

    // Compute upper
    res ^= ((gf256to4)gf256to2_mul(gf256to2_add(a, a>>16), gf256to2_add(b, b>>16))) << 16;
    res ^= lower << 16;
    return res;
}

/// \param a
/// \param b
/// \return
static inline gf256to4 gf256to4_add(const gf256to4 a,
                                    const gf256to4 b) {
    return a ^ b;
}

/// \param a
/// \param b
/// \return
static inline gf256to4 gf256to4_sub(const gf256to4 a,
                                    const gf256to4 b) {
    return a ^ b;
}

#ifdef USE_AVX2
///
/// \param a
/// \param b
/// \return
static inline __m256i gf256to4v_mul_u256(const __m256i a,
                                         const __m256i b) {
    __m256i c;
    const __m256i mask1 = _mm256_set1_epi32(0x0000FFFF);
    const __m256i mask2 = _mm256_set1_epi32(0xFFFF0000);
    const __m256i mulm  = _mm256_set1_epi32(0x2000);
    const __m256i ab = gf256to2v_mul_u256(a, b);
    const __m256i sa = _mm256_slli_epi32(a, 16) ^ a;
    const __m256i sb = _mm256_slli_epi32(b, 16) ^ b;

    // compute the higher
    c = (gf256to2v_mul_u256(sa, sb) ^ _mm256_slli_epi32(ab, 16)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to2v_mul_u256(_mm256_srli_epi32(ab, 16), mulm)) & mask1;
    return c;
}

/// 3 gf(256) mul + 11 inst = 119 instr
/// \param a
/// \param b
/// \return
static inline __m128i gf256to4v_mul_u128(const __m128i a,
                                         const __m128i b) {
    __m128i c;
    const __m128i mask1 = _mm_set1_epi32(0x0000FFFF);
    const __m128i mask2 = _mm_set1_epi32(0xFFFF0000);
    const __m128i mulm  = _mm_set1_epi16(0x2000);
    const __m128i ab = gf256to2v_mul_u128(a, b);
    const __m128i sa = _mm_slli_epi32(a, 16) ^ a;
    const __m128i sb = _mm_slli_epi32(b, 16) ^ b;

    // compute the higher
    c = (gf256to2v_mul_u128(sa, sb) ^ _mm_slli_epi32(ab, 16)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to2v_mul_u128 (_mm_srli_epi32(ab, 16), mulm)) & mask1;
    return c;
}
#endif

#endif //FQ_ARITHMETIC_ARITH_H
