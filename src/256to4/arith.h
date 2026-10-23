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

/// \return -a = a (characteristic 2)
static inline gf256to4 gf256to4_neg(const gf256to4 a) {
    return a;
}

/// inversion via the norm map into GF(256^2): for a = a_1*X + a_0
///     N(a) = a_0*(a_0 + a_1) + (256*32)*a_1^2, a^-1 = (a_1*X + a_0 + a_1) * N(a)^-1
/// NOTE: inv(0) = 0
/// \return a^-1
static inline gf256to4 gf256to4_inv(const gf256to4 a) {
    const gf256to2 a0 = a, a1 = a >> 16u;
    const gf256to2 s = a0 ^ a1;
    const gf256to2 n = gf256to2_mul(a0, s) ^ gf256to2_mul(0x2000, gf256to2_mul(a1, a1));
    const gf256to2 ninv = gf256to2_inv(n);
    return ((gf256to4)gf256to2_mul(a1, ninv) << 16u) | gf256to2_mul(s, ninv);
}

#ifdef USE_AVX2
/// \return [a_0+b_0, ..., a_7+b_7]
static inline __m256i gf256to4v_add_u256(const __m256i a,
                                         const __m256i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_7-b_7]
static inline __m256i gf256to4v_sub_u256(const __m256i a,
                                         const __m256i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_7]
/// \param b [b_0, ..., b_7]
/// \return [a_0*b_0, ..., a_7*b_7]
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

/// SSE version of `gf256to4v_mul_u256`
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

#ifdef USE_NEON
/// \return [a_0+b_0, ..., a_3+b_3]
static inline uint32x4_t gf256to4v_add_u128(const uint32x4_t a,
                                            const uint32x4_t b) {
    return veorq_u32(a, b);
}

/// \return [a_0-b_0, ..., a_3-b_3]
static inline uint32x4_t gf256to4v_sub_u128(const uint32x4_t a,
                                            const uint32x4_t b) {
    return veorq_u32(a, b);
}

/// NEON version of `gf256to4v_mul_u256`
/// \param a [a_0, ..., a_3]
/// \param b [b_0, ..., b_3]
/// \return [a_0*b_0, ..., a_3*b_3]
static inline uint32x4_t gf256to4v_mul_u128(const uint32x4_t a,
                                            const uint32x4_t b) {
    const uint32x4_t mask1 = vdupq_n_u32(0x0000FFFF);
    const uint32x4_t mask2 = vdupq_n_u32(0xFFFF0000);
    const uint16x8_t mulm  = vreinterpretq_u16_u32(vdupq_n_u32(0x2000));
    const uint32x4_t ab = vreinterpretq_u32_u16(gf256to2v_mul_u128(vreinterpretq_u16_u32(a),
                                                                   vreinterpretq_u16_u32(b)));
    const uint32x4_t sa = veorq_u32(vshlq_n_u32(a, 16), a);
    const uint32x4_t sb = veorq_u32(vshlq_n_u32(b, 16), b);

    // compute the higher limb
    const uint32x4_t h = vreinterpretq_u32_u16(gf256to2v_mul_u128(vreinterpretq_u16_u32(sa),
                                                                  vreinterpretq_u16_u32(sb)));
    uint32x4_t c = vandq_u32(veorq_u32(h, vshlq_n_u32(ab, 16)), mask2);

    // compute the lower limb
    const uint32x4_t l = vreinterpretq_u32_u16(gf256to2v_mul_u128(vreinterpretq_u16_u32(vshrq_n_u32(ab, 16)), mulm));
    c = veorq_u32(c, vandq_u32(veorq_u32(ab, l), mask1));
    return c;
}
#endif // USE_NEON

#ifdef USE_AVX512
/// \return [a_0+b_0, ..., a_15+b_15]
static inline __m512i gf256to4v_add_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_15-b_15]
static inline __m512i gf256to4v_sub_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_15]
/// \param b [b_0, ..., b_15]
/// \return [a_0*b_0, ..., a_15*b_15]
static inline __m512i gf256to4v_mul_u512(const __m512i a,
                                         const __m512i b) {
    __m512i c;
    const __m512i mask1 = _mm512_set1_epi32(0x0000FFFF);
    const __m512i mask2 = _mm512_set1_epi32((int)0xFFFF0000);
    const __m512i mulm  = _mm512_set1_epi32(0x2000);
    const __m512i ab = gf256to2v_mul_u512(a, b);
    const __m512i sa = _mm512_slli_epi32(a, 16) ^ a;
    const __m512i sb = _mm512_slli_epi32(b, 16) ^ b;

    // compute the higher
    c = (gf256to2v_mul_u512(sa, sb) ^ _mm512_slli_epi32(ab, 16)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to2v_mul_u512(_mm512_srli_epi32(ab, 16), mulm)) & mask1;
    return c;
}
#endif // USE_AVX512

#endif //FQ_ARITHMETIC_ARITH_H
