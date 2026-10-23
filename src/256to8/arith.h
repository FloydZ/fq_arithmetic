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

/// \return -a = a (characteristic 2)
static inline gf256to8 gf256to8_neg(const gf256to8 a) {
    return a;
}

/// inversion via the norm map into GF(256^4): for a = a_1*X + a_0
///     N(a) = a_0*(a_0 + a_1) + (65536*256*32)*a_1^2, a^-1 = (a_1*X + a_0 + a_1) * N(a)^-1
/// NOTE: inv(0) = 0
/// \return a^-1
static inline gf256to8 gf256to8_inv(const gf256to8 a) {
    const gf256to4 a0 = (gf256to4)a, a1 = (gf256to4)(a >> 32u);
    const gf256to4 s = a0 ^ a1;
    const gf256to4 n = gf256to4_mul(a0, s) ^ gf256to4_mul(0x20000000, gf256to4_mul(a1, a1));
    const gf256to4 ninv = gf256to4_inv(n);
    return ((gf256to8)gf256to4_mul(a1, ninv) << 32u) | gf256to4_mul(s, ninv);
}

#ifdef USE_AVX2
#include <immintrin.h>

/// \return [a_0+b_0, ..., a_3+b_3]
static inline __m256i gf256to8v_add_u256(const __m256i a,
                                         const __m256i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_3-b_3]
static inline __m256i gf256to8v_sub_u256(const __m256i a,
                                         const __m256i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_3]
/// \param b [b_0, ..., b_3]
/// \return [a_0*b_0, ..., a_3*b_3]
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

/// SSE version of `gf256to8v_mul_u256`
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

#ifdef USE_NEON
/// \return [a_0+b_0, a_1+b_1]
static inline uint64x2_t gf256to8v_add_u128(const uint64x2_t a,
                                            const uint64x2_t b) {
    return veorq_u64(a, b);
}

/// \return [a_0-b_0, a_1-b_1]
static inline uint64x2_t gf256to8v_sub_u128(const uint64x2_t a,
                                            const uint64x2_t b) {
    return veorq_u64(a, b);
}

/// NEON version of `gf256to8v_mul_u256`
/// \param a [a_0, a_1]
/// \param b [b_0, b_1]
/// \return [a_0*b_0, a_1*b_1]
static inline uint64x2_t gf256to8v_mul_u128(const uint64x2_t a,
                                            const uint64x2_t b) {
    const uint64x2_t mask1 = vdupq_n_u64(0x00000000FFFFFFFFull);
    const uint64x2_t mask2 = vdupq_n_u64(0xFFFFFFFF00000000ull);
    const uint32x4_t mulm  = vdupq_n_u32(0x20000000);
    const uint64x2_t ab = vreinterpretq_u64_u32(gf256to4v_mul_u128(vreinterpretq_u32_u64(a),
                                                                   vreinterpretq_u32_u64(b)));
    const uint64x2_t sa = veorq_u64(vshlq_n_u64(a, 32), a);
    const uint64x2_t sb = veorq_u64(vshlq_n_u64(b, 32), b);

    // compute the higher limb
    const uint64x2_t h = vreinterpretq_u64_u32(gf256to4v_mul_u128(vreinterpretq_u32_u64(sa),
                                                                  vreinterpretq_u32_u64(sb)));
    uint64x2_t c = vandq_u64(veorq_u64(h, vshlq_n_u64(ab, 32)), mask2);

    // compute the lower limb
    const uint64x2_t l = vreinterpretq_u64_u32(gf256to4v_mul_u128(vreinterpretq_u32_u64(vshrq_n_u64(ab, 32)), mulm));
    c = veorq_u64(c, vandq_u64(veorq_u64(ab, l), mask1));
    return c;
}
#endif // USE_NEON

#ifdef USE_AVX512
/// \return [a_0+b_0, ..., a_7+b_7]
static inline __m512i gf256to8v_add_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_7-b_7]
static inline __m512i gf256to8v_sub_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_7]
/// \param b [b_0, ..., b_7]
/// \return [a_0*b_0, ..., a_7*b_7]
static inline __m512i gf256to8v_mul_u512(const __m512i a,
                                         const __m512i b) {
    __m512i c;
    const __m512i mask1 = _mm512_set1_epi64(0x00000000FFFFFFFFll);
    const __m512i mask2 = _mm512_set1_epi64((long long)0xFFFFFFFF00000000ull);
    const __m512i mulm  = _mm512_set1_epi32(0x20000000);
    const __m512i ab = gf256to4v_mul_u512(a, b);
    const __m512i sa = _mm512_slli_epi64(a, 32u) ^ a;
    const __m512i sb = _mm512_slli_epi64(b, 32u) ^ b;

    // compute the higher
    c = (gf256to4v_mul_u512(sa, sb) ^ _mm512_slli_epi64(ab, 32u)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to4v_mul_u512(_mm512_srli_epi64(ab, 32u), mulm)) & mask1;
    return c;
}
#endif // USE_AVX512

#endif
