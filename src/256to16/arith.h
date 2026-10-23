#ifndef FQ_ARITHMETIC_GF256TO16_ARITH_H
#define FQ_ARITHMETIC_GF256TO16_ARITH_H

#include <stdint.h>
#include <string.h>

#include "../256to8/arith.h"
#include "../16/arith.h"

/// GF256to16:uint8_t[16]: X^2 + x + (4294967296 * 65536 * 256 * 32)
typedef uint8_t  gf256to16[16];

// NOTE: elements are accessed via `memcpy` and not via casts to `uint64_t *`
// or `__uint128_t *`, because those break strict aliasing (and alignment).
static inline void gf256to16_zero(gf256to16 res) {
    memset(res, 0, 16);
}

static inline void gf256to16_set(gf256to16 res,
                                 const uint8_t *b) {
    memcpy(res, b, 16);
}

static inline void gf256to16_set_(gf256to16 *res,
                                  const uint8_t *b) {
    memcpy(*res, b, 16);
}

/// X^2 + X + (4294967296*4096*256*32) = 0
/// \param res
/// \param a
/// \param b
static inline void gf256to16_mul(gf256to16 res,
                                 const gf256to16 a,
                                 const gf256to16 b) {
    gf256to8 a_[2], b_[2], r_[2];
    memcpy(a_, a, 16);
    memcpy(b_, b, 16);
    const gf256to8 upper = gf256to8_mul(a_[1], b_[1]);
    const gf256to8 lower = gf256to8_mul(a_[0], b_[0]);

    // Compute lower
    const gf256to8 factor = 0x2000000000000000;
    r_[0] = gf256to8_add(gf256to8_mul(upper, factor), lower);

    // Compute upper
    r_[1] = gf256to8_add(gf256to8_mul(gf256to8_add(a_[0], a_[1]), gf256to8_add(b_[0], b_[1])), lower);
    memcpy(res, r_, 16);
}

///
/// \param res
/// \param a
/// \param b
static inline void gf256to16_add(gf256to16 res,
                                 const gf256to16 a,
                                 const gf256to16 b) {
    for (uint32_t i = 0; i < 16; i++){
        res[i] = a[i] ^ b[i];
    }
}

/// res = a - b
static inline void gf256to16_sub(gf256to16 res,
                                 const gf256to16 a,
                                 const gf256to16 b) {
    gf256to16_add(res, a, b);
}

/// res = -a = a (characteristic 2)
static inline void gf256to16_neg(gf256to16 res,
                                 const gf256to16 a) {
    memmove(res, a, 16);
}

/// inversion via the norm map into GF(256^8): for a = a_1*X + a_0
///     N(a) = a_0*(a_0 + a_1) + c*a_1^2, a^-1 = (a_1*X + a_0 + a_1) * N(a)^-1
/// with c = 0x2000000000000000.
/// NOTE: inv(0) = 0
/// \param res = a^-1
static inline void gf256to16_inv(gf256to16 res,
                                 const gf256to16 a) {
    gf256to8 a_[2], r_[2];
    memcpy(a_, a, 16);
    const gf256to8 s = gf256to8_add(a_[0], a_[1]);
    const gf256to8 n = gf256to8_mul(a_[0], s) ^
                       gf256to8_mul(0x2000000000000000ull, gf256to8_mul(a_[1], a_[1]));
    const gf256to8 ninv = gf256to8_inv(n);
    r_[0] = gf256to8_mul(s, ninv);
    r_[1] = gf256to8_mul(a_[1], ninv);
    memcpy(res, r_, 16);
}

///
/// \param res
/// \param a
/// \param b
static inline void gf256to16_mul_gf256(gf256to16 res,
                                       const gf256to16 a,
                                       const gf256 b) {
    for (uint32_t i = 0; i < 16; ++i) {
        res[i] = gf256_mul(a[i], b);
    }
}

///
/// \param res
/// \param a
/// \param b
static inline void gf256to16_mul_gf16(gf256to16 res,
                                      const gf256to16 a,
                                      const gf16 b) {
    const gf256 t = gf256_expand_tab[b];
    gf256to16_mul_gf256(res, a, t);
}

#ifdef USE_AVX2

/// \return [a_0+b_0, a_1+b_1], 2 elements per register
static inline __m256i gf256to16v_add_u256(const __m256i a,
                                          const __m256i b) {
    return a ^ b;
}

/// \return [a_0-b_0, a_1-b_1], 2 elements per register
static inline __m256i gf256to16v_sub_u256(const __m256i a,
                                          const __m256i b) {
    return a ^ b;
}

/// \param a [a_0, a_1]
/// \param b [b_0, b_1]
/// \return [a_0*b_0, a_1*b_1]
static inline __m256i gf256to16v_mul_u256(const __m256i a,
                                          const __m256i b) {
    __m256i c;
    const __m256i mask1 = _mm256_set_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xFFFFFFFFFFFFFFFF);
    const __m256i mask2 = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000);
    const __m256i mulm  = _mm256_set1_epi64x(0x2000000000000000);
    const __m256i ab = gf256to8v_mul_u256(a, b);
    const __m256i sa = (_mm256_bslli_epi128(a, 8)) ^ a;
    const __m256i sb = (_mm256_bslli_epi128(b, 8)) ^ b;

    // compute the higher
    c = (gf256to8v_mul_u256(sa, sb) ^ (_mm256_bslli_epi128(ab, 8))) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to8v_mul_u256(_mm256_bsrli_epi128(ab, 8), mulm)) & mask1;
    return c;
}

/// SSE version of `gf256to16v_mul_u256`: a single element
/// \param a
/// \param b
/// \return a*b
static inline __m128i gf256to16v_mul_u128(const __m128i a,
                                          const __m128i b) {
    __m128i c;
    const __m128i mask1 = _mm_set_epi32(0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF);
    const __m128i mask2 = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000);
    const __m128i mulm  = _mm_set1_epi64x(0x2000000000000000);
    const __m128i ab = gf256to8v_mul_u128(a, b);
    const __m128i sa = _mm_slli_si128(a, 8) ^ a;
    const __m128i sb = _mm_slli_si128(b, 8) ^ b;

    // compute the higher
    c = (gf256to8v_mul_u128(sa, sb) ^ _mm_slli_si128(ab, 8)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to8v_mul_u128(_mm_srli_si128(ab, 8), mulm)) & mask1;
    return c;
}
#endif // USE_AVX2

#ifdef USE_NEON
/// a single element per register: [lower limb, upper limb]
/// \return a+b
static inline uint64x2_t gf256to16v_add_u128(const uint64x2_t a,
                                             const uint64x2_t b) {
    return veorq_u64(a, b);
}

/// \return a-b
static inline uint64x2_t gf256to16v_sub_u128(const uint64x2_t a,
                                             const uint64x2_t b) {
    return veorq_u64(a, b);
}

/// NEON version of `gf256to16_mul`: a single element per register
/// \return a*b
static inline uint64x2_t gf256to16v_mul_u128(const uint64x2_t a,
                                             const uint64x2_t b) {
    const uint64x1_t z = vdup_n_u64(0);
    // [a_0*b_0, a_1*b_1]
    const uint64x2_t ab = gf256to8v_mul_u128(a, b);
    // [a_0, a_0 + a_1]
    const uint64x2_t sa = veorq_u64(a, vcombine_u64(z, vget_low_u64(a)));
    const uint64x2_t sb = veorq_u64(b, vcombine_u64(z, vget_low_u64(b)));
    // [*, (a_0 + a_1)*(b_0 + b_1)]
    const uint64x2_t h = gf256to8v_mul_u128(sa, sb);
    // [c*a_1*b_1, 0]
    const uint64x2_t t = gf256to8v_mul_u128(vcombine_u64(vget_high_u64(ab), z),
                                            vdupq_n_u64(0x2000000000000000ull));

    const uint64x1_t r0 = veor_u64(vget_low_u64(ab), vget_low_u64(t));
    const uint64x1_t r1 = veor_u64(vget_high_u64(h), vget_low_u64(ab));
    return vcombine_u64(r0, r1);
}
#endif // USE_NEON

#ifdef USE_AVX512
/// \return [a_0+b_0, ..., a_3+b_3], 4 elements per register
static inline __m512i gf256to16v_add_u512(const __m512i a,
                                          const __m512i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_3-b_3], 4 elements per register
static inline __m512i gf256to16v_sub_u512(const __m512i a,
                                          const __m512i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_3]
/// \param b [b_0, ..., b_3]
/// \return [a_0*b_0, ..., a_3*b_3]
static inline __m512i gf256to16v_mul_u512(const __m512i a,
                                          const __m512i b) {
    __m512i c;
    const __m512i mask1 = _mm512_set_epi64(0, -1, 0, -1, 0, -1, 0, -1);
    const __m512i mask2 = _mm512_set_epi64(-1, 0, -1, 0, -1, 0, -1, 0);
    const __m512i mulm  = _mm512_set1_epi64(0x2000000000000000ll);
    const __m512i ab = gf256to8v_mul_u512(a, b);
    const __m512i sa = _mm512_bslli_epi128(a, 8) ^ a;
    const __m512i sb = _mm512_bslli_epi128(b, 8) ^ b;

    // compute the higher
    c = (gf256to8v_mul_u512(sa, sb) ^ _mm512_bslli_epi128(ab, 8)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256to8v_mul_u512(_mm512_bsrli_epi128(ab, 8), mulm)) & mask1;
    return c;
}
#endif // USE_AVX512

#endif //FQ_ARITHMETIC_ARITH_H
