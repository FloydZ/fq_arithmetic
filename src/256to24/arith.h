#ifndef FQ_ARITHMETIC_GF256TO24_ARITH_H
#define FQ_ARITHMETIC_GF256TO24_ARITH_H
#include <stdint.h>
#include <string.h>
#include "../256to16/arith.h"
#include "../256/arith.h"

/// in total we use the following representation for the field extension tower
///     GF256to24:uint8_t[24]: X^3 + x + 1;
typedef uint8_t  gf256to24[24];

static inline void gf256to24_zero(gf256to24 res) {
#ifdef __SIZEOF_INT128__
    for (uint32_t i = 0; i < 24; i++){
        res[i] = 0;
    }
#else
    for (uint32_t i = 0; i < 24; i++){
        res[i] = 0;
    }
#endif
}

static inline void gf256to24_set(gf256to24 res,
                                 const uint8_t *b) {
#ifdef __SIZEOF_INT128__
    // *(__uint128_t *)res = (*(__uint128_t *)b);
    for (uint32_t i = 0; i < 24; i++){
        res[i] = b[i];
    }
#else
    for (uint32_t i = 0; i < 24; i++){
        res[i] = b[i];
    }
#endif
}

static inline void gf256to24_set_(gf256to24 *res,
                                  const uint8_t *b) {
#ifdef __SIZEOF_INT128__
    // *(__uint128_t *)(*res) = (*(__uint128_t *)b);
    for (uint32_t i = 0; i < 24; i++){
        (*res)[i] = b[i];
    }
#else
    for (uint32_t i = 0; i < 24; i++){
        (*res)[i] = b[i];
    }
#endif
}

/// multiplication using karatsuba, reduced by X^3 + X + 1:
///     c_0 = p0, c_1 = p01 + p0 + p1, c_2 = p012 + p01 + p12,
///     c_3 = p12 + p1 + p2, c_4 = p2
///     r_0 = c_0 + c_3       = p0 + p1 + p2 + p12
///     r_1 = c_1 + c_3 + c_4 = p0 + p01 + p12
///     r_2 = c_2 + c_4       = p012 + p01 + p12 + p2
/// \param res
/// \param a
/// \param b
/// \reutrn
static inline void gf256to24_mul(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
    // NOTE: memcpy instead of casts to `uint64_t *`, which break strict aliasing
    uint64_t a_[3], b_[3], r_[3];
    memcpy(a_, a, 24);
    memcpy(b_, b, 24);

    const uint64_t p0 = gf256to8_mul(a_[0], b_[0]);
    const uint64_t p1 = gf256to8_mul(a_[1], b_[1]);
    const uint64_t p2 = gf256to8_mul(a_[2], b_[2]);

    const uint64_t a01 = gf256to8_add(a_[0], a_[1]);
    const uint64_t a12 = gf256to8_add(a_[1], a_[2]);
    const uint64_t b01 = gf256to8_add(b_[0], b_[1]);
    const uint64_t b12 = gf256to8_add(b_[1], b_[2]);
    const uint64_t p01 = gf256to8_mul(a01, b01);
    const uint64_t p12 = gf256to8_mul(a12, b12);

    const uint64_t a012= gf256to8_add(a01, a_[2]);
    const uint64_t b012= gf256to8_add(b01, b_[2]);
    const uint64_t p012= gf256to8_mul(a012, b012);

    r_[0] = gf256to8_add(p1, p2);
    r_[0] = gf256to8_add(r_[0], p12);
    r_[0] = gf256to8_add(r_[0], p0);

    r_[1] = gf256to8_add(p0, p01);
    r_[1] = gf256to8_add(r_[1], p12);

    r_[2] = gf256to8_add(p012, p01);
    r_[2] = gf256to8_add(r_[2], p12);
    r_[2] = gf256to8_add(r_[2], p2);
    memcpy(res, r_, 24);
}

#ifdef USE_AVX2
/// same as `gf256to24_mul`, but the 6 products are computed in 2 vectorized
/// multiplications.
/// NOTE: the 4th limb of a and b is ignored, the 4th limb of the output is zero
/// \param a: [a_0, a_1, a_2, *]
/// \param b: [b_0, b_1, b_2, *]
/// \return a*b
static inline __m256i gf256to24v_mul(const __m256i a,
                                     const __m256i b) {
    const __m256i zero = _mm256_setzero_si256();

    // q = [p0, p1, p2, *]
    const __m256i q = gf256to8v_mul_u256(a, b);

    // [a0^a1, a1^a2, a0^a1^a2, 0]
    const __m256i a1 = _mm256_blend_epi32(zero, _mm256_permute4x64_epi64(a, 0x10), 0x30);
    const __m256i b1 = _mm256_blend_epi32(zero, _mm256_permute4x64_epi64(b, 0x10), 0x30);
    const __m256i x = a ^ _mm256_permute4x64_epi64(a, 0xC9) ^ a1;
    const __m256i y = b ^ _mm256_permute4x64_epi64(b, 0xC9) ^ b1;

    // w = [p01, p12, p012, *]
    const __m256i w = gf256to8v_mul_u256(x, y);

    // s = p0 + p1 + p2 in every limb
    const __m256i s = _mm256_permute4x64_epi64(q, 0x00) ^
                      _mm256_permute4x64_epi64(q, 0x55) ^
                      _mm256_permute4x64_epi64(q, 0xAA);

    const __m256i w1 = _mm256_permute4x64_epi64(w, 0xC1); // [p12, p01, p01, *]
    const __m256i w2 = _mm256_permute4x64_epi64(w, 0xD4); // [*, p12, p12, *]
    const __m256i q1 = _mm256_permute4x64_epi64(q, 0xE0); // [*, p0, p2, *]

    const __m256i r0 = s ^ w1;              // [r0, *, *, *]
    const __m256i r1 = w1 ^ w2 ^ q1;        // [*, r1, p2 + p01 + p12, *]
    const __m256i r2 = r1 ^ w;              // [*, *, r2, *]

    __m256i r = _mm256_blend_epi32(r0, r1, 0x0C);
    r = _mm256_blend_epi32(r, r2, 0x30);
    return _mm256_blend_epi32(r, zero, 0xC0);
}
#endif

/// \param res
/// \param a
/// \param b
static inline void gf256to24_add(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
#ifdef __SIZEOF_INT128__
    for (uint32_t i = 0; i < 24; i++){
        res[i] = a[i] ^ b[i];
    }
#else
    for (uint32_t i = 0; i < 24; i++){
        res[i] = a[i] ^ b[i];
    }
#endif
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf256to2(gf256to24 res,
                                          const gf256to24 a,
                                          const gf256to2 b) {
    gf256to2 *res_ = (gf256to2 *)res;
    gf256to2 *a_   = (gf256to2 *)a;
    for (uint32_t i = 0; i < 12; ++i) {
        res_[i] = gf256to2_mul(a_[i], b);
    }
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf256(gf256to24 res,
                                       const gf256to24 a,
                                       const gf256 b) {
    for (uint32_t i = 0; i < 24; ++i) {
        res[i] = gf256_mul(a[i], b);
    }
}

/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul_gf16(gf256to24 res,
                                      const gf256to24 a,
                                      const gf16 b) {
    const gf256 t = gf256_expand_tab[b];
    gf256to24_mul_gf256(res, a, t);
}

#ifdef USE_AVX2

static inline void gf256to24_vector_add_u256(gf256to24 *out,
                                        const gf256to24 *in1,
                                        const uint32_t d) {
    size_t i = sizeof(gf256to24) * d;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)out) ^
                            _mm256_loadu_si256((__m256i *)in1));
        i   -= 32u;
        in1 += 2u;
        out += 2u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)out) ^
                         _mm_loadu_si128((__m128i *)in1));
        i   -= 16u;
        in1 += 1u;
        out += 1u;
    }

    for(uint32_t j = 0; j<((i+sizeof(gf256to24) - 1)/sizeof(gf256to24)); j++) {
        ((uint8_t *)out)[j] ^= ((uint8_t *)in1)[j];
    }
}

/// vector1 = vector2 + vector3
static inline void gf256to24_vector_add_u256_(gf256to24 *out,
                                         const gf256to24 *in1,
                                         const gf256to24 *in2,
                                         const uint32_t d) {
    size_t i = sizeof(gf256to24) * d;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        i   -= 32u;
        in1 += 2u;
        in2 += 2u;
        out += 2u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        i   -= 16u;
        in1 += 1u;
        in2 += 1u;
        out += 1u;
    }

    for(uint32_t j = 0; j<((i+sizeof(gf256to24) - 1)/sizeof(gf256to24)); j++) {
        ((uint8_t *)out)[j] = ((uint8_t *)in1)[j] ^ ((uint8_t *)in2)[j];
    }
}

/// out += in1 * scalar
static inline void gf256to24_vector_add_multiple_u256(gf256to24 *out,
                                                 const gf256to24 scalar,
                                                 const gf256to24 *in1,
                                                 const uint32_t d) {
    size_t i = d;
    __m256i mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    const __m256i s256 = _mm256_maskload_epi64((const long long *)scalar, mask);

    // avx2 code
    while (i > 0) {
        const __m256i l1 = _mm256_maskload_epi64((const long long *)in1, mask);
        const __m256i o1 = _mm256_maskload_epi64((const long long *)out, mask);
        const __m256i t1 = o1 ^ gf256to8v_mul_u256(l1, s256);
        // TODO not correct
        _mm256_maskstore_epi64((long long *)out, mask, t1);

        i   -= 1u;
        in1 += 1u;
        out += 1u;
    }
}
#endif
#endif
