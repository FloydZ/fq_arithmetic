#ifndef FQ_ARITHMETIC_GF256TO24_ARITH_H
#define FQ_ARITHMETIC_GF256TO24_ARITH_H
#include <stdint.h>
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

/// multiplication using karatsube
/// \param res
/// \param a
/// \param b
static inline void gf256to24_mul(gf256to24 res,
                                 const gf256to24 a,
                                 const gf256to24 b) {
    uint64_t *a_ = (uint64_t *)a;
    uint64_t *b_ = (uint64_t *)b;
    uint64_t *r_ = (uint64_t *)res;

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

    const uint64_t factor = 1; //0x100000000000000;

    r_[0] = gf256to8_add(p1, p2);
    r_[0] = gf256to8_add(r_[0], p12);
    r_[0] = gf256to8_mul(r_[0], factor);
    r_[0] = gf256to8_add(r_[0], p0);

    r_[1] = gf256to8_mul(p2, factor);
    r_[1] = gf256to8_add(r_[1], p01);
    r_[1] = gf256to8_add(r_[1], p0);
    r_[1] = gf256to8_add(r_[1], p1);

    r_[2] = gf256to8_add(p012, p01);
    r_[2] = gf256to8_add(r_[2], p12);
}

static inline __m256i gf256to24v_mul(const __m256i a,
                                     const __m256i b) {
    const __m256i pi = a ^ b;
    const __m256i a0i = a ^ _mm256_permute4x64_epi64(a, 0b00010000);
    const __m256i b0i = b ^ _mm256_permute4x64_epi64(b, 0b00010000);
    const __m256i p0i = gf256to8v_mul_u256(a0i, b0i);
    return pi;
}

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

/// out += in1 * value
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
