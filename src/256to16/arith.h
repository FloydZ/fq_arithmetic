#ifndef FQ_ARITHMETIC_GF256TO16_ARITH_H
#define FQ_ARITHMETIC_GF256TO16_ARITH_H

#include <stdint.h>

#include "../256to8/arith.h"
#include "../16/arith.h"

/// GF256to16:uint8_t[16]: X^2 + x + (4294967296 * 65536 * 256 * 32)
typedef uint8_t  gf256to16[16];

static inline void gf256to16_zero(gf256to16 res) {
#ifdef __SIZEOF_INT128__
    *((__uint128_t *)res) = 0;
#else
    for (uint32_t i = 0; i < 16; i++){
        res[i] = 0;
    }
#endif
}

static inline void gf256to16_set(gf256to16 res,
                                 const uint8_t *b) {
#ifdef __SIZEOF_INT128__
    *(__uint128_t *)res = (*(__uint128_t *)b);
#else
    for (uint32_t i = 0; i < 16; i++){
        res[i] = b[i];
    }
#endif
}

static inline void gf256to16_set_(gf256to16 *res,
                                  const uint8_t *b) {
#ifdef __SIZEOF_INT128__
    *(__uint128_t *)(*res) = (*(__uint128_t *)b);
#else
    for (uint32_t i = 0; i < 16; i++){
        res[i] = b[i];
    }
#endif
}

/// X^2 + X + (4294967296*4096*256*32) = 0
/// \param res
/// \param a
/// \param b
static inline void gf256to16_mul(gf256to16 res,
                                 const gf256to16 a,
                                 const gf256to16 b) {
    const gf256to8 upper = gf256to8_mul(*(uint64_t *)((uint8_t *)a + 8), *(uint64_t *)((uint8_t *)b + 8));
    const gf256to8 lower = gf256to8_mul(*(uint64_t *)a, *(uint64_t *)b);

    // Compute lower
    const gf256to8 factor = 0x2000000000000000;
    *(uint64_t *)res = gf256to8_mul(upper, factor);
    *(uint64_t *)res = gf256to8_add(*((gf256to8 *)res), lower);

    // Compute upper
    *(uint64_t *)(res + 8) = gf256to8_mul(gf256to8_add(*(uint64_t *)a, *(uint64_t *)((uint8_t *)a + 8)),
                                          gf256to8_add(*(uint64_t *)b, *(uint64_t *)((uint8_t *)b + 8)));
    *(uint64_t *)(res + 8) = gf256to8_add(*(uint64_t *)((uint8_t *)res + 8), lower);
}

///
/// \param res
/// \param a
/// \param b
static inline void gf256to16_add(gf256to16 res,
                                 const gf256to16 a,
                                 const gf256to16 b) {
#ifdef __SIZEOF_INT128__
    *(__uint128_t *)res = (*(__uint128_t *)a) ^ (*(__uint128_t *)b);
#else
    for (uint32_t i = 0; i < 16; i++){
        res[i] = a[i] ^ b[i];
    }
#endif
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

/// \param in
/// \param b
/// \return
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

/// \param in
/// \param b
/// \return
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

static inline void gf256to16_vector_add_u256(gf256to16 *out,
                                             const gf256to16 *in1,
                                             const uint32_t d) {
    size_t i = sizeof(gf256to16) * d;

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

    for(uint32_t j = 0; j<((i+sizeof(gf256to16) - 1)/sizeof(gf256to16)); j++) {
        ((uint8_t *)out)[j] ^= ((uint8_t *)in1)[j];
    }
}

/// vector1 = vector2 + vector3
static inline void gf256to16_vector_add_u256_(gf256to16 *out,
                                              const gf256to16 *in1,
                                              const gf256to16 *in2,
                                              const uint32_t d) {
    size_t i = sizeof(gf256to16) * d;

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

    for(uint32_t j = 0; j<((i+sizeof(gf256to16) - 1)/sizeof(gf256to16)); j++) {
        ((uint8_t *)out)[j] = ((uint8_t *)in1)[j] ^ ((uint8_t *)in2)[j];
    }
}

/// out += in1 * value
static inline void gf256to16_vector_add_scalar_u256(gf256to16 *out,
                                                 const gf256to16 scalar,
                                                 const gf256to16 *in1,
                                                 const uint32_t d) {
    size_t i = sizeof(gf256to16) * d;

    uint32_t *s32 = (uint32_t *)scalar;
    const __m256i s256 =_mm256_setr_epi32(s32[0], s32[1], s32[2], s32[3], s32[0], s32[1], s32[2], s32[3]);
    const __m128i s128 =_mm_setr_epi32(s32[0], s32[1], s32[2], s32[3]);

    // avx2 code
    while (i >= 32u) {
        const __m256i l1 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i o1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t1 = o1 ^ gf256to16v_mul_u256(l1, s256);
        _mm256_storeu_si256((__m256i *)out, t1);

        i   -= 32u;
        in1 += 2u;
        out += 2u;
    }

    // sse code
    while(i >= 16u) {
        const __m128i l1 = _mm_loadu_si128((__m128i *)in1);
        const __m128i o1 = _mm_loadu_si128((__m128i *)out);
        const __m128i t1 = o1 ^ gf256to16v_mul_u128(l1, s128);
        _mm_storeu_si128((__m128i *)out, t1);

        i   -= 16u;
        in1 += 1u;
        out += 1u;
    }

    // there will we nothing leftover
}

///  vector1 = vector2 + vector3 * value
static inline void gf256to16_vector_add_scalar_u256_(gf256to16 *out,
                                                  const gf256to16 *in1,
                                                  const gf256to16 scalar,
                                                  const gf256to16 *in2,
                                                  const uint32_t d) {
    size_t i = sizeof(gf256to16) * d;

    uint32_t *s32 = (uint32_t *)scalar;
    const __m256i s256 =_mm256_setr_epi32(s32[0], s32[1], s32[2], s32[3], s32[0], s32[1], s32[2], s32[3]);
    const __m128i s128 =_mm_setr_epi32(s32[0], s32[1], s32[2], s32[3]);

    // avx2 code
    while (i >= 32u) {
        const __m256i l1 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i l2 = _mm256_loadu_si256((__m256i *)in2);
        const __m256i t1 = l1 ^ gf256to16v_mul_u256(l2, s256);
        _mm256_storeu_si256((__m256i *)out, t1);

        i   -= 32u;
        in1 += 2u;
        in2 += 2u;
        out += 2u;
    }

    // sse code
    while(i >= 16u) {
        const __m128i l1 = _mm_loadu_si128((__m128i *)in1);
        const __m128i l2 = _mm_loadu_si128((__m128i *)in2);
        const __m128i t1 = l1 ^ gf256to16v_mul_u128(l2, s128);
        _mm_storeu_si128((__m128i *)out, t1);

        i   -= 16u;
        in1 += 1u;
        in2 += 1u;
        out += 1u;
    }

    // there will we nothing leftover
}
#endif

#ifdef __AVX512F__

#endif

#endif //FQ_ARITHMETIC_ARITH_H
