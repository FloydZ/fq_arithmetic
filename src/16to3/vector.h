#pragma once
#include "arith.h"

/// \param n
/// \return
static inline gf16to3* gf16to3_vector_alloc(const uint32_t n) {
    return (gf16to3 *)malloc(n*sizeof(gf16to3));
}

/// \param v
/// \param n
static inline void gf16to3_vector_print(gf16to3 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

/// \param v
/// \param n
static inline void gf16to3_vector_zero(gf16to3 *v,
                                       const uint32_t n) {
    memset(v, 0, n*sizeof(gf16to3));
}

/// \param v
/// \param n
static inline void gf16to3_vector_random(gf16to3 *v,
                                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFFF;
    }
}

/// \param v1
/// \param v2
/// \param n
static inline void gf16to3_vector_copy(gf16to3 *__restrict__ v1,
                                       const gf16to3 *__restrict__ v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add(gf16to3 *out,
                                      const gf16to3 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add_gf16(gf16to3 *__restrict__ out,
                                           const gf16 *__restrict__ in,
                                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const ff_t t = gf16_vector_get(in, i);
        out[i] ^= t;
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_add(gf16to3 *out,
                                             const gf16to3 c,
                                             const gf16to3 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(c, in[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_add_gf16(gf16to3 *__restrict__ out,
                                                  const gf16 c,
                                                  const gf16to3 *__restrict__ in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf16to3_mul(in[i], c);
    }
}

/// \brief vector1 = vector2 * scalar
///
/// \param[out] vector1 Vector over ff_mu
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf16to3_vector_scalar_gf16(gf16to3 *vector1,
                                              const gf16to3 scalar,
                                              const gf16 *vector2,
                                              const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const gf16to3 t = gf16_vector_get(vector2, i);
        vector1[i] = gf16to3_mul(scalar, t);
    }
}

/// \brief vector1 += vector2 * scalar
///
/// \param[out] vector1 Vector over ff_mu
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf16to3_vector_add_scalar_gf16(gf16to3 *vector1,
                                                  const gf16to3 scalar,
                                                  const gf16 *vector2,
                                                  const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const gf16to3 t = gf16_vector_get(vector2, i);
        vector1[i] ^= gf16to3_mul(scalar, t);
    }
}

#ifdef USE_AVX2


/// loads 8 elements/ 4 bytes and extends them to gf256
static inline __m128i gf16to3_vector_extend_gf16_x8(const gf16 *in) {
    const uint32_t t11 = *((uint32_t *)(in + 0));
    const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
    const __m128i t1 = _mm_set_epi64x(0, t21);
    return _mm_cvtepi8_epi16(t1);
}

/// loads 16 elements/ 8 bytes and extends them to gf256
static inline __m256i gf16to3_vector_extend_gf16_x16(const gf16 *in) {
    const uint32_t t11 = *((uint32_t *)(in + 0));
    const uint32_t t12 = *((uint32_t *)(in + 4));
    const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
    const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
    const __m128i t1 = _mm_set_epi64x(t22, t21);
    return _mm256_cvtepu8_epi16(t1);
}


static inline __m256i gf16to3_vector_extend_gf16_x32(const gf16 *in) {
    const uint16_t t11 = *((uint16_t *)(in + 0));
    const uint16_t t12 = *((uint16_t *)(in + 2));
    const uint16_t t13 = *((uint16_t *)(in + 4));
    const uint16_t t14 = *((uint16_t *)(in + 6));

    const uint64_t t21 = _pdep_u64(t11, 0x000F000F000F000F);
    const uint64_t t22 = _pdep_u64(t12, 0x000F000F000F000F);
    const uint64_t t23 = _pdep_u64(t13, 0x000F000F000F000F);
    const uint64_t t24 = _pdep_u64(t14, 0x000F000F000F000F);
    return _mm256_setr_epi64x(t21, t22, t23, t24);
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16to3_vector_add_gf16_u256(gf16to3 *__restrict__ out,
                                                const gf16 *__restrict__ in,
                                                const uint32_t n) {
    uint32_t i = n;
    // avx2 code
    while (i >= 16u) {
        const __m256i m2 = gf16to3_vector_extend_gf16_x16(in);
        const __m256i m1 = _mm256_loadu_si256((const __m256i *)out);

        _mm256_storeu_si256((__m256i *)out, m1 ^ m2);
        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32)));
        uint16_t *tmp2 = (uint16_t *)tmp;
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = in[j]; }

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);

        for (uint32_t j = 0; j < i; ++j) { tmp2[j] = out[j]; }
        const __m256i m1 = _mm256_load_si256((const __m256i *)tmp2);

        _mm256_store_si256((__m256i *)tmp, t1^m1);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }

    // while (i >= 8u) {
    //     const uint32_t t11 = *((uint32_t *)(in + 0));
    //     const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
    //     const __m128i t1 = _mm_set_epi64x(0, t21);
    //     const __m128i m2 = _mm_cvtepi8_epi16(t1);
    //     const __m128i m1 = _mm_loadu_si128((__m128i *)out);

    //     _mm_storeu_si128((__m128i *)out, m1 ^ m2);
    //     i   -= 8u;
    //     in  += 4u;
    //     out += 8u;
    // }

    // // TODO maybe just write everything into a buffer and
    // // apply the avx2 code
    // for (; i > 0; --i) {
    //     *out++ ^= (*in) & 0xF;
    //     i -= 1;
    //     if (i) {
    //         *out++ ^= (*in) >> 4;
    //         in++;
    //     }
    // }
}

/// out += in1
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256(gf16to3 *__restrict__ out,
                                           const gf16to3 *__restrict__ in1,
                                           const uint32_t d) {
    uint32_t i = d;
    // avx2 code
    while (i >= 16u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)out) ^
                            _mm256_loadu_si256((__m256i *)in1));
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)out) ^
                         _mm_loadu_si128((__m128i *)in1));
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++;
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_u256(gf16to3 *out,
                                                  const gf16to3 a,
                                                  const gf16to3 *in1,
                                                  const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf16to3v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf16to3_mul(a, *in1++);
    }
}

/// out = in1 + a*in2
static inline void gf16to3_vector_scalar_add_u256_3(gf16to3 *out,
                                                    const gf16to3 *in1,
                                                    const gf16to3 a,
                                                    const gf16to3 *in2,
                                                    const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in2);
        const __m256i t3 = gf16to3v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++ ^ gf16to3_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf16to3_vector_scalar_add_gf16_u256(gf16to3 *out,
                                                        const gf16 a,
                                                        const gf16to3 *in1,
                                                        const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);
    const __m128i a128 = _mm_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf16v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        const __m128i t1 = _mm_loadu_si128((__m128i *)out);
        const __m128i t2 = _mm_loadu_si128((__m128i *)in1);
        const __m128i t3 = gf16v_mul_u128(t2, a128);
        const __m128i t = t1 ^ t3;

        _mm_storeu_si128((__m128i *)out, t);
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf16to3_mul_gf16(*in1++, a);
    }
}

/// \brief vector1 = vector2 * scalar
///
/// \param[out] vector1 Vector over ff_mu
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf16to3_vector_add_scalar_gf16_u256(gf16to3 *vector1,
                                                       const gf16to3 scalar,
                                                       const gf16 *vector2,
                                                       const uint32_t ncols) {
    uint32_t i = ncols;
    const __m256i s = _mm256_set1_epi16(scalar);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = gf16to3_vector_extend_gf16_x16(vector2);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)vector1);

        _mm256_storeu_si256((__m256i *)vector1, t2^t3);
        i       -= 16u;
        vector2 += 8u;
        vector1 += 16u;
    }

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32)));
        uint16_t *tmp2 = (uint16_t *)tmp;
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = vector2[j]; }

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_store_si256((__m256i *)tmp, t2);

        for (uint32_t j = 0; j < i; ++j) { vector1[j] ^= tmp2[j]; }
    }
}

/// \param out = scalar*in
/// \param in
/// \param n
static inline void gf16to3_vector_scalar_gf16_u256(gf16to3 *__restrict__ out,
                                                   gf16to3 scalar,
                                                   const gf16 *__restrict__ in,
                                                   const uint32_t n) {
    uint32_t i = n;
    const __m256i s = _mm256_set1_epi16(scalar);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = gf16to3_vector_extend_gf16_x16(in);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_storeu_si256((__m256i *)out, t2);
        i   -= 16u;
        in  += 8u;
        out += 16u;
    }

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32)));
        uint16_t *tmp2 = (uint16_t *)tmp;
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = in[j]; }

        const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
        const __m256i t2 = gf16to3v_mul_u256(t1, s);

        _mm256_store_si256((__m256i *)tmp, t2);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp2[j]; }
    }
}
#endif