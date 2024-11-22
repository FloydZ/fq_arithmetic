#pragma once
#include <string.h>

#include "arith.h"
#include "../2/vector.h"

///
/// @param n
/// @return
static inline gf2to12* gf2to12_vector_alloc(const uint32_t n) {
    return (gf2to12 *)malloc(n*sizeof(gf2to12));
}

///
/// @param v
/// @param n
static inline void gf2to12_vector_print(gf2to12 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

///
/// @param v
/// @param n
static inline void gf2to12_vector_zero(gf2to12 *v,
                                       const uint32_t n) {
    memset(v, 0, n*sizeof(gf2to12));
}

///
/// @param v
/// @param n
static inline void gf2to12_vector_rand(gf2to12 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFFF;
    }
}

///
/// @param v1
/// @param v2
/// @param n
static inline void gf2to12_vector_copy(gf2to12 *__restrict__ v1,
                                       const gf2to12 *__restrict__ v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

///
/// @param out += in
/// @param in
/// @param n
static inline void gf2to12_vector_add(gf2to12 *out,
                                      const gf2to12 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}


///
/// @param out += in
/// @param in
/// @param n
static inline void gf2to12_vector_add_gf2(gf2to12 *__restrict__ out,
                                          const gf2 *__restrict__ in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 t = gf2_vector_get(in, i);
        out[i] ^= t;
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf2to12_vector_scalar_add(gf2to12 *out,
                                             const gf2to12 c,
                                             const gf2to12 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to12_mul(c, in[i]);
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf2to12_vector_scalar_add_gf2(gf2to12 *__restrict__ out,
                                                  const gf2 c,
                                                  const gf2to12 *__restrict__ in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to12_mul_gf2(in[i], c);
    }
}

#ifdef USE_AVX2


/// out += in1
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf2to12_vector_add_u256(gf2to12 *__restrict__ out,
                                           const gf2to12 *__restrict__ in1,
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

///
/// @param out += in
/// @param in
/// @param n
static inline void gf2to12_vector_add_gf2_u256(gf2to12 *__restrict__ out,
                                               const gf2 *__restrict__ in,
                                               const uint32_t n) {
    uint32_t i = n;

    while (i >= 16u) {
        const uint8_t t11 = *(in + 0);
        const uint8_t t12 = *(in + 1);

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);

        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i m2 = _mm256_cvtepu8_epi16(t1);
        const __m256i m1 = _mm256_loadu_si256((const __m256i *)out);

        _mm256_storeu_si256((__m256i *)out, m1 ^ m2);
        i   -= 16u;
        in  += 2u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint32_t t11 = *((uint32_t *)(in + 0));
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);
        const __m128i m1 = _mm_loadu_si128((__m128i *)out);

        _mm_storeu_si128((__m128i *)out, m1 ^ m2);
        i   -= 8u;
        in  += 1u;
        out += 8u;
    }

    // TODO maybe just write everything into a buffer and
    // apply the avx2 code
    for (; i > 0; --i) {
        *out++ ^= -((*in) & 0xF);
        i -= 1;
        if (i) {
            *out++ ^= -((*in) >> 4);
            in++;
        }
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_u256(gf2to12 *__restrict__ out,
                                                  const gf2to12 a,
                                                  const gf2to12 *__restrict__ in1,
                                                  const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf2to12v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf2to12_mul(a, *in1++);
    }
}

/// out = in1 + a*in2
static inline void gf2to12_vector_scalar_add_u256_3(gf2to12 *__restrict__ out,
                                                    const gf2to12 *__restrict__ in1,
                                                    const gf2to12 a,
                                                    const gf2to12 *__restrict__ in2,
                                                    const uint32_t d) {
    uint32_t i = d;
    const __m256i a256 = _mm256_set1_epi16(a);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in2);
        const __m256i t3 = gf2to12v_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++ ^ gf2to12_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_gf2_u256(gf2to12 *__restrict__ out,
                                                        const gf2 a,
                                                        const gf2to12 *__restrict__ in1,
                                                        const uint32_t d) {
    uint32_t i = d;
    const uint16_t aa = (-a) & 0xFFF;
    const __m256i a256 = _mm256_set1_epi16(aa);
    const __m128i a128 = _mm_set1_epi16(aa);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = gf2_mul_u256(t2, a256);
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    //// sse code
    while(i >= 8u) {
        const __m128i t1 = _mm_loadu_si128((__m128i *)out);
        const __m128i t2 = _mm_loadu_si128((__m128i *)in1);
        const __m128i t3 = gf2to12v_mul_u128(t2, a128);
        const __m128i t = t1 ^ t3;

        _mm_storeu_si128((__m128i *)out, t);
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++ & a;
    }
}
#endif
