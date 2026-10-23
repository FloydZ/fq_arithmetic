#pragma once

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "arith.h"
#include "../2/vector.h"

/// \param n size of the vector
/// \return vector(n), zero initialized
static inline gf2to12* gf2to12_vector_alloc(const uint32_t n) {
    return (gf2to12 *)calloc(n, sizeof(gf2to12));
}

/// \return v[i]
static inline gf2to12 gf2to12_vector_get(const gf2to12 *v,
                                         const uint32_t i) {
    return v[i];
}

/// v[i] = a
static inline void gf2to12_vector_set(gf2to12 *v,
                                      const uint32_t i,
                                      const gf2to12 a) {
    v[i] = a;
}

/// \param v vector to print
/// \param n size of the vector
static inline void gf2to12_vector_print(const gf2to12 *v,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

/// \param v = 0
/// \param n size of vector
static inline void gf2to12_vector_zero(gf2to12 *v,
                                       const uint32_t n) {
    memset(v, 0, n*sizeof(gf2to12));
}

/// \param v v[i] = rand()
/// \param n size of vector
static inline void gf2to12_vector_random(gf2to12 *v,
                                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFFF;
    }
}

///
/// \param v1 = v2
/// \param v2
/// \param n size of the two vectors
static inline void gf2to12_vector_copy(gf2to12 *__restrict__ v1,
                                       const gf2to12 *__restrict__ v2,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// a = b
static inline void gf2to12_vector_set_to_gf2(gf2to12 *a,
                                             const gf2 *b,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 t = gf2_vector_get(b, i);
        a[i] = t;
    }
}

/// \param out[in/out] += in
/// \param in[in]
/// \param n number of elements
static inline void gf2to12_vector_add(gf2to12 *out,
                                      const gf2to12 *in,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// \param out[in/out] += in
/// \param in[in]
/// \param n number of elements
static inline void gf2to12_vector_add_gf2(gf2to12 *__restrict__ out,
                                          const gf2 *__restrict__ in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 t = gf2_vector_get(in, i);
        out[i] ^= t;
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf2to12_vector_scalar_add(gf2to12 *out,
                                             const gf2to12 c,
                                             const gf2to12 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to12_mul(c, in[i]);
    }
}

/// out = in1 + c*in2
/// \param n number of elements
static inline void gf2to12_vector_scalar_add_v2(gf2to12 *out,
                                                const gf2to12 *in1,
                                                const gf2to12 c,
                                                const gf2to12 *in2,
                                                const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf2to12_mul(c, in2[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf2to12_vector_scalar_add_gf2(gf2to12 *__restrict__ out,
                                                  const gf2 c,
                                                  const gf2to12 *__restrict__ in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to12_mul_gf2(in[i], c);
    }
}


/// \param out += t*in
/// \param in
/// \param n length of the vectors
static inline void gf2to12_vector_scalar_add_gf2_v3(gf2to12 *__restrict__ out,
                                                    const gf2to12 t,
                                                    const gf2 *__restrict__ in,
                                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        const gf2to12 b = gf2to12_mul_gf2(t, a);
        out[i] ^= b;
    }
}

/// \param a
/// \param b
/// \param n length of the vector
/// \return sum(a[i] * b[i])
static inline gf2to12 gf2to12_vector_mul_acc(const gf2to12 *a,
                                             const gf2to12 *b,
                                             const uint32_t n) {
    gf2to12 acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc ^= gf2to12_mul(a[i], b[i]);
    }

    return acc;
}

#ifdef USE_AVX2


/// out += in1
/// \param out[in/out]
/// \param in1[in]
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

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param d number of elements NOT bytes
static inline void gf2to12_vector_add_u256_v2(gf2to12 *out,
                                              const gf2to12 *in1,
                                              const gf2to12 *in2,
                                              const uint32_t d) {
    uint32_t i = d;
    // avx2 code
    while (i >= 16u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 8u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ *in2++;
    }
}

/// \param out += in
/// \param in
/// \param n
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
        const uint32_t t11 = *in;
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);
        const __m128i m1 = _mm_loadu_si128((__m128i *)out);

        _mm_storeu_si128((__m128i *)out, m1 ^ m2);
        i   -= 8u;
        in  += 1u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
        const __m128i m1 = _mm_load_si128((__m128i *)tmp);

        const uint32_t t11 = in[0];
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);

        _mm_store_si128((__m128i *)tmp, m2 ^ m1);

        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// \param out = in1 + in2
/// \param in
/// \param n
static inline void gf2to12_vector_add_gf2_u256_v2(gf2to12 *__restrict__ out,
                                                  const gf2to12 *__restrict__ in1,
                                                  const gf2 *__restrict__ in2,
                                                  const uint32_t n) {
    uint32_t i = n;

    while (i >= 16u) {
        const uint8_t t11 = *(in2 + 0);
        const uint8_t t12 = *(in2 + 1);

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);

        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i m2 = _mm256_cvtepu8_epi16(t1);
        const __m256i m1 = _mm256_loadu_si256((const __m256i *)in1);

        _mm256_storeu_si256((__m256i *)out, m1 ^ m2);
        i   -= 16u;
        in2 += 2u;
        in1 += 16u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint32_t t11 = *in2;
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);
        const __m128i m1 = _mm_loadu_si128((__m128i *)in1);

        _mm_storeu_si128((__m128i *)out, m1 ^ m2);
        i   -= 8u;
        in2 += 1u;
        in1 += 8u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = in1[j];}
        const __m128i m1 = _mm_load_si128((__m128i *)tmp);

        const uint32_t t11 = in2[0];
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);

        _mm_store_si128((__m128i *)tmp, m2 ^ m1);

        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// out = in1 + a*in2
static inline void gf2to12_vector_scalar_add_u256_v2(gf2to12 *out,
                                                     const gf2to12 *in1,
                                                     const gf2to12 a,
                                                     const gf2to12 *in2,
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
        *out++ = *in1++ ^ gf2to12_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_u256(gf2to12 *out,
                                                  const gf2to12 a,
                                                  const gf2to12 *in1,
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

/// out += a*in1
static inline void gf2to12_vector_scalar_add_gf2_u256(gf2to12 *__restrict__ out,
                                                      const gf2 a,
                                                      const gf2to12 *__restrict__ in1,
                                                      const uint32_t d) {
    uint32_t i = d;
    const uint16_t aa = (-a) & 0xFFF;
    const __m256i a256 = _mm256_set1_epi16(aa);

    // avx2 code
    while (i >= 16u) {
        const __m256i t1 = _mm256_loadu_si256((__m256i *)out);
        const __m256i t2 = _mm256_loadu_si256((__m256i *)in1);
        const __m256i t3 = t2 & a256;
        const __m256i t = t1 ^ t3;

        _mm256_storeu_si256((__m256i *)out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++ & aa;
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_gf2_u256_v3(gf2to12 *__restrict__ out,
                                                         const gf2to12 a,
                                                         const gf2 *__restrict__ in,
                                                         const uint32_t n) {
    uint32_t i = n;
    const __m256i A256 = _mm256_set1_epi16(a);
    const __m128i A128 = _mm_set1_epi16(a);

    while (i >= 16u) {
        const uint8_t t11 = *(in + 0);
        const uint8_t t12 = *(in + 1);

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);

        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i mi = _mm256_cvtepu8_epi16(t1);
        const __m256i mo = _mm256_loadu_si256((const __m256i *)out);
        const __m256i t2 = gf2to12v_mul_gf2_u256(A256, mi);

        _mm256_storeu_si256((__m256i *)out, mo ^ t2);
        i   -= 16u;
        in  += 2u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint32_t t11 = *in;
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i mi = _mm_cvtepi8_epi16(t1);
        const __m128i mo = _mm_loadu_si128((__m128i *)out);
        const __m128i t2 = gf2to12v_mul_gf2_u128(A128, mi);

        _mm_storeu_si128((__m128i *)out, mo ^ t2);
        i   -= 8u;
        in  += 1u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};

        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
        const uint32_t t11 = *in;
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i mi = _mm_cvtepi8_epi16(t1);
        const __m128i mo = _mm_load_si128((__m128i *)tmp);
        const __m128i t2 = gf2to12v_mul_gf2_u128(A128, mi);

        _mm_store_si128((__m128i *)tmp, mo ^ t2);

        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// \param a
/// \param b
/// \param n length of the vector
/// \return sum(a[i] * b[i])
static inline gf2to12 gf2to12_vector_mul_acc_u256(const gf2to12 *a,
                                                  const gf2to12 *b,
                                                  const uint32_t n) {
    uint32_t i = n;

    __m256i acc = _mm256_setzero_si256();
    // avx2 code
    while (i >= 16u) {
        const __m256i ta = _mm256_loadu_si256((__m256i *)a);
        const __m256i tb = _mm256_loadu_si256((__m256i *)b);
        const __m256i t1 = gf2to12v_mul_u256(ta, tb);
        acc ^= t1;

        i -= 16u;
        a += 16u;
        b += 16u;
    }

    if (i) {
        uint16_t tmpa[16] __attribute__((aligned(32))) = {0},
                 tmpb[16] __attribute__((aligned(32))) = {0};
        for (uint32_t j = 0; j < i; j++) {
            tmpa[j] = a[j];
            tmpb[j] = b[j];
        }

        const __m256i ta = _mm256_load_si256((__m256i *)tmpa);
        const __m256i tb = _mm256_load_si256((__m256i *)tmpb);
        const __m256i t1 = gf2to12v_mul_u256(ta, tb);
        acc ^= t1;
    }

    return gf2to12_hadd_u256(acc);
}


static inline void gf2to12_vector_set_to_gf2_u256(gf2to12 *out,
                                                  const gf2 *in,
                                                  const uint32_t n) {
    uint32_t bytes = n/8;
    const uint32_t limit = n % 16;

    while (bytes >= 2u) {
        const uint8_t t11 = *(in + 0);
        const uint8_t t12 = *(in + 1);

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);

        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i mi = _mm256_cvtepu8_epi16(t1);
        _mm256_storeu_si256((__m256i *)out, mi);
        in  += 2u;
        out += 16u;
        bytes -= 2;
    }

    if (limit) {
        uint16_t tmp[16] __attribute__((aligned(32))) = {0};
        uint8_t t11,t12=0;

        t11 = *(in + 0);
        if(limit > 8) { t12 = *(in + 1); }

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i mi = _mm256_cvtepu8_epi16(t1);
        _mm256_store_si256((__m256i *)tmp, mi);

        for (uint32_t j = 0; j < limit; j++) { out[j] = tmp[j];}
    }
}

#ifdef USE_AVX512

/// out += in
/// \param n number of elements
static inline void gf2to12_vector_add_u512(gf2to12 *out,
                                           const gf2to12 *in,
                                           const uint32_t n) {
    uint32_t i = n;
    while (i >= 32u) {
        const __m512i a = _mm512_loadu_si512((const void *)in);
        const __m512i b = _mm512_loadu_si512((const void *)out);
        _mm512_storeu_si512((void *)out, gf2to12v_add_u512(a, b));
        i   -= 32u;
        in  += 32u;
        out += 32u;
    }

    if (i) {
        const __mmask32 m = (__mmask32)((1ull << i) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi16(m, (const void *)in);
        const __m512i b = _mm512_maskz_loadu_epi16(m, (const void *)out);
        _mm512_mask_storeu_epi16((void *)out, m, gf2to12v_add_u512(a, b));
    }
}

/// out = in1 + in2
/// \param n number of elements
static inline void gf2to12_vector_add_u512_v2(gf2to12 *out,
                                              const gf2to12 *in1,
                                              const gf2to12 *in2,
                                              const uint32_t n) {
    uint32_t i = n;
    while (i >= 32u) {
        const __m512i a = _mm512_loadu_si512((const void *)in1);
        const __m512i b = _mm512_loadu_si512((const void *)in2);
        _mm512_storeu_si512((void *)out, gf2to12v_add_u512(a, b));
        i   -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    if (i) {
        const __mmask32 m = (__mmask32)((1ull << i) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi16(m, (const void *)in1);
        const __m512i b = _mm512_maskz_loadu_epi16(m, (const void *)in2);
        _mm512_mask_storeu_epi16((void *)out, m, gf2to12v_add_u512(a, b));
    }
}

/// out += c*in
/// \param n number of elements
static inline void gf2to12_vector_scalar_add_u512(gf2to12 *out,
                                                  const gf2to12 c,
                                                  const gf2to12 *in,
                                                  const uint32_t n) {
    const __m512i c512 = _mm512_set1_epi16(c);
    uint32_t i = n;
    while (i >= 32u) {
        const __m512i a = _mm512_loadu_si512((const void *)in);
        const __m512i b = _mm512_loadu_si512((const void *)out);
        const __m512i t = gf2to12v_mul_u512(a, c512);
        _mm512_storeu_si512((void *)out, gf2to12v_add_u512(t, b));
        i   -= 32u;
        in  += 32u;
        out += 32u;
    }

    if (i) {
        const __mmask32 m = (__mmask32)((1ull << i) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi16(m, (const void *)in);
        const __m512i b = _mm512_maskz_loadu_epi16(m, (const void *)out);
        const __m512i t = gf2to12v_mul_u512(a, c512);
        _mm512_mask_storeu_epi16((void *)out, m, gf2to12v_add_u512(t, b));
    }
}

/// out = in1 + c*in2
/// \param n number of elements
static inline void gf2to12_vector_scalar_add_u512_v2(gf2to12 *out,
                                                     const gf2to12 *in1,
                                                     const gf2to12 c,
                                                     const gf2to12 *in2,
                                                     const uint32_t n) {
    const __m512i c512 = _mm512_set1_epi16(c);
    uint32_t i = n;
    while (i >= 32u) {
        const __m512i a = _mm512_loadu_si512((const void *)in2);
        const __m512i b = _mm512_loadu_si512((const void *)in1);
        const __m512i t = gf2to12v_mul_u512(a, c512);
        _mm512_storeu_si512((void *)out, gf2to12v_add_u512(t, b));
        i   -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    if (i) {
        const __mmask32 m = (__mmask32)((1ull << i) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi16(m, (const void *)in2);
        const __m512i b = _mm512_maskz_loadu_epi16(m, (const void *)in1);
        const __m512i t = gf2to12v_mul_u512(a, c512);
        _mm512_mask_storeu_epi16((void *)out, m, gf2to12v_add_u512(t, b));
    }
}

/// \return sum(a[i] * b[i])
static inline gf2to12 gf2to12_vector_mul_acc_u512(const gf2to12 *a,
                                                  const gf2to12 *b,
                                                  const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = n;
    while (i >= 32u) {
        const __m512i ta = _mm512_loadu_si512((const void *)a);
        const __m512i tb = _mm512_loadu_si512((const void *)b);
        acc = gf2to12v_add_u512(acc, gf2to12v_mul_u512(ta, tb));
        i -= 32u;
        a += 32u;
        b += 32u;
    }

    if (i) {
        const __mmask32 m = (__mmask32)((1ull << i) - 1ull);
        const __m512i ta = _mm512_maskz_loadu_epi16(m, (const void *)a);
        const __m512i tb = _mm512_maskz_loadu_epi16(m, (const void *)b);
        acc = gf2to12v_add_u512(acc, gf2to12v_mul_u512(ta, tb));
    }

    return gf2to12_hadd_u512(acc);
}
#endif // USE_AVX512
#endif // USE_AVX2

#ifdef USE_NEON
#include <arm_neon.h>

/// out += in1
/// \param out[in/out]
/// \param in1[in]
/// \param d  number of elements NOT bytes
static inline void gf2to12_vector_add_u128(gf2to12 *__restrict__ out,
                                           const gf2to12 *__restrict__ in1,
                                           const uint32_t d) {
    uint32_t i = d;
    while (i >= 16u) {
        const uint16x8x2_t m1 = vld1q_u16_x2(in1);
        const uint16x8x2_t m2 = vld1q_u16_x2(out);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(out, m3);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }
    while(i >= 8u) {
        const uint16x8_t m1 = vld1q_u16(in1);
        const uint16x8_t m2 = vld1q_u16(out);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(out, m3);
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++;
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param d number of elements NOT bytes
static inline void gf2to12_vector_add_u128_v2(gf2to12 *out,
                                              const gf2to12 *in1,
                                              const gf2to12 *in2,
                                              const uint32_t d) {
    uint32_t i = d;
    while (i >= 16u) {
        const uint16x8x2_t m1 = vld1q_u16_x2(in1);
        const uint16x8x2_t m2 = vld1q_u16_x2(in2);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(out, m3);
        i   -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    while(i >= 8u) {
        const uint16x8_t m1 = vld1q_u16(in1);
        const uint16x8_t m2 = vld1q_u16(in2);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(out, m3);
        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ *in2++;
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf2to12_vector_add_gf2_u128(gf2to12 *__restrict__ out,
                                               const gf2 *__restrict__ in,
                                               const uint32_t n) {
    uint32_t i = n;

    while (i >= 16u) {
        const uint16x8x2_t m1 = vld1q_u16_x2(out);
        const uint16x8x2_t m2 = gf2to12v_expand_gf2_x16_u256(in);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(out, m3);
        i   -= 16u;
        in  += 2u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint16x8_t m1 = vld1q_u16(out);
        const uint16x8_t m2 = gf2to12v_expand_gf2_x8_u128(in);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(out, m3);
        i   -= 8u;
        in  += 1u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
        const uint16x8_t m1 = vld1q_u16(tmp);
        const uint16x8_t m2 = gf2to12v_expand_gf2_x8_u128(in);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(tmp, m3);
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// \param out = in1 + in2
/// \param in
/// \param n
static inline void gf2to12_vector_add_gf2_u128_v2(gf2to12 *__restrict__ out,
                                                  const gf2to12 *__restrict__ in1,
                                                  const gf2 *__restrict__ in2,
                                                  const uint32_t n) {
    uint32_t i = n;

    while (i >= 16u) {
        const uint16x8x2_t m1 = vld1q_u16_x2(in1);
        const uint16x8x2_t m2 = gf2to12v_expand_gf2_x16_u256(in2);
        const uint16x8x2_t m3 = veorq_u16_x2(m1, m2);
        vst1q_u16_x2(out, m3);
        i   -= 16u;
        in2 += 2u;
        in1 += 16u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint16x8_t m1 = vld1q_u16(in1);
        const uint16x8_t m2 = gf2to12v_expand_gf2_x8_u128(in2);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(out, m3);
        i   -= 8u;
        in2 += 1u;
        in1 += 8u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = in1[j];}
        const uint16x8_t m1 = vld1q_u16(tmp);
        const uint16x8_t m2 = gf2to12v_expand_gf2_x8_u128(in2);
        const uint16x8_t m3 = veorq_u16(m1, m2);
        vst1q_u16(tmp, m3);
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// out = in1 + a*in2
static inline void gf2to12_vector_scalar_add_u128_v2(gf2to12 *out,
                                                     const gf2to12 *in1,
                                                     const gf2to12 a,
                                                     const gf2to12 *in2,
                                                     const uint32_t d) {
    uint32_t i = d;
    const uint16x8_t a128 = vdupq_n_u16(a);

    while (i >= 8u) {
        const uint16x8_t t1 = vld1q_u16(in1);
        const uint16x8_t t2 = vld1q_u16(in2);
        const uint16x8_t t3 = gf2to12v_mul_u128(t2, a128);
        vst1q_u16(out, veorq_u16(t1, t3));
        i   -= 8u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ = *in1++ ^ gf2to12_mul(a, *in2++);
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_u128(gf2to12 *out,
                                                  const gf2to12 a,
                                                  const gf2to12 *in1,
                                                  const uint32_t d) {
    uint32_t i = d;
    const uint16x8_t a128 = vdupq_n_u16(a);

    while (i >= 8u) {
        const uint16x8_t t1 = vld1q_u16(out);
        const uint16x8_t t2 = vld1q_u16(in1);
        const uint16x8_t t3 = gf2to12v_mul_u128(t2, a128);
        vst1q_u16(out, veorq_u16(t1, t3));
        i   -= 8u;
        in1 += 8u;
        out += 8u;
    }

    for (; i > 0; --i) {
        *out++ ^= gf2to12_mul(a, *in1++);
    }
}


/// out += a*in1
static inline void gf2to12_vector_scalar_add_gf2_u128(gf2to12 *__restrict__ out,
                                                      const gf2 a,
                                                      const gf2to12 *__restrict__ in1,
                                                      const uint32_t d) {
    uint32_t i = d;
    const uint16_t aa = (-a) & 0xFFF;
    const uint16x8x2_t a256 = vdupq_n_u16_x2(aa);

    while (i >= 16u) {
        const uint16x8x2_t t1 = vld1q_u16_x2(out);
        const uint16x8x2_t t2 = vld1q_u16_x2(in1);
        const uint16x8x2_t t3 = vandq_u16_x2(t2, a256);
        const uint16x8x2_t t = veorq_u16_x2(t1, t3);

        vst1q_u16_x2(out, t);
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *out++ ^= *in1++ & aa;
    }
}

/// out += a*in1
static inline void gf2to12_vector_scalar_add_gf2_u128_v3(gf2to12 *__restrict__ out,
                                                         const gf2to12 a,
                                                         const gf2 *__restrict__ in,
                                                         const uint32_t n) {
    uint32_t i = n;
    const uint16x8x2_t A256 = vdupq_n_u16_x2(a);
    const uint16x8_t A128 = vdupq_n_u16(a);

    while (i >= 16u) {
        const uint16x8x2_t mo = vld1q_u16_x2(out);
        const uint16x8x2_t mi = gf2to12v_expand_gf2_x16_u256(in);
        const uint16x8x2_t t1 = gf2to12v_mul_gf2_u256(A256, mi);
        const uint16x8x2_t t2 = veorq_u16_x2(mo, t1);
        vst1q_u16_x2(out, t2);
        i   -= 16u;
        in  += 2u;
        out += 16u;
    }

    while (i >= 8u) {
        const uint16x8_t mo = vld1q_u16(out);
        const uint16x8_t mi = gf2to12v_expand_gf2_x8_u128(in);
        const uint16x8_t t1 = gf2to12v_mul_gf2_u128(A128, mi);
        const uint16x8_t t2 = veorq_u16(mo, t1);
        vst1q_u16(out, t2);
        i   -= 8u;
        in  += 1u;
        out += 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}

        const uint16x8_t mo = vld1q_u16(tmp);
        const uint16x8_t mi = gf2to12v_expand_gf2_x8_u128(in);
        const uint16x8_t t1 = gf2to12v_mul_gf2_u128(A128, mi);
        const uint16x8_t t2 = veorq_u16(mo, t1);
        vst1q_u16(tmp, t2);

        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// \param a
/// \param b
/// \param n length of the vector
/// \return sum(a[i] * b[i])
static inline gf2to12 gf2to12_vector_mul_acc_u128(const gf2to12 *a,
                                                  const gf2to12 *b,
                                                  const uint32_t n) {
    uint32_t i = n;
    uint16x8_t acc = vdupq_n_u16(0);

    while (i >= 8u) {
        const uint16x8_t ta = vld1q_u16(a);
        const uint16x8_t tb = vld1q_u16(b);
        acc = veorq_u16(acc, gf2to12v_mul_u128(ta, tb));
        i -= 8u;
        a += 8u;
        b += 8u;
    }

    gf2to12 t = gf2to12_hadd_u128(acc);
    for (; i > 0; --i) {
        t ^= gf2to12_mul(*a++, *b++);
    }
    return t;
}


static inline void gf2to12_vector_set_to_gf2_u128(gf2to12 *out,
                                                  const gf2 *in,
                                                  const uint32_t n) {
    uint32_t i = n;

    while (i >= 16u) {
        const uint16x8x2_t t = gf2to12v_expand_gf2_x16_u256(in);
        vst1q_u16_x2(out, t);

        in  += 2u;
        out += 16u;
        i   -= 16u;
    }

    while (i >= 8u) {
        vst1q_u16(out, gf2to12v_expand_gf2_x8_u128(in));
        in  += 1u;
        out += 8u;
        i   -= 8u;
    }

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16)));
        vst1q_u16(tmp, gf2to12v_expand_gf2_x8_u128(in));
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}
#endif
