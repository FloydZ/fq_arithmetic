#pragma once

#include <string.h>

#include "arith.h"
#include "../2/vector.h"

/// \param n size of the vector
/// \return vector(n)
static inline gf2to12* gf2to12_vector_alloc(const uint32_t n) {
    return (gf2to12 *)malloc(n*sizeof(gf2to12));
}

/// \param v vector to print
/// \param n size of the vector
static inline void gf2to12_vector_print(gf2to12 *v,
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
        v[i] = rand() & 0x3FF;
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
static inline void gf2to12_vector_add_u256_v2(gf2to12 *__restrict__ out,
                                              const gf2to12 *__restrict__ in1,
                                              const gf2to12 *__restrict__ in2,
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

    if (i) {
        uint16_t tmp[8] __attribute__((aligned(16))) = {0};
        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
        const __m128i m1 = _mm_loadu_si128((__m128i *)tmp);

        uint64_t mask = (1ul << (i * 8)) - 1ul;
        const uint32_t t11 = in[0];
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101&mask);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i m2 = _mm_cvtepi8_epi16(t1);

        _mm_storeu_si128((__m128i *)tmp, m2 ^ m1);

        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
    }
}

/// out = in1 + a*in2
static inline void gf2to12_vector_scalar_add_u256_v2(gf2to12 *__restrict__ out,
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
        const uint32_t t11 = *((uint32_t *)(in + 0));
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
        uint16_t tmp[8] = {0};

        uint64_t mask = (1ul << (i * 8)) - 1ul;
        for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
        const uint32_t t11 = *((uint32_t *)(in + 0));
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101&mask);
        const __m128i t1 = _mm_set_epi64x(0, t21);
        const __m128i mi = _mm_cvtepi8_epi16(t1);
        const __m128i mo = _mm_loadu_si128((__m128i *)tmp);
        const __m128i t2 = gf2to12v_mul_gf2_u128(A128, mi);

        _mm_storeu_si128((__m128i *)tmp, mo ^ t2);

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
        const __m256i t1 = gf2_mul_u256(ta, tb);
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
        const __m256i t1 = gf2_mul_u256(ta, tb);
        acc ^= t1;
    }

    gf2to12 t = gf2to12_hadd_u256(acc);
    return t;
}


static inline void gf2to12_vector_set_to_gf2_u256(gf2to12 *out,
                                                  const gf2 *in,
                                                  const uint32_t n) {
    uint32_t bytes = (n+7)/8;

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

    if (bytes) {
        uint16_t tmp[16] __attribute__((aligned(32))) = {0};
        uint8_t t11, t12=0;

        t11 = *(in + 0);
        if (bytes > 1) { t12 = *(in + 1); }

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
        const __m128i t1 = _mm_set_epi64x(t22, t21);
        const __m256i mi = _mm256_cvtepu8_epi16(t1);
        _mm256_store_si256((__m256i *)tmp, mi);

        for (uint32_t j = 0; j < (bytes * 8); j++) { out[j] = tmp[j];}
    } 
}

#endif
