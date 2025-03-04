#pragma once

#include <string.h>
#include <stdlib.h>

#include "arith.h"


/// \return v[i]
gf2to128 gf2to128_vector_get(const gf2to128 *v,
                             const uint32_t i) {
    gf2to128 b = v[i];
    return b;
}

/// v[i] = a
void gf2to128_vector_set(gf2to128 *v,
                         const uint32_t i,
                         const gf2to128 a) {
    v[i] = a;
}

/// \param v
/// \param n
static inline void gf2to128_vector_print(gf2to128 *v,
                                         const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2to128 t = gf2to128_vector_get(v, i);
        printf("%llx ", (unsigned long long)t);
    }
    printf("\n");
}

/// \param n
/// \return
static inline gf2to128* gf2to128_vector_alloc(const uint32_t n) {
    return (gf2to128 *)malloc(n*sizeof(gf2to128));
}

/// \param v
/// \param n
static inline void gf2to128_vector_zero(gf2to128 *v,
                                    const uint32_t n) {
    memset(v, 0, n *sizeof(gf2to128));
}

/// \param v
/// \param n
static inline void gf2to128_vector_random(gf2to128 *v,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2to128 a = ((gf2to128)rand() <<  0u) ^ ((gf2to128)rand() << 32u) ^ 
                           ((gf2to128)rand() << 64u) ^ ((gf2to128)rand() << 96u);
        v[i] = a;
    }
}

/// \param v1
/// \param v2
/// \param n
static inline void gf2to128_vector_copy(gf2to128 *__restrict__ v1,
                                        const gf2to128 *__restrict__ v2,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf2to128_vector_add(gf2to128 *out,
                                       const gf2to128 *in,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}


#ifdef USE_AVX2 

/// poly1 = scalar * poly2
/// \param poly1[out]
/// \param scalar[in]
/// \param poly2[in]
static inline
void gf16_vector_multiple_u256(gf2to128 *poly1, 
                               const gf2to128 scalar, 
                               const gf2to128 *poly2,
                               const uint32_t d) {
    uint32_t i = 0;
    const __m256i s = _mm256_set_m128((__m128i)scalar, (__m128i)scalar);
    for (; (i+2) <= d; i+=2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(poly2 + i));
        const __m256i b = gf2to128v_mul_u256(s, b);
        _mm256_storeu_si256((__m256i *)(poly1 + i), b);
    }

    if (d % 2) {
        poly1[i] = mul_avx(scalar, poly2[i]);
    }
}
#endif
