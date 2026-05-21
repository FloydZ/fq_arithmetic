#pragma once

#include <emmintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"
#include "../16/vector.h"

/// \param r[out]:
/// \param v[in]: vector
/// \param i[in]: position to return
/// \return v[i]
static inline
void gf16to32_vector_get(gf16to32 r,
                         const gf16to32 *v,
                         const uint32_t i) {
    for (uint32_t t = 0; t < 16; t++ ) {
        r[t] = v[i][t];
    }
}

/// \param v[out]: v[i] = a
/// \param i[in]: index position
/// \param a[in]: value to set the array index to
static inline
void gf16to32_vector_set(gf16to32 *v,
                         const uint32_t i,
                         const gf16to32 a) {
    for (uint32_t t = 0; t < 16; t++ ) {
        v[i][t] = a[t];
    }
}

/// \param v[in]: vector to print
/// \param n[in]: length of the vector
static inline
void gf16to32_vector_print(const gf16to32 *v,
                           const uint32_t n) {
    gf16to32 t;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_vector_get(t, v, i);
        printf("%llx ", (unsigned long long)t);
    }
    printf("\n");
}

/// \param n size of the vector (number of elements)
/// \return vector with n elements
static inline
gf16to32* gf16to32_vector_alloc(const uint32_t n) {
    return (gf16to32 *)malloc(n*sizeof(gf16to32));
}

/// \param v[out]: vector to set to zero
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_zero(gf16to32 *v,
                          const uint32_t n) {
    memset(v, 0, n * sizeof(gf16to32));
}

/// \param v[out]: random vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_random(gf16to32 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t t = 0; t < 16; t++) {
            v[i][t] = rand();
        }
    }
}

/// \param v1[out]: out vector
/// \param v2[in]: in vector
/// \param n[in]: length of the vector (number of elements)
static inline void gf16to32_vector_copy(gf16to32 *__restrict__ v1,
                                        const gf16to32 *__restrict__ v2,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t t = 0; t < 16; t++) {
            v1[i][t] = v2[i][t];
        }
    }
}

/// \param out += in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline void gf16to32_vector_add(gf16to32 *out,
                                       const gf16to32 *in,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_add(out[i], out[i], in[i]);
    }
}


/// \param out[in/out] += in
/// \param in[in]
/// \param n[in]: length of the vector (number of elements)
static inline void gf16to32_vector_add_gf16(gf16to32 *__restrict__ out,
                                            const gf16 *__restrict__ in,
                                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 a = gf16_vector_get(in, i);
        for (uint32_t t = 0; t < 16; t++) {
            out[i][t] ^= a;
        }
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_scalar_add(gf16to32 *out,
                                const gf16to32 c,
                                const gf16to32 *in,
                                const uint32_t n) {
    gf16to32 r;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul(r, c, in[i]);
        gf16to32_add(out[i], out[i], r);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16(gf16to32 *__restrict__ out,
                                    const gf16 c,
                                    const gf16to32 *__restrict__ in,
                                    const uint32_t n) {
    gf16to32 r;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul_gf16(r, in[i], c);
        gf16to32_add(out[i], out[i], r);
    }
}

/// \param out += t*in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16_v3(gf16to32 *__restrict__ out,
                                       const gf16to32 t,
                                       const gf16 *__restrict__ in,
                                       const uint32_t n) {
    gf16to32 r;
    for (uint32_t i = 0; i < n; i++) {
        const gf16 a = gf16_vector_get(in, i);
        gf16to32_mul_gf16(r, t, a);
        gf16to32_add(out[i], out[i], r);
    }
}

/// out = in1 + a*in2
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_scalar_add_v2(gf16to32 *__restrict__ out,
                                   const gf16to32 *__restrict__ in1,
                                   const gf16to32 a,
                                   const gf16to32 *__restrict__ in2,
                                   const uint32_t n) {
    gf16to32 r;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul(r, a, in2[i]);
        gf16to32_add(out[i], in1[i], r);
    }
}

/// \param a
/// \param b
/// \param n[in]: length of the vector (number of elements)
/// \return sum(a[i] * b[i])
static inline
void gf16to32_vector_mul_acc(gf16to32 r,
                             const gf16to32 *a,
                             const gf16to32 *b,
                             const uint32_t n) {
    gf16to32 t;
    gf16to32_set_zero(r);
    for (uint32_t i = 0; i < n; ++i) {
        gf16to32_mul(t, a[i], b[i]);
        gf16to32_add(r, t, t);
    }
}


#ifdef USE_AVX2 

/// \param out[out]: vector of length n
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf16to32_vector_add_u128(gf16to32 *out,
                              const gf16to32 *in,
                              const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(out + i));
        const __m128i t3 = gf16to32_add_u128(t1, t2);
        _mm_storeu_si128((__m128i *)out, t3);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_u128_v2(gf16to32 *__restrict__ out,
                                 const gf16to32 *__restrict__ in1,
                                 const gf16to32 *__restrict__ in2,
                                 const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(in2 + i));
        const __m128i t3 = gf16to32_add_u128(t1, t2);
        _mm_storeu_si128((__m128i *)out, t3);
    }
}

/// \param out += in
/// \param in
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_gf16_u128(gf16to32 *__restrict__ out,
                                   const gf16 *__restrict__ in,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 b = gf16_vector_get(in, i);
        const __m128i a128 = _mm_loadu_si128((const __m128i *)(out + i));
        const __m128i b128 = _mm_set1_epi8(b ^ (b << 4));
        const __m128i c128 = _mm_xor_si128(a128, b128);
        _mm_storeu_si128((__m128i *)(out+i), c128);
    }
}

/// \param out = in1 + in2
/// \param in1[in]:
/// \param in2[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf16to32_vector_add_gf16_u256_v2(gf16to32 *__restrict__ out,
                                     const gf16to32 *__restrict__ in1,
                                     const gf16 *__restrict__ in2,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 b = gf16_vector_get(in2, i);
        const __m128i a128 = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i b128 = _mm_set1_epi8(b ^ (b << 4));
        const __m128i c128 = _mm_xor_si128(a128, b128);
        _mm_storeu_si128((__m128i *)(out + i), c128);
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param d[in]: length of the vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u256_v2(gf16to32 *__restrict__ out,
                                        const gf16to32 *__restrict__ in1,
                                        const gf16to32 a,
                                        const gf16to32 *__restrict__ in2,
                                        const uint32_t n) {
    gf16to32 t;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul_v2(t, a, in2[i]);
        gf16to32_add(out[i], in1[i], t);
    }
}

/// poly1 = scalar * poly2
/// \param poly1[out]
/// \param scalar[in]
/// \param poly2[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u256(gf16to32 *poly1,
                                     const gf16to32 scalar,
                                     const gf16to32 *poly2,
                                     const uint32_t n) {
    gf16to32 t;
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul_v2(poly1[i], scalar, poly2[i]);
    }
}

/// out = scalar * in
/// \param out[out]: over gf16to32
/// \param scalar[in]: over gf16to32
/// \param in[in]: over gf16
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16_u256_v2(gf16to32 *out,
                                            const gf16to32 scalar,
                                            const gf16 *in,
                                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul_gf16(out[i], scalar, in[i]);
    }
}

/// out = scalar * in
/// \param out[out]: over gf16to32
/// \param scalar[in]: over gf16
/// \param in[in]: over gf16to32
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16_u256(gf16to32 *__restrict__ out,
                                          const gf16 scalar,
                                          const gf16to32 *__restrict__ in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf16to32_mul_gf16(out[i], in[i], scalar);
    }
}
#endif

