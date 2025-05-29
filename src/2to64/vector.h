#pragma once

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "arith.h"
#include "../2/vector.h"


/// \return v[i]
static inline
gf2to64 gf2to64_vector_get(const gf2to64 *v,
                           const uint32_t i) {
    gf2to64 b = v[i];
    return b;
}

/// \param v[out]: v[i] = a
/// \param i[in]: index position
/// \param a[in]: value to set the array index to
static inline
void gf2to64_vector_set(gf2to64 *v,
                         const uint32_t i,
                         const gf2to64 a) {
    v[i] = a;
}

/// \param v[in]: vector to print
/// \param n[in]: length of the vector
static inline
void gf2to64_vector_print(const gf2to64 *v,
                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2to64 t = gf2to64_vector_get(v, i);
        printf("%llx ", (unsigned long long)t);
    }
    printf("\n");
}

/// \param n size of the vector (number of elements)
/// \return vector with n elements
static inline
gf2to64* gf2to64_vector_alloc(const uint32_t n) {
    return (gf2to64 *)malloc(n*sizeof(gf2to64));
}

/// \param v[out]: vector to set to zero
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_zero(gf2to64 *v,
                          const uint32_t n) {
    memset(v, 0, n * sizeof(gf2to64));
}

/// \param v[out]: random vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_random(gf2to64 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand();
    }
}

/// \param v1[out]: out vector
/// \param v2[in]: in vector
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_copy(gf2to64 *__restrict__ v1,
                                        const gf2to64 *__restrict__ v2,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_add(gf2to64 *out,
                                       const gf2to64 *in,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}


/// \param out[in/out] += in
/// \param in[in]
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to64_vector_add_gf2(gf2to64 *__restrict__ out,
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
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add(gf2to64 *out,
                                const gf2to64 c,
                                const gf2to64 *in,
                                const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to64_mul(c, in[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2(gf2to64 *__restrict__ out,
                                    const gf2 c,
                                    const gf2to64 *__restrict__ in,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to64_mul_gf2(in[i], c);
    }
}

/// \param out += t*in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_v2(gf2to64 *__restrict__ out,
                                       const gf2to64 t,
                                       const gf2 *__restrict__ in,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        const gf2to64 b = gf2to64_mul_gf2(t, a);
        out[i] ^= b;
    }
}

/// out = in1 + a*in2
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to64_vector_scalar_add_v2(gf2to64 *__restrict__ out,
                                  const gf2to64 *__restrict__ in1,
                                  const gf2to64 a,
                                  const gf2to64 *__restrict__ in2,
                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf2to64_mul(a, in2[i]);
    }
}

/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
/// \return sum(a[i] * b[i])
static inline
gf2to64 gf2to64_vector_mul_acc(const gf2to64 *a,
                               const gf2to64 *b,
                               const uint32_t n) {
    gf2to64 acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc ^= gf2to64_mul(a[i], b[i]);
    }

    return acc;
}


#ifdef USE_AVX2 

/// \param out[out]: vector of length n: = += in
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf2to64_vector_add_u256(gf2to64 *out,
                             const gf2to64 *in,
                             const uint32_t n) {
    for (uint32_t i = 0; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        out[n-1] = gf2to64_add(out[n-1], in[n-1]);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf2to64_vector_add_u256_v2(gf2to64 *__restrict__ out,
                                const gf2to64 *__restrict__ in1,
                                const gf2to64 *__restrict__ in2,
                                const uint32_t n) {
    for (uint32_t i = 0; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        out[n-1] = gf2to64_add(in1[n-1], in2[n-1]);
    }
}

/// \param out += in
/// \param in
/// \param n[in] number of elements NOT bytes
static inline
void gf2to64_vector_add_gf2_u256(gf2to64 *__restrict__ out,
                                 const gf2 *__restrict__ in,
                                 const uint32_t n) {
    const uint32_t limit = n % 8;
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = *in;
        __m256i tmp = gf2to64v_expand_gf2_x8_u256(b);
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out));
        tmp = tmp ^ a;
        _mm256_storeu_si256((__m256i *)(out), tmp);

        in += 1;
        out += 8;
    }

    if (limit) {
        uint32_t tmp2[8] __attribute__((aligned(32))) = {0};
        __m256i tmp = gf2to64v_expand_gf2_x8_u256(*in);
        for (uint32_t j = 0; j < limit; j++) { tmp2[j] = out[j]; }
        const __m256i a = _mm256_loadu_si256((const __m256i *)tmp2);
        tmp = tmp ^ a;
        _mm256_storeu_si256((__m256i *)tmp2, tmp);
        for (uint32_t j = 0; j < limit; j++) { out[j] = tmp2[j]; }
    }
}


/// \param out = in1 + in2
/// \param in1[in]:
/// \param in2[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf2to64_vector_add_gf2_u256_v2(gf2to64 *__restrict__ out,
                                     const gf2to64 *__restrict__ in1,
                                     const gf2 *__restrict__ in2,
                                     const uint32_t n) {
    const uint32_t limit = n % 8;
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        __m256i tmp = gf2to64v_expand_gf2_x8_u256(*in2);
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1));
        tmp = tmp ^ a;
        _mm256_storeu_si256((__m256i *)(out), tmp);

        in2 += 1;
        in1 += 8;
        out += 8;
    }

    if (limit) {
        uint32_t tmp2[8] __attribute__((aligned(32))) = {0};
        __m256i tmp = gf2to64v_expand_gf2_x8_u256(*in2);
        for (uint32_t j = 0; j < limit; j++) { tmp2[j] = in1[j]; }
        const __m256i a = _mm256_loadu_si256((const __m256i *)tmp2);
        tmp = tmp ^ a;
        _mm256_storeu_si256((__m256i *)tmp2, tmp);
        for (uint32_t j = 0; j < limit; j++) { out[j] = tmp2[j]; }
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param d[in]: length of the vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_u256_v2(gf2to64 *__restrict__ out,
                                        const gf2to64 *__restrict__ in1,
                                        const gf2to64 a,
                                        const gf2to64 *__restrict__ in2,
                                        const uint32_t d) {
    const __m256i aa = _mm256_set1_epi32(a);
    for (uint32_t i = 0; (i+8) <= d; i+=8) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)in2);

        const __m256i u = gf2to64v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;

        _mm256_storeu_si256((__m256i *)out, v);
        in1 += 8;
        in2 += 8;
        out += 8;
    }

    const uint32_t limit = d % 8;
    if (limit) {
        uint32_t tmp1[8] __attribute__((aligned(32))) = {0};
        uint32_t tmp2[8] __attribute__((aligned(32))) = {0};
        for (uint32_t i = 0; i < limit; i++) { tmp1[i] = in1[i]; }
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(tmp1));
        for (uint32_t i = 0; i < limit; i++) { tmp2[i] = in2[i]; }
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(tmp2));

        const __m256i u = gf2to64v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;

        _mm256_storeu_si256((__m256i *)tmp1, v);
        for (uint32_t i = 0; i < limit; i++) { out[i] = tmp1[i]; }
    }
}

/// poly1 += scalar * poly2
/// \param poly1[out]
/// \param scalar[in]
/// \param poly2[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_u256(gf2to64 *poly1,
                                    const gf2to64 scalar,
                                    const gf2to64 *poly2,
                                    const uint32_t n) {
    const __m256i aa = _mm256_set1_epi32(scalar);
    for (uint32_t i = 0; (i+8) <= n; i+=8) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)poly1);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)poly2);
        const __m256i u = gf2to64v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;
        _mm256_storeu_si256((__m256i *)poly1, v);
        poly1 += 8;
        poly2 += 8;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        uint32_t tmp1[8] __attribute__((aligned(32))) = {0};
        uint32_t tmp2[8] __attribute__((aligned(32))) = {0};
        for (uint32_t i = 0; i < limit; i++) { tmp1[i] = poly1[i]; }
        const __m256i t1 = _mm256_load_si256((const __m256i *)tmp1);
        for (uint32_t i = 0; i < limit; i++) { tmp2[i] = poly2[i]; }
        const __m256i t2 = _mm256_load_si256((const __m256i *)tmp2);

        const __m256i u = gf2to64v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;
        _mm256_store_si256((__m256i *)tmp1, v);
        for (uint32_t i = 0; i < limit; i++) { poly1[i] = tmp1[i]; }
    }
}

/// out += scalar * in
/// \param out[out]: over gf2to64
/// \param scalar[in]: over gf2to64
/// \param in[in]: over gf2
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_u256_v2(gf2to64 *out,
                                           const gf2to64 scalar,
                                           const gf2 *in,
                                           const uint32_t n) {
    const __m256i s = (__m256i)_mm256_set1_epi32(scalar);
    for (uint32_t i = 0; (i+8) <= n; i+=8) {
        const __m256i t1 = gf2to64v_expand_gf2_x8_ext_u256(*in);
        const __m256i t2 = gf2to64v_mul_gf2_u256(s, t1);
        _mm256_storeu_si256((__m256i *)out, t2);

        in += 1;
        out += 8;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        uint32_t tmp[8] __attribute__((aligned(32))) = {0};
        const __m256i t1 = gf2to64v_expand_gf2_x8_ext_u256(*in);
        const __m256i t2 = gf2to64v_mul_gf2_u256(s, t1);
        _mm256_storeu_si256((__m256i *)tmp, t2);
        for (uint32_t i = 0; i < limit; i++) { out[i] ^= tmp[i]; }
    }
}

/// out += scalar * in
/// \param out[out]: over gf2to64
/// \param scalar[in]: over gf2
/// \param in[in]: over gf2to64
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to64_vector_scalar_add_gf2_u256(gf2to64 *__restrict__ out,
                                         const gf2 scalar,
                                         const gf2to64 *__restrict__ in,
                                         const uint32_t n) {
    const __m256i s = _mm256_set1_epi32(-scalar);
    for (uint32_t i = 0; (i+8) <= n; i+=8) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in);
        const __m256i t2 = gf2to64v_mul_gf2_u256(t1, s);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)out);
        const __m256i t4 = t2 ^ t3;
        _mm256_storeu_si256((__m256i *)out, t4);

        in += 8;
        out += 8;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        uint32_t tmp[8] __attribute__((aligned(32))) = {0};
        for (uint32_t i = 0; i < limit; i++) { tmp[i] ^= in[i]; }

        const __m256i t1 = _mm256_loadu_si256((const __m256i *)tmp);
        const __m256i t2 = gf2to64v_mul_gf2_u256(s, t1);

        _mm256_storeu_si256((__m256i *)tmp, t2);
        for (uint32_t i = 0; i < limit; i++) { out[i] ^= tmp[i]; }
    }
}

#endif