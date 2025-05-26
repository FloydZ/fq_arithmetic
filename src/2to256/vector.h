#pragma once

#include <string.h>
#include <stdlib.h>

#include "arith.h"
#include "../2/vector.h"

/// \param r[out]:
/// \param v[in]: vector
/// \param i[in]: position to return
/// \return v[i]
static inline
void gf2to256_vector_get(gf2to256 r,
                         const gf2to256 *v,
                         const uint32_t i) {
    for (uint32_t t = 0; t < 4; t++ ) {
        r[t] = v[i][t];
    }
}

/// \param v[out]: v[i] = a
/// \param i[in]: index position
/// \param a[in]: value to set the array index to
static inline
void gf2to256_vector_set(gf2to256 *v,
                         const uint32_t i,
                         const gf2to256 a) {
    for (uint32_t t = 0; t < 4; t++ ) {
        v[i][t] = a[t];
    }
}

/// \param v[in]: vector to print
/// \param n[in]: length of the vector
static inline
void gf2to256_vector_print(const gf2to256 *v,
                           const uint32_t n) {
    gf2to256 t;
    for (uint32_t i = 0; i < n; i++) {
        gf2to256_vector_get(t, v, i);
        printf("%llx ", (unsigned long long)t);
    }
    printf("\n");
}

/// \param n size of the vector (number of elements)
/// \return vector with n elements
static inline
gf2to256* gf2to256_vector_alloc(const uint32_t n) {
    return (gf2to256 *)malloc(n*sizeof(gf2to256));
}

/// \param v[out]: vector to set to zero
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_zero(gf2to256 *v,
                          const uint32_t n) {
    memset(v, 0, n * sizeof(gf2to256));
}

/// \param v[out]: random vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_random(gf2to256 *v,
                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t t = 0; t < 4; t++) {
            v[i][t] = rand() ^ ((uint64_t)rand() << 32);
        }
    }
}

/// \param v1[out]: out vector
/// \param v2[in]: in vector
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to256_vector_copy(gf2to256 *__restrict__ v1,
                                        const gf2to256 *__restrict__ v2,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t t = 0; t < 4; t++) {
            v1[i][t] = v2[i][t];
        }
    }
}

/// \param out += in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to256_vector_add(gf2to256 *out,
                                       const gf2to256 *in,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        gf2to256_add(out[i], out[i], in[i]);
    }
}


/// \param out[in/out] += in
/// \param in[in]
/// \param n[in]: length of the vector (number of elements)
static inline void gf2to256_vector_add_gf2(gf2to256 *__restrict__ out,
                                          const gf2 *__restrict__ in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        for (uint32_t t = 0; t < 4; t++) {
            out[i][t] ^= a;
        }
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_scalar_add(gf2to256 *out,
                                const gf2to256 c,
                                const gf2to256 *in,
                                const uint32_t n) {
    gf2to256 r;
    for (uint32_t i = 0; i < n; i++) {
        gf2to256_mul(r, c, in[i]);
        gf2to256_add(out[i], out[i], r);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_scalar_add_gf2(gf2to256 *__restrict__ out,
                                    const gf2 c,
                                    const gf2to256 *__restrict__ in,
                                    const uint32_t n) {
    gf2to256 r;
    for (uint32_t i = 0; i < n; i++) {
        gf2to256_mul_gf2(r, in[i], c);
        gf2to256_add(out[i], out[i], r);
    }
}

/// \param out += t*in
/// \param in
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_scalar_add_gf2_v3(gf2to256 *__restrict__ out,
                                       const gf2to256 t,
                                       const gf2 *__restrict__ in,
                                       const uint32_t n) {
    gf2to256 r;
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        gf2to256_mul_gf2(r, t, a);
        gf2to256_add(out[i], out[i], r);
    }
}

/// out = in1 + a*in2
/// \param n[in]: length of the vector (number of elements)
static inline
void gf2to256_vector_scalar_add_v2(gf2to256 *__restrict__ out,
                                   const gf2to256 *__restrict__ in1,
                                   const gf2to256 a,
                                   const gf2to256 *__restrict__ in2,
                                   const uint32_t n) {
    gf2to256 r;
    for (uint32_t i = 0; i < n; i++) {
        gf2to256_mul(r, a, in2[i]);
        gf2to256_add(out[i], in1[i], r);
    }
}

/// \param a
/// \param b
/// \param n[in]: length of the vector (number of elements)
/// \return sum(a[i] * b[i])
static inline
void gf2to256_vector_mul_acc(gf2to256 r,
                             const gf2to256 *a,
                             const gf2to256 *b,
                             const uint32_t n) {
    gf2to256 t;
    gf2to256_set_zero(r);
    for (uint32_t i = 0; i < n; ++i) {
        gf2to256_mul(t, a[i], b[i]);
        gf2to256_add(r, t, t);
    }
}


#ifdef USE_AVX2 

/// \param out[out]: vector of length n
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf2to256_vector_add_u256(gf2to256 *out,
                              const gf2to256 *in,
                              const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i t3 = gf2to256v_add_u256(t1, t2);
        _mm256_storeu_si256((__m256i *)out, t3);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf2to256_vector_add_u256_v2(gf2to256 *__restrict__ out,
                                 const gf2to256 *__restrict__ in1,
                                 const gf2to256 *__restrict__ in2,
                                 const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));
        const __m256i t3 = gf2to256v_add_u256(t1, t2);
        _mm256_storeu_si256((__m256i *)out, t3);
    }
}

/// \param out += in
/// \param in
/// \param n[in] number of elements NOT bytes
static inline
void gf2to256_vector_add_gf2_u256(gf2to256 *__restrict__ out,
                                  const gf2 *__restrict__ in,
                                  const uint32_t n) {
    __m256i tmp[8];
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = *in;
        gf2to256v_expand_gf2_x8_u256(tmp, b);
        for (uint32_t j = 0; j < 8; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(out + j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + j), tmp[j]);
        }

        in += 1;
        out += 8;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        gf2to256v_expand_gf2_x8_u256(tmp, *in);
        for (uint32_t j = 0; j < limit; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(out + j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + j), tmp[j]);
        }
    }
}

/// \param out = in1 + in2
/// \param in1[in]:
/// \param in2[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf2to256_vector_add_gf2_u256_v2(gf2to256 *__restrict__ out,
                                     const gf2to256 *__restrict__ in1,
                                     const gf2 *__restrict__ in2,
                                     const uint32_t n) {

    __m256i tmp[8];
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = *in2;
        gf2to256v_expand_gf2_x8_u256(tmp, b);
        for (uint32_t j = 0; j < 8; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i + j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + i + j), tmp[j]);
        }

        in2 += 1;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        gf2to256v_expand_gf2_x8_u256(tmp, *in2);
        for (uint32_t j = 0; j < limit; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(out + j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + j), tmp[j]);
        }
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param d[in]: length of the vectors (number of elements)
static inline
void gf2to256_vector_scalar_add_u256_v2(gf2to256 *__restrict__ out,
                                        const gf2to256 *__restrict__ in1,
                                        const gf2to256 a,
                                        const gf2to256 *__restrict__ in2,
                                        const uint32_t n) {
    const __m256i aa = _mm256_loadu_si256((const __m256i *)a);
    for (uint32_t i = 0; i < n; i++) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));

        const __m256i u = gf2to256v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;
   
        _mm256_storeu_si256((__m256i *)(out + i), v);
    }
}

/// poly1 = scalar * poly2
/// \param poly1[out]
/// \param scalar[in]
/// \param poly2[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to256_vector_scalar_add_u256(gf2to256 *poly1,
                                     const gf2to256 scalar,
                                     const gf2to256 *poly2,
                                     const uint32_t n) {
    const __m256i s = _mm256_loadu_si256((const __m256i *)scalar);
    for (uint32_t i = 0; i < n; i++) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(poly2 + i));

        const __m256i u = gf2to256v_mul_u256(s, t1);
        _mm256_storeu_si256((__m256i *)(poly1 + i), u);
    }
}

/// out = scalar * in
/// \param out[out]: over gf2to256
/// \param scalar[in]: over gf2to256
/// \param in[in]: over gf2
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to256_vector_scalar_add_gf2_u256_v2(gf2to256 *out,
                                            const gf2to256 scalar,
                                            const gf2 *in,
                                            const uint32_t n) {
    uint32_t i = 0;
    __m256i buffer[4];
    const __m256i s = _mm256_loadu_si256((const __m256i *)scalar);
    for (; (i+8) <= n; i+=8) {
        gf2to256v_expand_gf2_x8_u256(buffer, *in);
        for (uint32_t t = 0; t < 8; t++) {
            const __m256i tmp = gf2to256v_mul_gf2_u256(s, buffer[t]);
            _mm256_storeu_si256((__m256i *)(out + t), tmp);
        }

        in += 1;
        out += 8;
    }

    const uint32_t limit = n % 8;
    if (limit) {
        gf2to256v_expand_gf2_x8_u256(buffer, *in);
        for (uint32_t t = 0; t < limit; t++) {
            const __m256i tmp = gf2to256v_mul_gf2_u256(s, buffer[t]);
            _mm256_storeu_si256((__m256i *)(out + t), tmp);
        }
    }
}

/// out = scalar * in
/// \param out[out]: over gf2to256
/// \param scalar[in]: over gf2
/// \param in[in]: over gf2to256
/// \param n[in]: length of vectors (number of elements)
static inline
void gf2to256_vector_scalar_add_gf2_u256(gf2to256 *__restrict__ out,
                                         const gf2 scalar,
                                         const gf2to256 *__restrict__ in,
                                         const uint32_t n) {
    const __m256i s = _mm256_set1_epi8(-scalar);
    uint32_t i = 0;
    for (; i < n; i++) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in);
        const __m256i t2 = t1 & s;
        _mm256_storeu_si256((__m256i *)(out), t2);

        in += 2;
        out += 2;
    }
}

#endif