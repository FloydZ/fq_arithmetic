#pragma once

#include <string.h>
#include <stdlib.h>

#include "arith.h"
#include "../2/vector.h"


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


/// \param out[in/out] += in
/// \param in[in]
/// \param n number of elements
static inline void gf2to128_vector_add_gf2(gf2to128 *__restrict__ out,
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
static inline void gf2to128_vector_scalar_add(gf2to128 *out,
                                             const gf2to128 c,
                                             const gf2to128 *in,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to128_mul(c, in[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf2to128_vector_scalar_add_gf2(gf2to128 *__restrict__ out,
                                                  const gf2 c,
                                                  const gf2to128 *__restrict__ in,
                                                  const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf2to128_mul_gf2(in[i], c);
    }
}


/// \param out += t*in
/// \param in
/// \param n length of the vectors
static inline void gf2to128_vector_scalar_add_gf2_v3(gf2to128 *__restrict__ out,
                                                    const gf2to128 t,
                                                    const gf2 *__restrict__ in,
                                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf2 a = gf2_vector_get(in, i);
        const gf2to128 b = gf2to128_mul_gf2(t, a);
        out[i] ^= b;
    }
}


/// out = in1 + a*in2
static inline void gf2to128_vector_scalar_add_v2(gf2to128 *__restrict__ out,
                                                    const gf2to128 *__restrict__ in1,
                                                    const gf2to128 a,
                                                    const gf2to128 *__restrict__ in2,
                                                    const uint32_t d) {
    for (uint32_t i = 0; i < d; i++) {
        out[i] = in1[i] ^ gf2to128_mul(a, in2[i]);
    }
}

/// \param a
/// \param b
/// \param n length of the vector
/// \return sum(a[i] * b[i])
static inline gf2to128 gf2to128_vector_mul_acc(const gf2to128 *a,
                                             const gf2to128 *b,
                                             const uint32_t n) {
    gf2to128 acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc ^= gf2to128_mul(a[i], b[i]);
    }

    return acc;
}


#ifdef USE_AVX2 

static inline void gf2to128_vector_add_u256(gf2to128 *out,
                                            const gf2to128 *in,
                                            const uint32_t n) {
    for (uint32_t i = 0; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        out[n-1] = gf2to128_add(out[n-1], in[n-1]);
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param d number of elements NOT bytes
static inline void gf2to128_vector_add_u256_v2(gf2to128 *__restrict__ out,
                                              const gf2to128 *__restrict__ in1,
                                              const gf2to128 *__restrict__ in2,
                                              const uint32_t n) {
    for (uint32_t i = 0; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        out[n-1] = gf2to128_add(in1[n-1], in2[n-1]);
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf2to128_vector_add_gf2_u256(gf2to128 *__restrict__ out,
                                                const gf2 *__restrict__ in,
                                                const uint32_t n) {
    __m256i tmp[4];
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = *in;
        gf2to128v_expand_gf2_x8_u256(tmp, b);
        for (uint32_t j = 0; j < 4; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i + 2*j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + i + 2*j), tmp[j]);
        }

        in += 1;
    }

    const uint8_t j = *in;
    for (; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(out + i));
        const __m256i b = gf2to128v_expand_gf2_x2_u256((j >> i) & 3u);
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        const gf2to128 a = gf2to128v_expand_gf2_x1((j >> (i&3u)) & 1u);
        const gf2to128 b = out[i];
        const gf2to128 c = a ^ b;
        out[i] = c;
    }
}


/// \param out = in1 + in2
/// \return 
static inline void gf2to128_vector_add_gf2_u256_v2(gf2to128 *__restrict__ out,
                                                  const gf2to128 *__restrict__ in1,
                                                  const gf2 *__restrict__ in2,
                                                  const uint32_t n) {

    __m256i tmp[4];
    uint32_t i = 0;
    for (; (i+8) <= n; i += 8) {
        const uint8_t b = *in2;
        gf2to128v_expand_gf2_x8_u256(tmp, b);
        for (uint32_t j = 0; j < 4; j++) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i + 2*j));
            tmp[j] ^= a;
            _mm256_storeu_si256((__m256i *)(out + i + 2*j), tmp[j]);
        }

        in2 += 1;
    }

    const uint8_t j = *in2;
    for (; (i+2) <= n; i += 2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = gf2to128v_expand_gf2_x2_u256((j >> i) & 3u);
        const __m256i c = a ^ b;
        _mm256_storeu_si256((__m256i *)(out + i), c);
    }

    if (n & 1u) {
        const gf2to128 a = gf2to128v_expand_gf2_x1((j >> (i&3u)) & 1u);
        const gf2to128 b = in1[i];
        const gf2to128 c = a ^ b;
        out[i] = c;
    }
}


/// out = in1 + a*in2
static inline void gf2to128_vector_scalar_add_u256_v2(gf2to128 *__restrict__ out,
                                                    const gf2to128 *__restrict__ in1,
                                                    const gf2to128 a,
                                                    const gf2to128 *__restrict__ in2,
                                                    const uint32_t d) {
    const __m256i aa = _mm256_setr_m128i((__m128i)a, (__m128i)a);
    for (uint32_t i = 0; (i+2) <= d; i+=2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));

        const __m256i u = gf2to128v_mul_u256(aa, t2);
        const __m256i v = u ^ t1;
   
        _mm256_storeu_si256((__m256i *)(out + i), v);
    }

    if (d & 1u) {
        const gf2to128 t1 = gf2to128_mul(a, in2[d-1]);
        const gf2to128 t2 = in1[d-1] ^ t1;
        out[d-1] = t2;
    }
}

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
        const __m256i b = gf2to128v_mul_u256(s, a);
        _mm256_storeu_si256((__m256i *)(poly1 + i), b);
    }

    if (d % 2) {
        // NOTE: this uses also avx instructions
        poly1[i] = gf2to128_mul(scalar, poly2[i]);
    }
}



#endif
