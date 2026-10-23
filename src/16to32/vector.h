#pragma once

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"
#include "../16/vector.h"

/// \param r[out]: = v[i]
/// \param v[in]: vector
/// \param i[in]: position to return
static inline
void gf16to32_vector_get(gf16to32 r,
                         const gf16to32 *v,
                         const uint32_t i) {
    gf16to32_copy(r, v[i]);
}

/// \param v[out]: v[i] = a
/// \param i[in]: index position
/// \param a[in]: value to set the array index to
static inline
void gf16to32_vector_set(gf16to32 *v,
                         const uint32_t i,
                         const gf16to32 a) {
    gf16to32_copy(v[i], a);
}

/// \param v[in]: vector to print
/// \param n[in]: length of the vector
static inline
void gf16to32_vector_print(const gf16to32 *v,
                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (int32_t t = 15; t >= 0; t--) {
            printf("%02x", v[i][t]);
        }
        printf(" ");
    }
    printf("\n");
}

/// \param n size of the vector (number of elements)
/// \return zero initialized vector with n elements
static inline
gf16to32* gf16to32_vector_alloc(const uint32_t n) {
    return (gf16to32 *)calloc(n, sizeof(gf16to32));
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
        gf16to32_set_random(v[i]);
    }
}

/// \param v1[out]: out vector
/// \param v2[in]: in vector
/// \param n[in]: length of the vector (number of elements)
static inline void gf16to32_vector_copy(gf16to32 *__restrict__ v1,
                                        const gf16to32 *__restrict__ v2,
                                        const uint32_t n) {
    memcpy(v1, v2, n * sizeof(gf16to32));
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
/// \param in[in]: vector over gf16 (nibble packed)
/// \param n[in]: length of the vector (number of elements)
static inline void gf16to32_vector_add_gf16(gf16to32 *__restrict__ out,
                                            const gf16 *__restrict__ in,
                                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i][0] ^= gf16_vector_get(in, i);
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
/// \param c: over gf16
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
/// \param in: vector over gf16 (nibble packed)
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

/// \param r[out]: = sum(a[i] * b[i])
/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_mul_acc(gf16to32 r,
                             const gf16to32 *a,
                             const gf16to32 *b,
                             const uint32_t n) {
    gf16to32 t, acc;
    gf16to32_set_zero(acc);
    for (uint32_t i = 0; i < n; ++i) {
        gf16to32_mul(t, a[i], b[i]);
        gf16to32_add(acc, acc, t);
    }
    gf16to32_copy(r, acc);
}


#ifdef USE_AVX2

/// out += in
/// \param out[in/out]: vector of length n
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf16to32_vector_add_u256(gf16to32 *out,
                              const gf16to32 *in,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf16to32v_add_u256(t1, t2));
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(out + i));
        _mm_storeu_si128((__m128i *)(out + i), gf16to32v_add_u128(t1, t2));
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_u256_v2(gf16to32 *out,
                                 const gf16to32 *in1,
                                 const gf16to32 *in2,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf16to32v_add_u256(t1, t2));
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(in2 + i));
        _mm_storeu_si128((__m128i *)(out + i), gf16to32v_add_u128(t1, t2));
    }
}

/// out += scalar * in
/// \param out[in/out]
/// \param scalar[in]
/// \param in[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u256(gf16to32 *out,
                                     const gf16to32 scalar,
                                     const gf16to32 *in,
                                     const uint32_t n) {
    const __m128i s128 = _mm_loadu_si128((const __m128i *)scalar);
    const __m256i s = _mm256_broadcastsi128_si256(s128);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), t2 ^ gf16to32v_mul_u256(s, t1));
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(out + i));
        _mm_storeu_si128((__m128i *)(out + i), t2 ^ gf16to32v_mul_u128(s128, t1));
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param n[in]: length of the vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u256_v2(gf16to32 *out,
                                        const gf16to32 *in1,
                                        const gf16to32 a,
                                        const gf16to32 *in2,
                                        const uint32_t n) {
    const __m128i s128 = _mm_loadu_si128((const __m128i *)a);
    const __m256i s = _mm256_broadcastsi128_si256(s128);
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), t1 ^ gf16to32v_mul_u256(s, t2));
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(in2 + i));
        _mm_storeu_si128((__m128i *)(out + i), t1 ^ gf16to32v_mul_u128(s128, t2));
    }
}

/// \param r[out]: = sum(a[i] * b[i])
/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_mul_acc_u256(gf16to32 r,
                                  const gf16to32 *a,
                                  const gf16to32 *b,
                                  const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    __m128i acc128 = _mm_setzero_si128();
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf16to32v_mul_u256(t1, t2);
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(a + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(b + i));
        acc128 = gf16to32v_mul_u128(t1, t2);
    }

    acc128 ^= _mm256_castsi256_si128(acc) ^ _mm256_extracti128_si256(acc, 1);
    _mm_storeu_si128((__m128i *)r, acc128);
}

/// out += in
/// \param out[in/out]:
/// \param in[in]: vector over gf16 (nibble packed)
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_gf16_u128(gf16to32 *__restrict__ out,
                                   const gf16 *__restrict__ in,
                                   const uint32_t n) {
    gf16to32_vector_add_gf16(out, in, n);
}

/// \param out = in1 + in2
/// \param in1[in]:
/// \param in2[in]: vector over gf16 (nibble packed)
/// \param n[in]: number of elements NOT bytes
static inline
void gf16to32_vector_add_gf16_u256_v2(gf16to32 *__restrict__ out,
                                      const gf16to32 *__restrict__ in1,
                                      const gf16 *__restrict__ in2,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const __m128i a128 = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i b128 = _mm_cvtsi32_si128(gf16_vector_get(in2, i));
        _mm_storeu_si128((__m128i *)(out + i), _mm_xor_si128(a128, b128));
    }
}

/// out += scalar * in
/// \param out[in/out]: over gf16to32
/// \param scalar[in]: over gf16to32
/// \param in[in]: over gf16 (nibble packed)
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16_u256_v2(gf16to32 *out,
                                             const gf16to32 scalar,
                                             const gf16 *in,
                                             const uint32_t n) {
    const __m128i s = _mm_loadu_si128((const __m128i *)scalar);
    for (uint32_t i = 0; i < n; i++) {
        const gf16 c = gf16_vector_get(in, i);
        const __m128i cc = _mm_set1_epi8((char)(c * 0x11));
        const __m128i o = _mm_loadu_si128((const __m128i *)(out + i));
        _mm_storeu_si128((__m128i *)(out + i), o ^ gf16to32v_gf16_mul_u128(s, cc));
    }
}

/// out += scalar * in
/// \param out[in/out]: over gf16to32
/// \param scalar[in]: over gf16
/// \param in[in]: over gf16to32
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_gf16_u256(gf16to32 *__restrict__ out,
                                          const gf16 scalar,
                                          const gf16to32 *__restrict__ in,
                                          const uint32_t n) {
    const __m256i s = _mm256_set1_epi8((char)((scalar & 0xF) * 0x11));
    uint32_t i = 0;
    for (; (i + 2) <= n; i += 2) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), t2 ^ gf16to32v_gf16_mul_u256(t1, s));
    }

    if (i < n) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)(in + i));
        const __m128i t2 = _mm_loadu_si128((const __m128i *)(out + i));
        _mm_storeu_si128((__m128i *)(out + i),
                         t2 ^ gf16to32v_gf16_mul_u128(t1, _mm256_castsi256_si128(s)));
    }
}
#endif // USE_AVX2

#ifdef USE_AVX512
/// \return mask of the 64-bit limbs of the first `nr_elements` (< 4) elements
static inline
__mmask8 gf16to32v_tail_mask_u512(const uint32_t nr_elements) {
    return (__mmask8)((1u << (2u * nr_elements)) - 1u);
}

/// out += in
/// \param out[in/out]: vector of length n
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf16to32_vector_add_u512(gf16to32 *out,
                              const gf16to32 *in,
                              const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(in + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), gf16to32v_add_u512(t1, t2));
    }

    if (i < n) {
        const __mmask8 m = gf16to32v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(m, (const void *)(in + i));
        const __m512i t2 = _mm512_maskz_loadu_epi64(m, (const void *)(out + i));
        _mm512_mask_storeu_epi64((void *)(out + i), m, gf16to32v_add_u512(t1, t2));
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_u512_v2(gf16to32 *out,
                                 const gf16to32 *in1,
                                 const gf16to32 *in2,
                                 const uint32_t n) {
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf16to32v_add_u512(t1, t2));
    }

    if (i < n) {
        const __mmask8 m = gf16to32v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(m, (const void *)(in1 + i));
        const __m512i t2 = _mm512_maskz_loadu_epi64(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi64((void *)(out + i), m, gf16to32v_add_u512(t1, t2));
    }
}

/// out += scalar * in
/// \param out[in/out]
/// \param scalar[in]
/// \param in[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u512(gf16to32 *out,
                                     const gf16to32 scalar,
                                     const gf16to32 *in,
                                     const uint32_t n) {
    const __m512i s = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)scalar));
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(in + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), t2 ^ gf16to32v_mul_u512(s, t1));
    }

    if (i < n) {
        const __mmask8 m = gf16to32v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(m, (const void *)(in + i));
        const __m512i t2 = _mm512_maskz_loadu_epi64(m, (const void *)(out + i));
        _mm512_mask_storeu_epi64((void *)(out + i), m, t2 ^ gf16to32v_mul_u512(s, t1));
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param n[in]: length of the vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u512_v2(gf16to32 *out,
                                        const gf16to32 *in1,
                                        const gf16to32 a,
                                        const gf16to32 *in2,
                                        const uint32_t n) {
    const __m512i s = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)a));
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), t1 ^ gf16to32v_mul_u512(s, t2));
    }

    if (i < n) {
        const __mmask8 m = gf16to32v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(m, (const void *)(in1 + i));
        const __m512i t2 = _mm512_maskz_loadu_epi64(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi64((void *)(out + i), m, t1 ^ gf16to32v_mul_u512(s, t2));
    }
}

/// \param r[out]: = sum(a[i] * b[i])
/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_mul_acc_u512(gf16to32 r,
                                  const gf16to32 *a,
                                  const gf16to32 *b,
                                  const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i + 4) <= n; i += 4) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc ^= gf16to32v_mul_u512(t1, t2);
    }

    if (i < n) {
        const __mmask8 m = gf16to32v_tail_mask_u512(n - i);
        const __m512i t1 = _mm512_maskz_loadu_epi64(m, (const void *)(a + i));
        const __m512i t2 = _mm512_maskz_loadu_epi64(m, (const void *)(b + i));
        acc ^= gf16to32v_mul_u512(t1, t2);
    }

    uint8_t tmp[64];
    _mm512_storeu_si512((void *)tmp, acc);
    for (uint32_t t = 0; t < 16; t++) {
        r[t] = tmp[t] ^ tmp[16 + t] ^ tmp[32 + t] ^ tmp[48 + t];
    }
}
#endif // USE_AVX512

#ifdef USE_NEON

/// out += in
/// \param out[in/out]: vector of length n
/// \param in[in]: vector of length n
/// \param n[in]: size of the vector, number of elements
static inline
void gf16to32_vector_add_u128(gf16to32 *out,
                              const gf16to32 *in,
                              const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        vst1q_u8(out[i], gf16to32v_add_u128(vld1q_u8(out[i]), vld1q_u8(in[i])));
    }
}

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param n[in] number of elements NOT bytes
static inline
void gf16to32_vector_add_u128_v2(gf16to32 *out,
                                 const gf16to32 *in1,
                                 const gf16to32 *in2,
                                 const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        vst1q_u8(out[i], gf16to32v_add_u128(vld1q_u8(in1[i]), vld1q_u8(in2[i])));
    }
}

/// out += scalar * in
/// \param out[in/out]
/// \param scalar[in]
/// \param in[in]
/// \param n[in]: length of vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u128(gf16to32 *out,
                                     const gf16to32 scalar,
                                     const gf16to32 *in,
                                     const uint32_t n) {
    const uint8x16_t s = vld1q_u8(scalar);
    for (uint32_t i = 0; i < n; i++) {
        const uint8x16_t t = gf16to32v_mul_u128(s, vld1q_u8(in[i]));
        vst1q_u8(out[i], gf16to32v_add_u128(vld1q_u8(out[i]), t));
    }
}

/// out = in1 + a*in2
/// \param out[out]:
/// \param in1[in]:
/// \param a[in]:
/// \param in2[in]:
/// \param n[in]: length of the vectors (number of elements)
static inline
void gf16to32_vector_scalar_add_u128_v2(gf16to32 *out,
                                        const gf16to32 *in1,
                                        const gf16to32 a,
                                        const gf16to32 *in2,
                                        const uint32_t n) {
    const uint8x16_t s = vld1q_u8(a);
    for (uint32_t i = 0; i < n; i++) {
        const uint8x16_t t = gf16to32v_mul_u128(s, vld1q_u8(in2[i]));
        vst1q_u8(out[i], gf16to32v_add_u128(vld1q_u8(in1[i]), t));
    }
}

/// \param r[out]: = sum(a[i] * b[i])
/// \param a[in]: first input vector
/// \param b[in]: second input vector
/// \param n[in]: length of the vector (number of elements)
static inline
void gf16to32_vector_mul_acc_u128(gf16to32 r,
                                  const gf16to32 *a,
                                  const gf16to32 *b,
                                  const uint32_t n) {
    uint8x16_t acc = vdupq_n_u8(0);
    for (uint32_t i = 0; i < n; i++) {
        acc = gf16to32v_add_u128(acc, gf16to32v_mul_u128(vld1q_u8(a[i]), vld1q_u8(b[i])));
    }
    vst1q_u8(r, acc);
}
#endif // USE_NEON
