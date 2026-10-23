#pragma once

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "arith.h"

/// NOTE: vectors over GF(16) are stored packed: two elements per byte,
/// element 2i in the lower nibble of byte i, element 2i+1 in the upper nibble.
/// `n` always counts elements, not bytes.

/// \return v[i]
static inline
gf16 gf16_vector_get(const gf16 *v,
                     const uint32_t i) {
    const uint32_t pos = i >> 1;
    uint8_t b = 0;
    if (i & 1u) {
        b = v[pos] >> 4;
    } else {
        b = v[pos] & 0xF;
    }
    return b;
}

/// v[i] = a
static inline
void gf16_vector_set(gf16 *v,
                     const uint32_t i,
                     const gf16 a) {
    const uint32_t pos = i >> 1;
    const uint8_t b = a & 0xF;
    if (i & 1u) {
        v[pos] = (v[pos] & 0xF ) ^ (b << 4);
    } else {
        v[pos] = (v[pos] & 0xF0) ^ b;
    }
}

/// \param v[in]: vector to print
/// \param n[in]: number of elements
static inline void gf16_vector_print(const gf16 *v,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t = gf16_vector_get(v, i);
        printf("%2d ", (uint32_t )t);
    }
    printf("\n");
}

/// \param n[in]: number of elements
/// \return zero initialized vector of n elements
static inline gf16* gf16_vector_alloc(const uint32_t n) {
    return (gf16 *)calloc((n+1)/2, sizeof(gf16));
}

/// \param v[out]: = 0
/// \param n[in]: number of elements
static inline void gf16_vector_zero(gf16 *v,
                                    const uint32_t n) {
    memset(v, 0, ((n + 1)/2) *sizeof(gf16));
}

/// \param v[out]: random vector, the unused upper nibble of the last byte is zero
/// \param n[in]: number of elements
static inline void gf16_vector_random(gf16 *v,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n/2; i++) {
        v[i] = rand() & 0xFF;
    }
    if (n & 1u) {
        v[n/2] = rand() & 0x0F;
    }
}

/// \param v1[out]: = v2
/// \param v2[in]:
/// \param n[in]: number of elements
static inline void gf16_vector_copy(gf16 *__restrict__ v1,
                                    const gf16 *__restrict__ v2,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n/2; i++) {
        v1[i] = v2[i];
    }
    if (n & 1u) {
        v1[n/2] = (v1[n/2] & 0xF0) ^ (v2[n/2] & 0x0F);
    }
}

/// \param out[in/out]: += in
/// \param in[in]:
/// \param n[in]: number of elements
static inline void gf16_vector_add(gf16 *out,
                                   const gf16 *in,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < n/2; i++) {
        out[i] ^= in[i];
    }
    if (n & 1u) {
        out[n/2] ^= in[n/2] & 0x0F;
    }
}

/// \param out[in/out]: += c*in
/// \param c[in]: scalar
/// \param in[in]:
/// \param n[in]: number of elements
static inline void gf16_vector_scalar_add(gf16 *out,
                                          const gf16 c,
                                          const gf16 *in,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t = gf16_vector_get(out, i) ^ gf16_mul(c, gf16_vector_get(in, i));
        gf16_vector_set(out, i, t);
    }
}

/// \param out[out]: = in1 + c*in2
/// \param in1[in]:
/// \param c[in]: scalar
/// \param in2[in]:
/// \param n[in]: number of elements
static inline void gf16_vector_scalar_add_v2(gf16 *out,
                                             const gf16 *in1,
                                             const gf16 c,
                                             const gf16 *in2,
                                             const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t = gf16_vector_get(in1, i) ^ gf16_mul(c, gf16_vector_get(in2, i));
        gf16_vector_set(out, i, t);
    }
}

/// \param a[in]: first vector
/// \param b[in]: second vector
/// \param n[in]: number of elements
/// \return sum_i a_i*b_i
static inline gf16 gf16_vector_mul_acc(const gf16 *a,
                                       const gf16 *b,
                                       const uint32_t n) {
    gf16 acc = 0;
    for (uint32_t i = 0; i < n; i++) {
        acc ^= gf16_mul(gf16_vector_get(a, i), gf16_vector_get(b, i));
    }
    return acc;
}

/// internal helper: handles the bytes [start, n/2) and the odd nibble of
/// out = in1 + c*in2, where in1 == out is allowed.
static inline void gf16_vector_scalar_add_tail(gf16 *out,
                                               const gf16 *in1,
                                               const gf16 c,
                                               const gf16 *in2,
                                               const uint32_t start,
                                               const uint32_t n) {
    const uint8_t cc = (c & 0xF) ^ ((c & 0xF) << 4);
    for (uint32_t i = start; i < n/2; i++) {
        out[i] = in1[i] ^ gf16v_mul(in2[i], cc);
    }
    if (n & 1u) {
        const uint8_t t = (in1[n/2] ^ gf16_mul(c & 0xF, in2[n/2] & 0xF)) & 0x0F;
        out[n/2] = (out[n/2] & 0xF0) ^ t;
    }
}

/// internal helper: returns the xor of all elements in the bytes [start, n/2)
/// and the odd nibble of sum_i a_i*b_i
static inline gf16 gf16_vector_mul_acc_tail(const gf16 *a,
                                            const gf16 *b,
                                            const uint32_t start,
                                            const uint32_t n) {
    uint8_t acc = 0;
    for (uint32_t i = start; i < n/2; i++) {
        acc ^= gf16v_mul(a[i], b[i]);
    }
    acc = (acc & 0xF) ^ (acc >> 4);
    if (n & 1u) {
        acc ^= gf16_mul(a[n/2] & 0xF, b[n/2] & 0xF);
    }
    return acc;
}

#ifdef USE_AVX2

/// out = in1 + in2
/// \param out[out]:
/// \param in1[in]:
/// \param in2[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u256_v2(gf16 *out,
                             const gf16 *in1,
                             const gf16 *in2,
                             const uint32_t n) {
    const uint32_t bytes = n/2;
    uint32_t i = 0;
    for (; (i+32) <= bytes; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
        _mm256_storeu_si256((__m256i *)(out + i), gf16v_add_u256(a, b));
    }

    for (; (i+16) <= bytes; i += 16) {
        const __m128i a = _mm_loadu_si128((const __m128i *)(in1 + i));
        const __m128i b = _mm_loadu_si128((const __m128i *)(in2 + i));
        _mm_storeu_si128((__m128i *)(out + i), _mm_xor_si128(a, b));
    }

    for (; i < bytes; i++) {
        out[i] = in1[i] ^ in2[i];
    }

    if (n & 1u) {
        out[bytes] = (out[bytes] & 0xF0) ^ ((in1[bytes] ^ in2[bytes]) & 0x0F);
    }
}

/// out += in
/// \param out[in/out]:
/// \param in[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u256(gf16 *out,
                          const gf16 *in,
                          const uint32_t n) {
	gf16_vector_add_u256_v2(out, out, in, n);
}

/// d = number of elements not bytes
/// poly1 =  scalar * poly2
/// NOTE: processes ceil(d/2) bytes, i.e. the unused upper nibble of the last
///     byte is also written (scalar * upper nibble of the last byte of poly2)
static inline
void gf16_vector_multiple_u256(ff_t *poly1,
                               const ff_t scalar,
                               const ff_t *poly2,
                               const uint32_t d) {
	uint32_t i = (d+1)/2;
    const __m256i ml = gf16v_tbl32_multab(scalar);
    const __m256i mh = _mm256_slli_epi16(ml, 4);
    const __m256i xf = _mm256_set1_epi8(0xF);

	// avx2 code
	while (i >= 32u) {
        const __m256i p2 = _mm256_loadu_si256((__m256i *)poly2);
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p2, xf);
        _mm256_storeu_si256((__m256i *)poly1, t1);
		i -= 32u;
		poly1 += 32u;
		poly2 += 32u;
	}

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32))) = {0};
        for (uint32_t j = 0; j<i; j++) { tmp[j] = poly2[j]; }
        const __m256i p3 = _mm256_load_si256((__m256i *)tmp);
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p3, xf);
        _mm256_store_si256((__m256i *)tmp, t1);
        for (uint32_t j = 0; j<i; j++) { poly1[j] = tmp[j]; }
    }
}

/// out = in1 + c * in2
/// \param out[out]:
/// \param in1[in]:
/// \param c[in]: scalar
/// \param in2[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u256_v2(gf16 *out,
                                    const gf16 *in1,
                                    const gf16 c,
                                    const gf16 *in2,
                                    const uint32_t n) {
    const uint32_t bytes = n/2;
    const __m256i ml = gf16v_tbl32_multab(c & 0xF);
    const __m256i mh = _mm256_slli_epi16(ml, 4);
    const __m256i xf = _mm256_set1_epi8(0xF);

    uint32_t i = 0;
    for (; (i+32) <= bytes; i += 32) {
        const __m256i p2 = _mm256_loadu_si256((const __m256i *)(in2 + i));
        const __m256i p1 = _mm256_loadu_si256((const __m256i *)(in1 + i));
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p2, xf);
        _mm256_storeu_si256((__m256i *)(out + i), p1 ^ t1);
    }

    gf16_vector_scalar_add_tail(out, in1, c, in2, i, n);
}

/// out += c * in
/// \param out[in/out]:
/// \param c[in]: scalar
/// \param in[in]:
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u256(gf16 *out,
                                 const gf16 c,
                                 const gf16 *in,
                                 const uint32_t n) {
	gf16_vector_scalar_add_u256_v2(out, out, c, in, n);
}

/// \param a[in]:
/// \param b[in]:
/// \param n[in]: number of elements NOT bytes
/// \return sum_i a_i*b_i
static inline
gf16 gf16_vector_mul_acc_u256(const gf16 *a,
                              const gf16 *b,
                              const uint32_t n) {
    const uint32_t bytes = n/2;
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i+32) <= bytes; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf16v_mul_u256(t1, t2);
    }

    const uint64_t t = gf16_hadd_u256_64(acc);
    uint8_t r = 0;
    for (uint32_t k = 0; k < 8; k++) { r ^= (uint8_t)(t >> (8*k)); }
    r = (r & 0xF) ^ (r >> 4);
    return r ^ gf16_vector_mul_acc_tail(a, b, i, n);
}
#endif // USE_AVX2

#ifdef USE_NEON

/// out = in1 + in2
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u128_v2(gf16 *out,
                             const gf16 *in1,
                             const gf16 *in2,
                             const uint32_t n) {
    const uint32_t bytes = n/2;
    uint32_t i = 0;
    for (; (i+16) <= bytes; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, gf16v_add_u128(a, b));
    }

    for (; i < bytes; i++) {
        out[i] = in1[i] ^ in2[i];
    }

    if (n & 1u) {
        out[bytes] = (out[bytes] & 0xF0) ^ ((in1[bytes] ^ in2[bytes]) & 0x0F);
    }
}

/// out += in
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u128(gf16 *out,
                          const gf16 *in,
                          const uint32_t n) {
    gf16_vector_add_u128_v2(out, out, in, n);
}

/// out = in1 + c*in2
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u128_v2(gf16 *out,
                                    const gf16 *in1,
                                    const gf16 c,
                                    const gf16 *in2,
                                    const uint32_t n) {
    const uint32_t bytes = n/2;
    const uint8x16_t cc = vdupq_n_u8(c & 0xF);
    uint32_t i = 0;
    for (; (i+16) <= bytes; i += 16) {
        const uint8x16_t a = vld1q_u8(in1 + i);
        const uint8x16_t b = vld1q_u8(in2 + i);
        vst1q_u8(out + i, veorq_u8(a, gf16v_mul_lo_u128(b, cc)));
    }

    gf16_vector_scalar_add_tail(out, in1, c, in2, i, n);
}

/// out += c*in
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u128(gf16 *out,
                                 const gf16 c,
                                 const gf16 *in,
                                 const uint32_t n) {
    gf16_vector_scalar_add_u128_v2(out, out, c, in, n);
}

/// \param n[in]: number of elements NOT bytes
/// \return sum_i a_i*b_i
static inline
gf16 gf16_vector_mul_acc_u128(const gf16 *a,
                              const gf16 *b,
                              const uint32_t n) {
    const uint32_t bytes = n/2;
    uint8x16_t acc = vdupq_n_u8(0);
    uint32_t i = 0;
    for (; (i+16) <= bytes; i += 16) {
        const uint8x16_t t1 = vld1q_u8(a + i);
        const uint8x16_t t2 = vld1q_u8(b + i);
        acc = veorq_u8(acc, gf16v_mul_u128(t1, t2));
    }

    uint8_t tmp[16];
    vst1q_u8(tmp, acc);
    uint8_t r = 0;
    for (uint32_t k = 0; k < 16; k++) { r ^= tmp[k]; }
    r = (r & 0xF) ^ (r >> 4);
    return r ^ gf16_vector_mul_acc_tail(a, b, i, n);
}
#endif // USE_NEON

#ifdef USE_AVX512

/// out = in1 + in2
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u512_v2(gf16 *out,
                             const gf16 *in1,
                             const gf16 *in2,
                             const uint32_t n) {
    const uint32_t bytes = n/2;
    uint32_t i = 0;
    for (; (i+64) <= bytes; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf16v_add_u512(a, b));
    }

    if (i < bytes) {
        const __mmask64 m = (__mmask64)((1ull << (bytes - i)) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi8(m, (const void *)(in1 + i));
        const __m512i b = _mm512_maskz_loadu_epi8(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi8((void *)(out + i), m, gf16v_add_u512(a, b));
    }

    if (n & 1u) {
        out[bytes] = (out[bytes] & 0xF0) ^ ((in1[bytes] ^ in2[bytes]) & 0x0F);
    }
}

/// out += in
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_add_u512(gf16 *out,
                          const gf16 *in,
                          const uint32_t n) {
    gf16_vector_add_u512_v2(out, out, in, n);
}

/// out = in1 + c*in2
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u512_v2(gf16 *out,
                                    const gf16 *in1,
                                    const gf16 c,
                                    const gf16 *in2,
                                    const uint32_t n) {
    const uint32_t bytes = n/2;
    const __m512i cc = _mm512_set1_epi8((char)((c & 0xF) ^ ((c & 0xF) << 4)));
    uint32_t i = 0;
    for (; (i+64) <= bytes; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), a ^ gf16v_mul_u512(b, cc));
    }

    if (i < bytes) {
        const __mmask64 m = (__mmask64)((1ull << (bytes - i)) - 1ull);
        const __m512i a = _mm512_maskz_loadu_epi8(m, (const void *)(in1 + i));
        const __m512i b = _mm512_maskz_loadu_epi8(m, (const void *)(in2 + i));
        _mm512_mask_storeu_epi8((void *)(out + i), m, a ^ gf16v_mul_u512(b, cc));
    }

    gf16_vector_scalar_add_tail(out, in1, c, in2, bytes, n);
}

/// out += c*in
/// \param n[in]: number of elements NOT bytes
static inline
void gf16_vector_scalar_add_u512(gf16 *out,
                                 const gf16 c,
                                 const gf16 *in,
                                 const uint32_t n) {
    gf16_vector_scalar_add_u512_v2(out, out, c, in, n);
}

/// \param n[in]: number of elements NOT bytes
/// \return sum_i a_i*b_i
static inline
gf16 gf16_vector_mul_acc_u512(const gf16 *a,
                              const gf16 *b,
                              const uint32_t n) {
    const uint32_t bytes = n/2;
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i+64) <= bytes; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc = _mm512_xor_si512(acc, gf16v_mul_u512(t1, t2));
    }

    if (i < bytes) {
        const __mmask64 m = (__mmask64)((1ull << (bytes - i)) - 1ull);
        const __m512i t1 = _mm512_maskz_loadu_epi8(m, (const void *)(a + i));
        const __m512i t2 = _mm512_maskz_loadu_epi8(m, (const void *)(b + i));
        acc = _mm512_xor_si512(acc, gf16v_mul_u512(t1, t2));
    }

    uint8_t tmp[64];
    _mm512_storeu_si512((void *)tmp, acc);
    uint8_t r = 0;
    for (uint32_t k = 0; k < 64; k++) { r ^= tmp[k]; }
    r = (r & 0xF) ^ (r >> 4);
    return r ^ gf16_vector_mul_acc_tail(a, b, bytes, n);
}
#endif // USE_AVX512
