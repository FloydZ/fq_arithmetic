#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "../16/vector.h"
#include "../2/vector.h"

/// unaligned load of 4 bytes
static inline uint32_t gf256_load_u32(const uint8_t *in) {
    uint32_t r;
    memcpy(&r, in, sizeof(r));
    return r;
}

/// unaligned load of 8 bytes
static inline uint64_t gf256_load_u64(const uint8_t *in) {
    uint64_t r;
    memcpy(&r, in, sizeof(r));
    return r;
}

/// unaligned store of 8 bytes
static inline void gf256_store_u64(uint8_t *out, const uint64_t a) {
    memcpy(out, &a, sizeof(a));
}

/// \param n size of the vector
/// \return vector(n), initialized to zero
static inline gf256* gf256_vector_alloc(const uint32_t n) {
    return (gf256 *)calloc(n, sizeof(gf256));
}

/// \return v[i]
static inline gf256 gf256_vector_get(const gf256 *v,
                                     const uint32_t i) {
    return v[i];
}

/// v[i] = a
static inline void gf256_vector_set(gf256 *v,
                                    const uint32_t i,
                                    const gf256 a) {
    v[i] = a;
}

/// \param v vector to print
/// \param n size of the vector
static inline void gf256_vector_print(const gf256 *v,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

/// \param v = 0
/// \param n size of vector
static inline void gf256_vector_zero(gf256 *v,
                                     const uint32_t n) {
    memset(v, 0, n*sizeof(gf256));
}

/// \param v v[i] = rand()
/// \param n size of vector
static inline void gf256_vector_random(gf256 *v,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0xFF;
    }
}

/// \param v1 = v2
/// \param v2
/// \param n size of the two vectors
static inline void gf256_vector_copy(gf256 *__restrict__ v1,
                                     const gf256 *__restrict__ v2,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

/// a = b
static inline void gf256_vector_set_to_gf16(gf256 *a,
                                            const gf16 *b,
                                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16  t1 = gf16_vector_get(b, i);
        const gf256 t2 = gf256_expand_tab[t1];
        a[i] = t2;
    }
}

/// a = b
static inline void gf256_vector_set_to_gf2(gf256 *a,
                                           const gf2 *b,
                                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf256 t1 = gf2_vector_get(b, i);
        a[i] = t1;
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf256_vector_add(gf256 *out,
                                    const gf256 *in,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf256_vector_add_gf16(gf256 *__restrict__ out,
                                        const gf16 *__restrict__ in,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t1 = gf16_vector_get(in, i);
        const gf256 t2 = gf256_expand_tab[t1];
        out[i] ^= t2;
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf256_vector_add_gf2(gf256 *__restrict__ out,
                                        const gf2 *__restrict__ in,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf256 t1 = gf2_vector_get(in, i);
        out[i] ^= t1;
    }
}

/// \param out = in1 + in2
/// \param in1
/// \param in2
/// \param n
static inline void gf256_vector_add_2_gf2(gf256 *__restrict__ out,
                                          const gf256 *__restrict__ in1,
                                          const gf2 *__restrict__ in2,
                                          const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf256 t1 = gf2_vector_get(in2, i);
        out[i] = in1[i] ^ t1;
    }
}
/// \brief vector1 = vector2 + vector3
///
/// \param[out] vector1 Vector over gf256
/// \param[in] vector2 Vector over gf256
/// \param[in] vector3 Vector over gf16
/// \param[in] ncols number of columns
static inline void gf256_vector_add_2_gf16(gf256 *vector1,
                                           const gf256 *vector2,
                                           const gf16 *vector3,
                                           const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const ff_t t = gf16_vector_get(vector3, i);
        vector1[i] = vector2[i] ^ gf256_expand_tab[t];
    }
}


/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf256_vector_scalar_add(gf256 *out,
                                           const gf256 c,
                                           const gf256 *in,
                                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf256_mul(c, in[i]);
    }
}

/// \param out = in1 + c*in2
/// \param in1
/// \param c
/// \param in2
/// \param n
static inline void gf256_vector_scalar_add_v2(gf256 *out,
                                              const gf256 *in1,
                                              const gf256 c,
                                              const gf256 *in2,
                                              const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] = in1[i] ^ gf256_mul(c, in2[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n
static inline void gf256_vector_scalar_add_gf16(gf256 *__restrict__ out,
                                               const gf16 c,
                                               const gf256 *__restrict__ in,
                                               const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf256_mul_gf16(in[i], c);
    }
}


/// \param out += t*in
/// \param in
/// \param n length of the vectors
static inline void gf256_vector_scalar_add_gf2_v3(gf256 *__restrict__ out,
                                                    const gf256 t,
                                                    const gf16 *__restrict__ in,
                                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 a = gf16_vector_get(in, i);
        const gf256 b = gf256_mul_gf16(t, a);
        out[i] ^= b;
    }
}

/// out[i] = scalar*in2[i] for all i = 0...bytes-1
static inline void gf256_vector_scalar_gf2(gf256 *out,
                                           const gf256 scalar,
                                           const gf2 *in2,
                                           const size_t bytes) {
    for (uint32_t i = 0; i < bytes; i++) {
        out[i] = gf256_mul(gf2_vector_get(in2, i), scalar);
    }
}

/// \brief vector1 = vector2 * scalar
///
/// \param[out] vector1 Vector over gf256
/// \param[in] scalar Scalar over ff_mu
/// \param[in] vector2 Vector over ff
/// \param[in] ncols number of columns
static inline void gf256_vector_scalar_gf16(gf256 *vector1,
                                            const gf256 scalar,
                                            const gf16 *vector2,
                                            const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; i++) {
        const gf16 t = gf16_vector_get(vector2, i);
        vector1[i] = gf256_mul(scalar, gf256_expand_tab[t]);
    }
}

/// \param a
/// \param b
/// \param n length of the vector
/// \return sum(a[i] * b[i])
static inline gf256 gf256_vector_mul_acc(const gf256 *a,
                                             const gf256 *b,
                                             const uint32_t n) {
    gf256 acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc ^= gf256_mul(a[i], b[i]);
    }

    return acc;
}

#ifdef USE_AVX2

/// loads 32 elements/ 16 bytes and extends them to gf256
static inline __m256i gf256_vector_extend_gf16_x32(const gf16 *in) {
    const __m128i tbl   = _mm_load_si128((__m128i *)gf256_expand_tab);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    const __m128i load = _mm_loadu_si128((__m128i *) (in));
    const __m128i ll = _mm_shuffle_epi8(tbl, load & mask);
    const __m128i lh = _mm_shuffle_epi8(tbl, _mm_srli_epi16(load, 4) & mask);
    const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
    const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
    return _mm256_unpacklo_epi8(tl, th);
}

/// \param in 4 bytes
/// \return
static inline __m256i gf256v_expand_gf2_x32_u256(const uint8_t *in) {
    const uint8_t t11 = *(in + 0);
    const uint8_t t12 = *(in + 1);
    const uint8_t t13 = *(in + 2);
    const uint8_t t14 = *(in + 3);

    const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
    const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
    const uint64_t t23 = _pdep_u64(t13, 0x0101010101010101);
    const uint64_t t24 = _pdep_u64(t14, 0x0101010101010101);
    return _mm256_setr_epi64x(t21, t22, t23, t24);
}

/// \param in 2 bytes
/// \return
static inline __m128i gf256v_expand_gf2_x16_u128(const uint8_t *in) {
    const uint32_t t11 = *(in + 0);
    const uint32_t t12 = *(in + 1);
    const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
    const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
    return _mm_set_epi64x(t22, t21);
}

/// \param in 1 byte
/// \return
static inline uint64_t gf256v_expand_gf2_x8_u64(const uint8_t *in) {
    const uint32_t t11 = *(in + 0);
    return _pdep_u64(t11, 0x0101010101010101);
}

/// \param a[in]: [a_0, ..., a_31]
/// \return a_0 + ... + a_31
static inline gf256 gf256v_hadd_u256(const __m256i a) {
    const __m128i t1 = _mm256_castsi256_si128(a) ^ _mm256_extracti128_si256(a, 1);
    const __m128i t2 = t1 ^ _mm_bsrli_si128(t1, 8);
    uint64_t t = (uint64_t)_mm_cvtsi128_si64(t2);
    t ^= t >> 32;
    t ^= t >> 16;
    t ^= t >>  8;
    return (gf256)t;
}

/// \param out += in
/// \param in
/// \param n number of elements
static inline void gf256_vector_add_u256(gf256 *out,
                                         const gf256 *in,
                                         const uint32_t n) {
    uint32_t i = 0;
    for (; (i+32) <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(in + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *)(out + i));
        _mm256_storeu_si256((__m256i *)(out + i), a ^ b);
    }

    for (; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// out = in1 ^ in2
static inline void gf256_vector_add_u256_v2(gf256 *out,
                                            const gf256 *in1,
                                            const gf256 *in2,
                                            const uint32_t n) {
    uint32_t i = n;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((const __m256i *)in1) ^
                            _mm256_loadu_si256((const __m256i *)in2));
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((const __m128i *)in1) ^
                         _mm_loadu_si128((const __m128i *)in2));
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    for(uint32_t j = 0; j<i; j++) {
        out[j] = in1[j] ^ in2[j];
    }
}

/// \param out = in1 + in2
/// \param in1
/// \param in2
/// \param n
static inline void gf256_vector_add_2_gf2_u256(gf256 *out,
                                               const gf256 *in1,
                                               const gf2 *in2,
                                               const uint32_t n) {
    uint32_t i = n;
    while (i >= 32u) {
        const uint32_t t11 = *(in2 + 0);
        const uint32_t t12 = *(in2 + 1);
        const uint32_t t13 = *(in2 + 2);
        const uint32_t t14 = *(in2 + 3);

        const __m256i m = _mm256_loadu_si256((const __m256i *)in1);
        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
        const uint64_t t23 = _pdep_u64(t13, 0x0101010101010101);
        const uint64_t t24 = _pdep_u64(t14, 0x0101010101010101);

        const __m256i t = _mm256_setr_epi64x(t21, t22, t23, t24);
        _mm256_storeu_si256((__m256i *)out, t^m);

        in2 += 4u;
        in1 += 32u;
        out += 32u;
        i   -= 32u;
    }

    while (i >= 8u) {
        const uint32_t t1 = *(in2 + 0);
        const uint64_t m = gf256_load_u64(in1);
        const uint64_t t2 = _pdep_u64(t1, 0x0101010101010101);
        gf256_store_u64(out, t2 ^ m);

        in2 += 1u;
        in1 += 8u;
        out += 8u;
        i   -= 8u;
    }

    if (i) {
        uint8_t tmp[16] = {0};
        const uint8_t mask = (1u << i) - 1u;
        const uint32_t t1 = (*in2) & mask;

        for (uint64_t j = 0; j < i; ++j) { tmp[j] = in1[j]; }
        uint64_t m = gf256_load_u64(tmp);

        const uint64_t t2 = _pdep_u64(t1, 0x0101010101010101);
        gf256_store_u64(tmp, t2 ^ m);
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j]; }
    }
}

/// \param out = in1 + in2
/// \param in1
/// \param in2
/// \param n
static inline void gf256_vector_add_2_gf16_u256(gf256 *__restrict__ out,
                                                const gf256 *__restrict__ in1,
                                                const gf16 *__restrict__ in2,
                                                const uint32_t n) {
    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);

    uint32_t i = n;
    while (i >= 32u) {
        const __m256i m = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t = gf256_vector_extend_gf16_x32(in2);
        _mm256_storeu_si256((__m256i *)out, t^m);

        in2 += 16u;
        in1 += 32u;
        out += 32u;
        i   -= 32u;
    }

    while (i >= 8u) {
        const uint32_t t1 = gf256_load_u32(in2);
        const uint64_t t2 = _pdep_u64(t1, 0x0F0F0F0F0F0F0F0F);

        const uint64_t m = gf256_load_u64(in1);
        const __m128i t3 = _mm_set_epi64x(0, (long long)t2);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        const uint64_t t5 = _mm_extract_epi64(t4, 0);
        gf256_store_u64(out, t5 ^ m);

        in2 += 4u;
        in1 += 8u;
        out += 8u;
        i   -= 8u;
    }

    if (i) {
        uint8_t tmp[8] = {0};
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = in2[j]; }
        const uint32_t t1 = gf256_load_u32(tmp);

        for (uint64_t j = 0; j < i; ++j) { tmp[j] = in1[j]; }
        uint64_t m = gf256_load_u64(tmp);

        const uint64_t t2 = _pdep_u64(t1, 0x0F0F0F0F0F0F0F0F);
        const __m128i t3 = _mm_set_epi64x(0, (long long)t2);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        const uint64_t t5 = _mm_extract_epi64(t4, 0);

        gf256_store_u64(tmp, t5 ^ m);
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j]; }
    }
}

/// out[i] = in1[i] + scalar*in2[i] for all i = 0...n-1
static inline void gf256_vector_scalar_add_u256_v2(gf256 *out,
                                                   const gf256 *in1,
                                                   const gf256 scalar,
                                                   const gf256 *in2,
                                                   const uint32_t n) {
    uint32_t i = n;
    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    const __m128i ml128 = _mm256_extracti128_si256(ml, 0);
    const __m128i mh128 = _mm256_extracti128_si256(mh, 0);
    const __m128i mask128 = _mm_set1_epi8(0xF);

    // avx2 code
    while (i >= 32u) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)in2);
        const __m256i tmp = gf256_linear_transform_8x8_256b(ml, mh, a, mask);
        _mm256_storeu_si256((__m256i *)out, _mm256_loadu_si256((const __m256i *)in1) ^ tmp);
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        const __m128i a = _mm_loadu_si128((const __m128i *)in2);
        const __m128i tmp = gf256_linear_transform_8x8_128b(ml128, mh128, a, mask128);

        _mm_storeu_si128((__m128i *)out, _mm_loadu_si128((const __m128i *)in1) ^ tmp);
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    if (i) {
        uint8_t tmp[16] __attribute__((aligned(32))) = {0};
        for (uint32_t k = 0; k < i; k++) { tmp[k] = in2[k]; }
        const __m128i a = _mm_load_si128((const __m128i *) tmp);
        for (uint32_t k = 0; k < i; k++) { tmp[k] = in1[k]; }
        const __m128i b = _mm_load_si128((const __m128i *) tmp);
        const __m128i c = gf256_linear_transform_8x8_128b(ml128, mh128, a, mask128);
        const __m128i d = c ^ b;

        _mm_store_si128((__m128i *) tmp, d);
        for (uint32_t k = 0; k < i; k++) { out[k] = tmp[k]; }
    }
}

/// out[i] += scalar*in[i] for all i = 0...n-1
static inline void gf256_vector_scalar_add_u256(gf256 *out,
                                                const gf256 scalar,
                                                const gf256 *in,
                                                const uint32_t n) {
    gf256_vector_scalar_add_u256_v2(out, out, scalar, in, n);
}

/// \param a[in]: vector of length n
/// \param b[in]: vector of length n
/// \param n[in]: number of elements
/// \return sum(a[i] * b[i])
static inline gf256 gf256_vector_mul_acc_u256(const gf256 *a,
                                              const gf256 *b,
                                              const uint32_t n) {
    __m256i acc = _mm256_setzero_si256();
    uint32_t i = 0;
    for (; (i+32) <= n; i += 32) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(b + i));
        acc ^= gf256v_mul_u256(t1, t2);
    }

    if (i < n) {
        uint8_t tmp1[32] __attribute__((aligned(32))) = {0};
        uint8_t tmp2[32] __attribute__((aligned(32))) = {0};
        for (uint32_t k = 0; k < (n - i); k++) { tmp1[k] = a[i + k]; }
        for (uint32_t k = 0; k < (n - i); k++) { tmp2[k] = b[i + k]; }
        const __m256i t1 = _mm256_load_si256((const __m256i *)tmp1);
        const __m256i t2 = _mm256_load_si256((const __m256i *)tmp2);
        acc ^= gf256v_mul_u256(t1, t2);
    }

    return gf256v_hadd_u256(acc);
}


/// out[i] = scalar*in2[i] for all i = 0...bytes-1
static inline void gf256_vector_scalar_gf2_u256(gf256 *out,
                                               const gf256 scalar,
                                               const gf2 *in2,
                                               const size_t bytes) {
    uint32_t i = bytes;
    const __m256i s = _mm256_set1_epi8(scalar);
    const __m256i zero = _mm256_setzero_si256();

    while (i >= 32u) {
        const uint32_t t11 = *(in2 + 0);
        const uint32_t t12 = *(in2 + 1);
        const uint32_t t13 = *(in2 + 2);
        const uint32_t t14 = *(in2 + 3);

        const uint64_t t21 = _pdep_u64(t11, 0x8080808080808080);
        const uint64_t t22 = _pdep_u64(t12, 0x8080808080808080);
        const uint64_t t23 = _pdep_u64(t13, 0x8080808080808080);
        const uint64_t t24 = _pdep_u64(t14, 0x8080808080808080);

        const __m256i t1 = _mm256_setr_epi64x(t21, t22, t23, t24);
        const __m256i t2 = _mm256_blendv_epi8(zero, s, t1);

        _mm256_storeu_si256((__m256i *)out, t2);

        in2 += 4u;
        out += 32u;
        i   -= 32u;
    }

    if (i) {
        uint64_t tmp[4] __attribute__((aligned(32))) = {0};
        uint8_t *tmp2 = (uint8_t *)tmp;
        for (uint32_t j = 0; j < (i+7)/8; ++j) {
            tmp[j] = _pdep_u64(in2[j], 0x8080808080808080);
        }

        const __m256i t2 = _mm256_blendv_epi8(zero, s, _mm256_load_si256((const __m256i *)(tmp)));
        _mm256_storeu_si256((__m256i *)tmp, t2);
        for (uint32_t j = 0; j < i; j++) { out[j] = tmp2[j]; }
    }
}

static inline void gf256_vector_add_scalar_gf2_u256(gf256 *out,
                                                    const gf256 scalar,
                                                    const gf2 *in2,
                                                    const size_t bytes) {
    uint32_t i = bytes;
    const __m256i s = _mm256_set1_epi8(scalar);
    const __m256i zero = _mm256_setzero_si256();

    while (i >= 32u) {
        const uint32_t t11 = *(in2 + 0);
        const uint32_t t12 = *(in2 + 1);
        const uint32_t t13 = *(in2 + 2);
        const uint32_t t14 = *(in2 + 3);

        const uint64_t t21 = _pdep_u64(t11, 0x8080808080808080);
        const uint64_t t22 = _pdep_u64(t12, 0x8080808080808080);
        const uint64_t t23 = _pdep_u64(t13, 0x8080808080808080);
        const uint64_t t24 = _pdep_u64(t14, 0x8080808080808080);

        const __m256i t1 = _mm256_setr_epi64x(t21, t22, t23, t24);
        const __m256i t2 = _mm256_blendv_epi8(zero, s, t1);

        const __m256i m1 = _mm256_loadu_si256((const __m256i *)out);
        _mm256_storeu_si256((__m256i *)out, t2^m1);

        in2 += 4u;
        out += 32u;
        i   -= 32u;
    }

    if (i) {
        uint64_t tmp[4] __attribute__((aligned(32))) = {0};
        uint8_t *tmp2 = (uint8_t *)tmp;
        for (uint32_t j = 0; j < (i+7)/8; ++j) {
            tmp[j] = _pdep_u64(in2[j], 0x8080808080808080);
        }

        const __m256i t2 = _mm256_blendv_epi8(zero, s, _mm256_load_si256((const __m256i *)(tmp)));
        _mm256_storeu_si256((__m256i *)tmp, t2);
        for (uint32_t j = 0; j < i; j++) { out[j] ^= tmp2[j]; }
    }
}

/// out[i] = scalar*in2[i] for all i = 0...bytes-1
static inline void gf256_vector_scalar_gf16_u256(gf256 *out,
                                                const gf256 scalar,
                                                const gf16 *in2,
                                                const size_t bytes) {
    uint32_t i = bytes;

    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    while (i >= 32u) {
        const __m256i a = gf256_vector_extend_gf16_x32(in2);
        const __m256i t = gf256_linear_transform_8x8_256b(ml, mh, a, mask);
        _mm256_storeu_si256((__m256i *)out, t);

        in2 += 16u;
        out += 32u;
        i   -= 32u;
    }

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32))) = {0};
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = in2[j]; }

        const __m256i a = gf256_vector_extend_gf16_x32(tmp);
        const __m256i t = gf256_linear_transform_8x8_256b(ml, mh, a, mask);
        _mm256_store_si256((__m256i *)tmp, t);

        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp[j]; }
    }
}

/// \param n = number of elements in `in`, not bytes.
static inline void gf256_vector_set_to_gf16_u256(gf256 *out,
                                                 const gf16 *in,
                                                 const uint32_t n) {
    uint32_t i = n;
    const gf16 *in2 = in;

    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    while (i >= 32u) {
        const __m128i load = _mm_loadu_si128((const __m128i *)in);
        const __m128i ll = _mm_shuffle_epi8(perm128, load & mask);
        const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask);
        const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
        const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
        const __m256i t4 = _mm256_unpacklo_epi8(tl, th);
        _mm256_storeu_si256((__m256i *)out, t4);

        in  += 16u;
        out += 32u;
        i   -= 32u;
    }

    while (i >= 16u) {
        const uint32_t t11 = gf256_load_u32(in + 0);
        const uint32_t t12 = gf256_load_u32(in + 4);

        const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
        const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);

        const __m128i t3 = _mm_set_epi64x((long long)t22, (long long)t21);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        _mm_storeu_si128((__m128i *)out, t4);
        in  += 8u;
        out += 16u;
        i   -= 16u;
    }

    while (i >= 8) {
        const uint32_t t1 = gf256_load_u32(in);
        const uint64_t t2 = _pdep_u64(t1, 0x0F0F0F0F0F0F0F0F);

        const __m128i t3 = _mm_set_epi64x(0, (long long)t2);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        const uint64_t t5 = _mm_extract_epi64(t4, 0);
        gf256_store_u64(out, t5);
        in  += 4u;
        out += 8u;
        i   -= 8u;
    }

    while (i) {
        const gf16 t1 = gf16_vector_get(in2, n - i);
        const gf256 t2 = gf256_expand_tab[t1];
        *out = t2;
        out++;
        i -= 1;
    }
}

/// same function as `gf256_vectro_set_to_gf16_u256` just with another core idea
static inline void gf256_vector_set_to_gf16_u256_v2(gf256 *out,
                                                    const gf16 *in,
                                                    const uint32_t n) {
    uint32_t i = n;
    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    while (i >= 32u) {
        const __m128i load = _mm_loadu_si128((const __m128i *)in);
        const __m128i ll = _mm_shuffle_epi8(perm128, load & mask);
        const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask);
        const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
        const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
        const __m256i t = _mm256_unpacklo_epi8(tl, th);
        _mm256_storeu_si256((__m256i *)out, t);

        i   -= 32;
        out += 32;
        in  += 16;
    }

    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32))) = {0};
        for (uint32_t j = 0; j < (i+1)/2; ++j) { tmp[j] = in[j]; }
        const __m128i load = _mm_load_si128((const __m128i *)tmp);
        const __m128i ll = _mm_shuffle_epi8(perm128, load & mask);
        const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask);
        const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
        const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
        const __m256i t = _mm256_unpacklo_epi8(tl, th);
        _mm256_store_si256((__m256i *)tmp, t);
        for (uint32_t j = 0; j < i; ++j) { out[j] = tmp[j];}
    }
}

/// \param n = number of elements in `in`, not bytes.
static inline void gf256_vector_set_to_gf2_u256(gf256 *out,
                                                const gf2 *in,
                                                const uint32_t n) {
    uint32_t i = n;
    while (i >= 32u) {
        const uint32_t t11 = *(in + 0);
        const uint32_t t12 = *(in + 1);
        const uint32_t t13 = *(in + 2);
        const uint32_t t14 = *(in + 3);

        const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
        const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);
        const uint64_t t23 = _pdep_u64(t13, 0x0101010101010101);
        const uint64_t t24 = _pdep_u64(t14, 0x0101010101010101);

        const __m256i t = _mm256_setr_epi64x(t21, t22, t23, t24);
        _mm256_storeu_si256((__m256i *)out, t);

        in  += 4u;
        out += 32u;
        i   -= 32u;
    }

    while (i >= 8u) {
        const uint32_t t1 = *(in + 0);
        const uint64_t t2 = _pdep_u64(t1, 0x0101010101010101);
        gf256_store_u64(out, t2);

        in  += 1u;
        out += 8u;
        i   -= 8u;
    }

    if (i) {
        uint8_t tmp[8];
        const uint8_t mask = (1u << i) - 1u;
        const uint32_t t1 = (*in) & mask;
        const uint64_t t2 = _pdep_u64(t1, 0x0101010101010101);
        gf256_store_u64(tmp, t2);
        for (uint32_t j = 0; j < i; j++) {
            out[j] = tmp[j];
        }
    }
}

#elif defined(USE_NEON)

/// \param out += in
/// \param in
/// \param n number of elements
static inline void gf256_vector_add_u128(gf256 *out,
                                         const gf256 *in,
                                         const uint32_t n) {
    uint32_t i = 0;
    for (; (i+16) <= n; i += 16) {
        vst1q_u8(out + i, veorq_u8(vld1q_u8(out + i), vld1q_u8(in + i)));
    }

    for (; i < n; i++) {
        out[i] ^= in[i];
    }
}

/// \param out = in1 + in2
/// \param in1
/// \param in2
/// \param n number of elements
static inline void gf256_vector_add_u128_v2(gf256 *out,
                                            const gf256 *in1,
                                            const gf256 *in2,
                                            const uint32_t n) {
    uint32_t i = 0;
    for (; (i+16) <= n; i += 16) {
        vst1q_u8(out + i, veorq_u8(vld1q_u8(in1 + i), vld1q_u8(in2 + i)));
    }

    for (; i < n; i++) {
        out[i] = in1[i] ^ in2[i];
    }
}

/// \param out = in1 + c*in2
/// \param in1
/// \param c
/// \param in2
/// \param n number of elements
static inline void gf256_vector_scalar_add_u128_v2(gf256 *out,
                                                   const gf256 *in1,
                                                   const gf256 c,
                                                   const gf256 *in2,
                                                   const uint32_t n) {
    const uint8x16_t s = vdupq_n_u8(c);
    uint32_t i = 0;
    for (; (i+16) <= n; i += 16) {
        const uint8x16_t t = gf256v_mul_u128(s, vld1q_u8(in2 + i));
        vst1q_u8(out + i, veorq_u8(vld1q_u8(in1 + i), t));
    }

    for (; i < n; i++) {
        out[i] = in1[i] ^ gf256_mul(c, in2[i]);
    }
}

/// \param out += c*in
/// \param c
/// \param in
/// \param n number of elements
static inline void gf256_vector_scalar_add_u128(gf256 *out,
                                                const gf256 c,
                                                const gf256 *in,
                                                const uint32_t n) {
    gf256_vector_scalar_add_u128_v2(out, out, c, in, n);
}

/// \param a[in]: vector of length n
/// \param b[in]: vector of length n
/// \param n[in]: number of elements
/// \return sum(a[i] * b[i])
static inline gf256 gf256_vector_mul_acc_u128(const gf256 *a,
                                              const gf256 *b,
                                              const uint32_t n) {
    uint8x16_t acc = vdupq_n_u8(0);
    uint32_t i = 0;
    for (; (i+16) <= n; i += 16) {
        acc = veorq_u8(acc, gf256v_mul_u128(vld1q_u8(a + i), vld1q_u8(b + i)));
    }

    const uint64x2_t t1 = vreinterpretq_u64_u8(acc);
    uint64_t t = vgetq_lane_u64(t1, 0) ^ vgetq_lane_u64(t1, 1);
    t ^= t >> 32;
    t ^= t >> 16;
    t ^= t >>  8;
    gf256 r = (gf256)t;

    for (; i < n; i++) {
        r ^= gf256_mul(a[i], b[i]);
    }
    return r;
}
#endif // end avx/neon


#ifdef USE_AVX512

/// \param out += in
/// \param in
/// \param n number of elements
static inline void gf256_vector_add_u512(gf256 *out,
                                         const gf256 *in,
                                         const uint32_t n) {
    uint32_t i = 0;
    for (; (i+64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in + i));
        const __m512i b = _mm512_loadu_si512((const void *)(out + i));
        _mm512_storeu_si512((void *)(out + i), gf256v_add_u512(a, b));
    }

    gf256_vector_add_u256(out + i, in + i, n - i);
}

/// out = in1 + in2
/// \param n number of elements
static inline void gf256_vector_add_u512_v2(gf256 *out,
                                            const gf256 *in1,
                                            const gf256 *in2,
                                            const uint32_t n) {
    uint32_t i = 0;
    for (; (i+64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), gf256v_add_u512(a, b));
    }

    gf256_vector_add_u256_v2(out + i, in1 + i, in2 + i, n - i);
}

/// out = in1 + scalar*in2
/// \param n number of elements
static inline void gf256_vector_scalar_add_u512_v2(gf256 *out,
                                                   const gf256 *in1,
                                                   const gf256 scalar,
                                                   const gf256 *in2,
                                                   const uint32_t n) {
    const __m512i s = _mm512_set1_epi8((char)scalar);
    uint32_t i = 0;
    for (; (i+64) <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
        const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
        _mm512_storeu_si512((void *)(out + i), a ^ gf256v_mul_u512(b, s));
    }

    gf256_vector_scalar_add_u256_v2(out + i, in1 + i, scalar, in2 + i, n - i);
}

/// out += scalar*in
/// \param n number of elements
static inline void gf256_vector_scalar_add_u512(gf256 *out,
                                                const gf256 scalar,
                                                const gf256 *in,
                                                const uint32_t n) {
    gf256_vector_scalar_add_u512_v2(out, out, scalar, in, n);
}

/// \return sum(a[i] * b[i])
static inline gf256 gf256_vector_mul_acc_u512(const gf256 *a,
                                              const gf256 *b,
                                              const uint32_t n) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t i = 0;
    for (; (i+64) <= n; i += 64) {
        const __m512i t1 = _mm512_loadu_si512((const void *)(a + i));
        const __m512i t2 = _mm512_loadu_si512((const void *)(b + i));
        acc ^= gf256v_mul_u512(t1, t2);
    }

    const __m256i t = _mm512_castsi512_si256(acc) ^ _mm512_extracti64x4_epi64(acc, 1);
    return gf256v_hadd_u256(t) ^ gf256_vector_mul_acc_u256(a + i, b + i, n - i);
}

#endif
