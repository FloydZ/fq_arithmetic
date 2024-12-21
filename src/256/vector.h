#pragma once 

#include "arith.h"
#include "../16/vector.h"

/// \param n size of the vector
/// \return vector(n)
static inline gf256* gf256_vector_alloc(const uint32_t n) {
    return (gf256 *)malloc(n*sizeof(gf256));
}

/// \param v vector to print
/// \param n size of the vector
static inline void gf256_vector_print(gf256 *v,
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
static inline void gf256_vector_rand(gf256 *v,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0x3FF;
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
        const gf16 t1 = gf16_vector_get(b, i);
        const gf16 t2 = gf256_expand_tab[t1];
        a[i] = t2;
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

/// out = in1 ^ in2
static inline void gf256_vector_add_u256(gf256 *out,
                                         const gf256 *in1,
                                         const gf256 *in2,
                                         const size_t bytes) {
    size_t i = bytes;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    for(uint32_t j = 0; j<i; j++) {
        out[j] = in1[j] ^ in2[j];
    }
}

/// out[i] = in1[i] + scalar*in2[i] for all i = 0...bytes-1
static inline void gf256_vector_add_scalar_3_u256(gf256 *out,
                                                  const gf256 *in1,
                                                  const gf256 scalar,
                                                  const gf256 *in2,
                                                  const size_t bytes) {
    size_t i = bytes;
    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    const __m128i ml128 = _mm256_extracti128_si256(ml, 0);
    const __m128i mh128 = _mm256_extracti128_si256(mh, 0);
    const __m128i mask128 = _mm_set1_epi8(0xF);

    // avx2 code
    while (i >= 32u) {
        const __m256 a = _mm256_loadu_si256((__m256i *)in2);
        const __m256i tmp = gf256_linear_transform_8x8_256b(ml, mh, a, mask);
        _mm256_storeu_si256((__m256i *)out, _mm256_loadu_si256((__m256i *)in1) ^ tmp);
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        const __m128i a = _mm_loadu_si128((__m128i *)in2);
        const __m128i tmp = gf256_linear_transform_8x8_128b(ml128, mh128, a, mask128);

        _mm_storeu_si128((__m128i *)out, _mm_loadu_si128((__m128i *)in1) ^ tmp);
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    if (i) {
        uint8_t tmp[16] __attribute__((aligned(32)));
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

/// out[i] += scalar*in2[i] for all i = 0...bytes-1
static inline void gf256_vector_add_scalar_2_u256(gf256 *out,
                                                  const gf256 scalar,
                                                  const gf256 *in2,
                                                  const size_t bytes) {
    size_t i = bytes;
    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    const __m128i ml128 = _mm256_extracti128_si256(ml, 0);
    const __m128i mh128 = _mm256_extracti128_si256(mh, 0);
    const __m128i mask128 = _mm_set1_epi8(0xF);

    // avx2 code
    while (i >= 32u) {
        const __m256 a = _mm256_loadu_si256((__m256i *)in2);
        const __m256i tmp = gf256_linear_transform_8x8_256b(ml, mh, a, mask);
        _mm256_storeu_si256((__m256i *)out, _mm256_loadu_si256((__m256i *)out) ^ tmp);
        i -= 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        const __m128i a = _mm_loadu_si128((__m128i *)in2);
        const __m128i tmp = gf256_linear_transform_8x8_128b(ml128, mh128, a, mask128);

        _mm_storeu_si128((__m128i *)out, _mm_loadu_si128((__m128i *)out) ^ tmp);
        in2 += 16u;
        out += 16u;
        i -= 16;
    }


    if (i) {
        uint8_t tmp[16] __attribute__((aligned(32)));
        for (uint32_t k = 0; k < i; k++) { tmp[k] = in2[k]; }
        const __m128i a = _mm_load_si128((const __m128i *) tmp);
        for (uint32_t k = 0; k < i; k++) { tmp[k] = out[k]; }
        const __m128i b = _mm_load_si128((const __m128i *) tmp);
        const __m128i c = gf256_linear_transform_8x8_128b(ml128, mh128, a, mask128);
        const __m128i d = c ^ b;

        _mm_store_si128((__m128i *) tmp, d);
        for (uint32_t k = 0; k < i; k++) { out[k] = tmp[k]; }
    }
}

/// \param n = number of elements in `in`, not bytes.
static inline void gf256_vector_set_to_gf16_u256(gf256 *out,
                                                 const gf16 *in,
                                                 const uint32_t n) {
    uint32_t i = n;
    const gf16 *in2 = in;

    const __m256i perm256 = _mm256_load_si256((const __m256i *)gf256_expand_tab);
    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    while (i >= 32u) {
        // slower
        // const uint32_t t11 = *(uint32_t *)(in +  0);
        // const uint32_t t12 = *(uint32_t *)(in +  4);
        // const uint32_t t13 = *(uint32_t *)(in +  8);
        // const uint32_t t14 = *(uint32_t *)(in + 12);

        // const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
        // const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
        // const uint64_t t23 = _pdep_u64(t13, 0x0F0F0F0F0F0F0F0F);
        // const uint64_t t24 = _pdep_u64(t14, 0x0F0F0F0F0F0F0F0F);

        // const __m256i t3 = _mm256_setr_epi64x(t21, t22, t23, t24);
        // const __m256i t4 = _mm256_shuffle_epi8(perm256, t3);

        // faster
        const __m128i load = _mm_loadu_si128((__m128i *)in);
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
        const uint32_t t11 = *(uint32_t *)(in +  0);
        const uint32_t t12 = *(uint32_t *)(in +  4);

        const __m64 t21 = (__m64)_pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
        const __m64 t22 = (__m64)_pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);

        const __m128i t3 = _mm_setr_epi64(t21, t22);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        _mm_storeu_si128((__m128i *)out, t4);
        in  += 8u;
        out += 16u;
        i   -= 16u;
    }

    while (i >= 8) {
        const uint32_t t1 = *(uint32_t *)(in +  0);
        const uint64_t t2 = _pdep_u64(t1, 0x0F0F0F0F0F0F0F0F);

        const __m128i t3 = _mm_setr_epi64((__m64)t2, (__m64)0ull);
        const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
        const uint64_t t5 = _mm_extract_epi64(t4, 0);
        *(uint64_t *)out = t5;
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
        const __m128i load = _mm_loadu_si128((__m128i *)in);
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
        uint8_t tmp[32] __attribute__((aligned(32)));
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

#elif defined(USE_NEON)
// TODO
#endif // end avx/neon


#ifdef USE_AVX512

/// \return a * b
static inline __m512i gf256v_mul_u512(const __m512i a,
                                      const __m512i b) {
    return _mm512_gf2p8mul_epi8(a, b);
}

/// out = in1 ^ in2
static inline void gf256v_vector_add_u512(gf256 *out,
                                          const gf256 *in1,
                                          const gf256 *in2,
                                          const size_t bytes) {
    size_t i = bytes;
    // avx512 code
    while (i >= 64u) {
        _mm512_storeu_si512((__m512i *)out,
                            _mm512_loadu_si512((__m512i *)in1) ^
                            _mm512_loadu_si512((__m512i *)in2));
        i   -= 64u;
        in1 += 64u;
        in2 += 64u;
        out += 64u;
    }

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    for(uint32_t j = 0; j<i; j++) {
        out[j] = in1[j] ^ in2[j];
    }
}

/// out = in1 ^ scalar* in2
static inline void gf256_vector_add_scalar_u512(gf256 *out,
                                                const gf256 *in1,
                                                const gf256 scalar,
                                                const gf256 *in2,
                                                const size_t bytes) {
    size_t i = bytes;
    const __m512i s = _mm512_set1_epi8(scalar);

    // avx512 code
    while (i >= 64u) {
        const __m512i tmp = gf256v_mul_u512(_mm512_loadu_si512((__m512i *)in2), s);
        _mm512_storeu_si512((__m512i *)out, _mm512_loadu_si512((__m512i *)in1) ^ tmp);
        i   -= 64u;
        in1 += 64u;
        in2 += 64u;
        out += 64u;
    }

    uint8_t tmp[64];
    for (uint32_t j = 0; j < i; ++j) {
        tmp[j] = in2[j];
    }

    const __m512i t = gf256v_mul_u512(_mm512_loadu_si512((__m512i *)tmp), s);
    _mm512_storeu_si512((__m512i *)out, t);

    for (uint32_t j = 0; j < i; ++j) {
        out[j] = in1[j] ^ tmp[j];
    }
}

#endif
