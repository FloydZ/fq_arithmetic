#pragma once 

#include "arith.h"
#include "../16/vector.h"

/// @param n size of the vector
/// @return vector(n)
static inline gf256* gf256_vector_alloc(const uint32_t n) {
    return (gf256 *)malloc(n*sizeof(gf256));
}

/// @param v vector to print
/// @param n size of the vector
static inline void gf256_vector_print(gf256 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        printf("%04d ", v[i]);
    }
    printf("\n");
}

/// @param v = 0
/// @param n size of vector
static inline void gf256_vector_zero(gf256 *v,
                                     const uint32_t n) {
    memset(v, 0, n*sizeof(gf256));
}

/// @param v v[i] = rand()
/// @param n size of vector
static inline void gf256_vector_rand(gf256 *v,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v[i] = rand() & 0x3FF;
    }
}

///
/// @param v1 = v2 
/// @param v2
/// @param n size of the two vectors
static inline void gf256_vector_copy(gf256 *__restrict__ v1,
                                     const gf256 *__restrict__ v2,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        v1[i] = v2[i];
    }
}

///
/// a = b
static inline void gf256_vector_set_to_gf16(gf256 *a,
                                            const gf16 *b,
                                            const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t = gf16_vector_get(b, i);
        a[i] = t;
    }
}

///
/// @param out += in
/// @param in
/// @param n
static inline void gf256_vector_add(gf256 *out,
                                    const gf256 *in,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= in[i];
    }
}


///
/// @param out += in
/// @param in
/// @param n
static inline void gf256_vector_add_gf2(gf256 *__restrict__ out,
                                        const gf16 *__restrict__ in,
                                        const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t1 = gf16_vector_get(in, i);
        const gf256 t2 = gf256_expand_tab[t1];
        out[i] ^= t2;
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf256_vector_scalar_add(gf256 *out,
                                           const gf256 c,
                                           const gf256 *in,
                                           const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf256_mul(c, in[i]);
    }
}

/// @param out += c*in
/// @param c
/// @param in
/// @param n
static inline void gf256_vector_scalar_add_gf2(gf256 *__restrict__ out,
                                               const gf16 c,
                                               const gf256 *__restrict__ in,
                                               const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        out[i] ^= gf256_mul_gf16(in[i], c);
    }
}


/// @param out += t*in
/// @param in
/// @param n length of the vectors
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
static inline void gf256_vector_add_scalar_u256(gf256 *out,
                                                const gf256 *in1,
                                                const gf256 scalar,
                                                const gf256 *in2,
                                                const size_t bytes) {
    size_t i = bytes;
    __m256i s256 = _mm256_set1_epi32(scalar), tmp;
    __m128i s128 = _mm_set1_epi32(scalar), tmp128;

    // avx2 code
    while (i >= 32u) {
        tmp = gf256v_mul_u256(_mm256_loadu_si256((__m256i *)in2), s256);
        _mm256_storeu_si256((__m256i *)out, _mm256_loadu_si256((__m256i *)in1) ^ tmp);
        i -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        tmp128 = gf256v_mul_u128(_mm_loadu_si128((__m128i *)in2), s128);
        _mm_storeu_si128((__m128i *)out, _mm_loadu_si128((__m128i *)in1) ^ tmp128);
        in1 += 16u;
        in2 += 16u;
        out += 16u;
        i -= 16;
    }

    for(uint32_t j = 0; j<i; j++) {
        out[j] = in1[j] ^ gf256_mul(scalar, in2[j]);
    }
}

/// TODO unfinished
/// \param n = number of elements in `in`, not bytes.
static inline void gf2to12_vector_set_to_gf16_u256(gf256 *out,
                                                   const gf16 *in,
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
