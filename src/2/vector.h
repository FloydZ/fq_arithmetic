#pragma once
#include <stdlib.h>

#include "arith.h"

#define gf2_vector_size(N) ((N+7)/8)

/// \param n
/// \return
static inline gf2* gf2_vector_alloc(const uint32_t n) {
    return (gf2 *)malloc(((n + 7)/8)*sizeof(gf2));
}

/// \param vector
/// \param i
/// \return vector[i]
static inline gf2 gf2_vector_get(const gf2 *vector,
                                 const uint32_t i) {
    return (vector[i / (8)] >> (i % 8)) & 0x01;
}

/// vector[i] = scalar
/// \param vector
/// \param i 
/// \param scalar 
static inline void gf2_vector_set(gf2 *vector,
                                  const uint32_t i,
                                  const gf2 scalar) {
    const uint8_t mask = 0xff ^ (1 << (i % 8));
    vector[i/8] = (vector[i/8] & mask) ^ (scalar << (i % 8));
}

/// \param[in/out] arg1 = random
static inline void gf2_vector_random(gf2 *arg1,
                                     const uint32_t d) {
    const uint32_t limbs = (d+7)/8;
    for (uint32_t i = 0; i < limbs; i++) {
        arg1[i] = rand();
    }

    const uint32_t l = d%8;
    if (l) {
        const uint8_t mask = (1u << l) - 1u;
        arg1[limbs - 1] &= mask;
    }
}

/// \param[in/out] arg1 = 0
static inline void gf2_vector_zero(gf2 *arg1,
                                  const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = 0;
    }
}

/// arg1 = arg2 + arg3
/// \param arg1
/// \param arg2
/// \param arg3
/// \param d
static inline void gf2_vector_add(gf2 *arg1,
                                  const gf2 *arg2,
                                  const gf2 *arg3,
                                  const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] ^ arg3[i];
    }
}


/// arg1 = scalar*arg2
/// \param arg1
/// \param arg2
/// \param value
/// \param d
static inline void gf2_vector_scalar(gf2 *arg1,
                                     const gf2 *arg2,
                                     const gf2 value,
                                     const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] & (-value);
    }
}

/// arg1 += scalar*arg2
/// \param arg1
/// \param arg2
/// \param value
/// \param d
static inline void gf2_vector_scalar_add(gf2 *arg1,
                                         const gf2 *arg2,
                                         const gf2 value,
                                         const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] ^= arg2[i] & (-value);
    }
}

/// arg1 = arg2 + scalar*arg3
/// \param arg1
/// \param arg2
/// \param scalar
/// \param arg3
/// \param d
static inline void gf2_vector_scalar_add_v2(gf2 *arg1,
                                            const gf2 *arg2,
                                            const gf2 scalar,
                                            const gf2 *arg3,
                                            const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] ^ (arg3[i] & (-scalar));
    }
}

/// \return arg1(value)
static inline gf2 gf2_vector_eval(const gf2 *arg1,
                                  const gf2 value,
                                  const uint32_t d) {
	gf2 ret = 0;
	for (int i = ((d+7)/8)-1; i>=0; i--) {
        const gf2 t = gf2_vector_get(arg1, i);
		ret ^= gf2_mul(value, t);
	}

	return ret;
}

#ifdef USE_AVX2

/// out = in1 + in2
/// \param out[out]
/// \param in1[in]
/// \param in2[in]
/// \param d number of
static inline void gf2_vector_add_u256(gf2 *out,
                                       const gf2 *in1,
                                       const gf2 *in2,
                                       const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;

    // avx2 code
    while (nbytes >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        nbytes -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(nbytes >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        nbytes -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    for (; nbytes > 0; --nbytes) {
        *out++ = *in1++ ^ *in2++;
    }
}

/// out = scalar*in1
/// \param out
/// \param in1
/// \param scalar
/// \param d
static inline void gf2_vector_scalar_u256(gf2 *out,
                                          const gf2 *in1,
                                          const gf2 scalar,
                                          const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;
    const uint8_t m = -scalar;
    const __m256i s256 = _mm256_set1_epi8(m);
    const __m128i s128 = _mm_set1_epi8(m);

    // avx2 code
    while (nbytes >= 32u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = t1&s256;
        _mm256_storeu_si256((__m256i *)out, t2);

        nbytes -= 32u;
        in1 += 32u;
        out += 32u;
    }

    // sse code
    while(nbytes >= 16u) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)in1);
        const __m128i t2 = t1&s128;
        _mm_storeu_si128((__m128i *)out, t2);

        nbytes -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; nbytes > 0; --nbytes) {
        *out++ = *in1++ & m;
    }
}

/// out += scalar*in1
/// \param out
/// \param in1
/// \param scalar
/// \param d
static inline void gf2_vector_scalar_add_u256(gf2 *out,
                                              const gf2 *in1,
                                              const gf2 scalar,
                                              const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;
    const uint8_t m = -scalar;
    const __m256i s256 = _mm256_set1_epi8(m);
    const __m128i s128 = _mm_set1_epi8(m);

    // avx2 code
    while (nbytes >= 32u) {
        const __m256i to = _mm256_loadu_si256((const __m256i *)out);
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = to ^ (t1&s256);
        _mm256_storeu_si256((__m256i *)out, t2);

        nbytes -= 32u;
        in1 += 32u;
        out += 32u;
    }

    // sse code
    while(nbytes >= 16u) {
        const __m128i to = _mm_loadu_si128((const __m128i *)out);
        const __m128i t1 = _mm_loadu_si128((const __m128i *)in1);
        const __m128i t2 = to ^ (t1&s128);
        _mm_storeu_si128((__m128i *)out, t2);

        nbytes -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; nbytes > 0; --nbytes) {
        *out++ ^= *in1++ & m;
    }
}


/// out = in1 + scalar*in2
/// \param out[out]
/// \param in1
/// \param scalar
/// \param in2
/// \param d
static inline void gf2_vector_scalar_add_u256_v2(gf2 *out,
                                                 const gf2 *in1,
                                                 const gf2 scalar,
                                                 const gf2 *in2,
                                                 const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;
    const uint8_t m = -scalar;
    const __m256i s256 = _mm256_set1_epi8(m);
    const __m128i s128 = _mm_set1_epi8(m);

    // avx2 code
    while (nbytes >= 32u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)in2);
        const __m256i t3 = t1 ^ (t2&s256);
        _mm256_storeu_si256((__m256i *)out, t3);

        nbytes -= 32u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // sse code
    while(nbytes >= 16u) {
        const __m128i t1 = _mm_loadu_si128((const __m128i *)in1);
        const __m128i t2 = _mm_loadu_si128((const __m128i *)in2);
        const __m128i t3 = t1 ^ (t2&s128);
        _mm_storeu_si128((__m128i *)out, t3);

        nbytes -= 16u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    for (; nbytes > 0; --nbytes) {
        *out++ = *in1++ ^ (*in2++ & m);
    }
}

/// \return in1(scalar)
static inline gf2 gf2_vector_eval_u256(const gf2 *in1,
                                       const gf2 scalar, 
                                       const uint32_t d) {
    uint32_t nbytes = (d + 7)/8;
    const uint8_t m = -scalar;
    const __m256i s256 = _mm256_set1_epi8(m);

    __m256i acc = _mm256_setzero_si256();
    // avx2 code
    while (nbytes >= 32u) {
        const __m256i t1 = _mm256_loadu_si256((const __m256i *)in1);
        const __m256i t2 = t1&s256;
        acc ^= t2;

        nbytes -= 32u;
        in1 += 32u;
    }

    
    if (nbytes) {
        uint8_t tmp[32] __attribute__((aligned(32))) = {0};
        for (uint32_t i = 0; i < nbytes; i++) { tmp[i] = in1[i]; }

        __m256i t1 = _mm256_load_si256((const __m256i *)tmp);
        const __m256i t2 = t1&s256;
        acc ^= t2;
    }
    
    return gf2_hadd_u256(acc);
}

#endif
