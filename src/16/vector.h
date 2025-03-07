#pragma once

#include <string.h>
#include <stdlib.h>

#include "arith.h"

/// \return v[i]
gf16 gf16_vector_get(const uint8_t *v,
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
void gf16_vector_set(uint8_t *v,
                     const uint32_t i,
                     const uint8_t a) {
    const uint32_t pos = i >> 1;
    const uint8_t b = a & 0xF;
    if (i & 1u) {
        v[pos] = (v[pos] & 0xF ) ^ (b << 4);
    } else {
        v[pos] = (v[pos] & 0xF0) ^ b;
    }
}

/// \param v
/// \param n
static inline void gf16_vector_print(gf16 *v,
                                     const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const gf16 t = gf16_vector_get(v, i);
        printf("%2d ", (uint32_t )t);
    }
    printf("\n");
}

/// \param n
/// \return
static inline gf16* gf16_vector_alloc(const uint32_t n) {
    return (gf16 *)malloc(((n+1)/2)*sizeof(gf16));
}

/// \param v
/// \param n
static inline void gf16_vector_zero(gf16 *v,
                                    const uint32_t n) {
    memset(v, 0, ((n + 1)/2) *sizeof(gf16));
}

/// \param v
/// \param n
static inline void gf16_vector_random(gf16 *v,
                                      const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        v[i] = rand() & 0xFF;
    }
}

/// \param v1
/// \param v2
/// \param n
static inline void gf16_vector_copy(gf16 *__restrict__ v1,
                                    const gf16 *__restrict__ v2,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        v1[i] = v2[i];
    }
}

/// \param out += in
/// \param in
/// \param n
static inline void gf16_vector_add(gf16 *out,
                                   const gf16 *in,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        out[i] ^= in[i];
    }
}

#ifdef USE_AVX2 

/// poly1 = poly2 + poly3
/// simd addition: automatically chooses the fastes code path (avx, sse, or normal)
/// d number of elements not bytes
static inline
void gf16_vector_add_u256_(ff_t *poly1,
                           const ff_t *poly2,
                           const ff_t *poly3,
                           const uint32_t d) {
	uint32_t i = d/2u;
	// avx2 code
	while (i >= 32u) {
		_mm256_storeu_si256((__m256i *)poly1, 
				_mm256_loadu_si256((__m256i *)poly2) ^ 
				_mm256_loadu_si256((__m256i *)poly3));
		i -= 32u;
		poly1 += 32u;
		poly2 += 32u;
		poly3 += 32u;
	}

	// sse code
	while(i >= 16u) {
		_mm_storeu_si128((__m128i *)poly1, 
				_mm_loadu_si128((__m128i *)poly2) ^
				_mm_loadu_si128((__m128i *)poly3));
		poly1 += 16u;
		poly2 += 16u;
		poly3 += 16u;
		i -= 16;
	}

	for(uint32_t j = 0; j<i; j++) { 
		poly1[j] = poly2[j] ^ poly3[j]; 
	}
}

/// d = number of elements not bytes
/// poly1 += poly2 
static inline
void gf16_vector_add_u256(ff_t *poly1, 
                          const ff_t *poly2,
                          const uint32_t d) {
	gf16_vector_add_u256_(poly1, poly1, poly2, d);
}

/// d = number of elements not bytes
/// poly1 =  scalar * poly3
static inline
void gf16_vector_multiple_u256(ff_t *poly1, 
                               const ff_t scalar, 
                               const ff_t *poly2,
                               const uint32_t d) {
	uint32_t i = d/2;
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
        uint8_t tmp[32] __attribute__((aligned(32)));
        for (uint32_t j = 0; j<i; j++) { tmp[j] = poly2[j]; }
        const __m256i p3 = _mm256_load_si256((__m256i *)tmp);
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p3, xf);
        _mm256_store_si256((__m256i *)tmp, t1);
        for (uint32_t j = 0; j<i; j++) { poly1[j] = tmp[j]; }
    }
}

/// d = number of elements not bytes
/// poly1 = poly2 + scalar * poly3
static inline
void gf16_vector_add_multiple_u256_(ff_t *poly1, 
                                    const ff_t *poly2,
                                    const ff_t scalar, 
                                    const ff_t *poly3,
                                    const uint32_t d) {
	uint32_t i = d/2;
    const __m256i ml = gf16v_tbl32_multab(scalar);
    const __m256i mh = _mm256_slli_epi16(ml, 4);
    const __m256i xf = _mm256_set1_epi8(0xF);

	// avx2 code
	while (i >= 32u) {
        const __m256i p3 = _mm256_loadu_si256((__m256i *)poly3);
        const __m256i p2 = _mm256_loadu_si256((__m256i *)poly2);
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p3, xf);
        const __m256i t2 = p2 ^ t1;
        _mm256_storeu_si256((__m256i *)poly1, t2);
		i -= 32u;
		poly1 += 32u;
		poly2 += 32u;
		poly3 += 32u;
	}


    if (i) {
        uint8_t tmp[32] __attribute__((aligned(32)));
        for (uint32_t j = 0; j<i; j++) { tmp[j] = poly3[j]; }
        const __m256i p3 = _mm256_load_si256((__m256i *)tmp);
        const __m256i t1 = linear_transform_8x8_256b(ml, mh, p3, xf);

        for (uint32_t j = 0; j<i; j++) { tmp[j] = poly2[j]; }
        const __m256i p2 = _mm256_load_si256((__m256i *)tmp);
        const __m256i t2 = p2 ^ t1;
        _mm256_store_si256((__m256i *)tmp, t2);
        
        for (uint32_t j = 0; j<i; j++) { poly1[j] = tmp[j]; }
    }
}

/// d = number of elements not bytes
/// poly1 ^= scalar * poly2
static inline
void gf16_vector_add_multiple_u256(ff_t *poly1,
                                   const ff_t scalar,
                                   const ff_t *poly2,
                                   const uint32_t d) {
	gf16_vector_add_multiple_u256_(poly1, poly1, scalar, poly2, d);
}
#endif 
