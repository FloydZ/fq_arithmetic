#pragma once

#include <stdint.h>
#include <stdio.h>

#include "../2/arith.h"

/// Representation:
/// the generic base type is `uint64_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.

typedef __uint128_t gf2to128;

// using irreducible polynomial x^128+x^7+x^2+x+1
// We need only the last word
#define MODULUS 0b10000111

/// \param a[in]: 
/// \param b[in]: 
/// \param return:
static inline
gf2to128 gf2to128_add(const gf2to128 a, 
                      const gf2to128 b) {
	return a ^ b;
}

/// \param a[in]: 
/// \param b[in]: 
/// \param return:
static inline
gf2to128 gf2to128_sub(const gf2to128 a, 
                      const gf2to128 b) {
	return a ^ b;
}

/// original correct multiplication
static inline
gf2to128 gf2to128_mul(const gf2to128 a,
                      const gf2to128 b) {
    gf2to128 shifted=a, result=0,m=MODULUS;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            if (b & (1ull << j)) {
                result ^= shifted;
            }

            if (shifted >> 127) {
                shifted <<= 1;
                shifted ^= MODULUS;
            } else {
                shifted <<= 1;
            }
        }
    }

    return result;
}

/// \return a*b
gf2to128 gf2to128_mul_gf2(const gf2to128 a,
                          const gf2 b) {
    gf2to128 c = ((gf2to128)-1ull)*b;
    return a & c;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// \param out[out]: must be 4 registers
/// \param in[in]:
/// \return nothing
static inline void gf2to128v_expand_gf2_x8_u256(__m256i *out,
                                                const uint8_t in) {
    const __m256i mask = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF,0,0xFFFFFFFFFFFFFFFF,0);
    const uint64_t t21 = _pdep_u64(in>>0, 0x0001000100010001);
    const uint64_t t22 = _pdep_u64(in>>4, 0x0001000100010001);

    const __m128i t31 = _mm_set_epi64x(0, t21);
    const __m128i t32 = _mm_set_epi64x(0, t22);

    const __m256i t41 = _mm256_cvtepi16_epi64(t31);
    const __m256i t42 = _mm256_cvtepi16_epi64(t32);

    const __m256i t51 = _mm256_permute4x64_epi64(t41, 0b11011000);
    const __m256i t52 = _mm256_permute4x64_epi64(t42, 0b11011000);

    out[0] = t51&mask;
    out[1] = _mm256_bsrli_epi128(t51, 8);
    out[2] = t52&mask;
    out[3] = _mm256_bsrli_epi128(t52, 8);
}

/// \param in[in]: 2bits
/// \return [0, in_1, 0, in_0];
static inline __m256i gf2to128v_expand_gf2_x2_u256(const uint8_t in) {
    const __m256i mask = _mm256_setr_epi32(0x0, 0x2, 0x2, 0x2, 0x1, 0x2, 0x2, 0x2);
    const uint64_t t1 = _pdep_u64(in, 0x100000001);
    const __m256i t2 = _mm256_setr_epi64x(t1, 0, 0, 0);
    const __m256i t3 = _mm256_permutevar8x32_epi32(t2, mask);
    return t3;
}

/// \param in[in]:
static inline
gf2to128 gf2to128v_expand_gf2_x1(const uint8_t in) {
    return (gf2to128)(in & 1u);
}

static inline
gf2to128 gf2to128_mul_avx2(const gf2to128 a,
                           const gf2to128 b) {
    const __m128i modulus = _mm_setr_epi32(MODULUS, 0, 0, 0);

    /* compute the 256-bit result of a * b with the 4x64-bit multiplication
       intrinsic */
    __m128i mul256_high = _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0x11); /* high of both */
    __m128i mul256_low = _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0x00); /* low of both */

    __m128i mul256_mid1 = _mm_clmulepi64_si128((__m128i)a,(__m128i)b, 0x01); /* low of a, high of b */
    __m128i mul256_mid2 = _mm_clmulepi64_si128((__m128i)a,(__m128i)b, 0x10); /* high of a, low of b */

    /* Add the 4 terms together */
    __m128i mul256_mid = _mm_xor_si128(mul256_mid1, mul256_mid2);
    /* lower 64 bits of mid don't intersect with high, and upper 64 bits don't intersect with low */
    mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(mul256_mid, 8));
    mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(mul256_mid, 8));

    /* done computing mul256_low and mul256_high, time to reduce */

    /* reduce w.r.t. high half of mul256_high */
    __m128i tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x01);
    mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(tmp, 8));
    mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(tmp, 8));

    /* reduce w.r.t. low half of mul256_high */
    tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x00);
    mul256_low = _mm_xor_si128(mul256_low, tmp);
    return (gf2to128)mul256_low;
}

/// \param []
static inline
__m256i gf2to128v_mul_u256(const __m256i a, 
                           const __m256i b) {
    __m256i ret;
    const __m128i modulus = _mm_setr_epi32(MODULUS, 0, 0, 0);
    const __m128i a1 = _mm256_castsi256_si128(a);
    const __m128i a2 = _mm256_extracti128_si256(a, 0x1);
    const __m128i b1 = _mm256_castsi256_si128(b);
    const __m128i b2 = _mm256_extracti128_si256(b, 0x1);

    __m128i m1_high = _mm_clmulepi64_si128(a1, b1, 0x11); // high of both
    __m128i m2_high = _mm_clmulepi64_si128(a2, b2, 0x11); // high of both
    __m128i m1_low  = _mm_clmulepi64_si128(a1, b1, 0x00); // low of both
    __m128i m2_low  = _mm_clmulepi64_si128(a2, b2, 0x00); // low of both
    __m128i m1_mid  = _mm_clmulepi64_si128(a1, b1, 0x01); // low of a, high of b
    __m128i m2_mid  = _mm_clmulepi64_si128(a2, b2, 0x01); // low of a, high of b
    m1_mid ^= _mm_clmulepi64_si128(a1, b1, 0x10); // high of a, low of b
    m2_mid ^= _mm_clmulepi64_si128(a2, b2, 0x10); // high of a, low of b

    m1_high = _mm_xor_si128(m1_high, _mm_srli_si128(m1_mid, 8));
    m2_high = _mm_xor_si128(m2_high, _mm_srli_si128(m2_mid, 8));
    m1_low  = _mm_xor_si128(m1_low,  _mm_slli_si128(m1_mid, 8));
    m2_low  = _mm_xor_si128(m2_low,  _mm_slli_si128(m2_mid, 8));
    
    /* reduce w.r.t. high half of mul256_high */
    __m128i tmp = _mm_clmulepi64_si128(m1_high, modulus, 0x01);
    m1_low  = _mm_xor_si128(m1_low, _mm_slli_si128(tmp, 8));
    m1_high = _mm_xor_si128(m1_high, _mm_srli_si128(tmp, 8));
    tmp = _mm_clmulepi64_si128(m2_high, modulus, 0x01);
    m2_low  = _mm_xor_si128(m2_low, _mm_slli_si128(tmp, 8));
    m2_high = _mm_xor_si128(m2_high, _mm_srli_si128(tmp, 8));

    /* reduce w.r.t. low half of mul256_high */
    tmp = _mm_clmulepi64_si128(m1_high, modulus, 0x00);
    m1_low = _mm_xor_si128(m1_low, tmp);
    tmp = _mm_clmulepi64_si128(m2_high, modulus, 0x00);
    m2_low = _mm_xor_si128(m2_low, tmp);
    ret = _mm256_set_m128i(m2_low, m1_low);
    return ret;
}

/// roughly 5 times slower then the function above (Intel N97:word):
/// -------------------------------------------------------------------------------
/// Benchmark                     Time             CPU   Iterations UserCounters...
/// -------------------------------------------------------------------------------
// BM_gf2to128_mul_avx        40.2 ns         40.1 ns     16424693 cycles=19.9797
// BM_gf2to128_mul_avx2       80.7 ns         80.6 ns      8657807 cycles=99.4822
/// \param a[in]: 
/// \param b[in]: 
/// \return a*b
static inline
gf2to128 gf2to128_mul_slow(const gf2to128 a,
                           const gf2to128 b) {
	uint64_t tmp[4] = {0};
#ifdef USE_AVX512
    *(__m256 *) = _mm256_clmulepi64_epi128(a, b);
#else
	((__m128i   *)(((uint64_t *)tmp) + 0))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b00000);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b00001);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b10000);
	((__m128i   *)(((uint64_t *)tmp) + 2))[0] ^= _mm_clmulepi64_si128((__m128i)a, (__m128i)b, 0b10001);
#endif

    // reduction
	for (uint32_t i = 3; i >=2; i--){
		uint64_t w = tmp[i];
		uint64_t s = (w>>57) ^ (w>>62) ^ (w>>63);
		tmp[i-1] ^= s;
		
		s = w^(w<<1)^(w<<2)^(w<<7);

		tmp[i-2] ^= s;
	}

	return *((gf2to128*)tmp);
}


/// \param a
/// \param b in gf2, not compresses: a single bit in
/// \return
static inline __m256i gf2to128v_mul_gf2_u256(const __m256i a,
                                             const __m256i b) {
    const __m256i m1 = _mm256_set1_epi16(-1);
    const __m256i t1 = _mm256_sign_epi16(b, m1);
    return a & t1;
}

/// \param a
/// \param b in gf2, not compresses: a single bit in
/// \return
static inline __m128i gf2to128v_mul_gf2_u128(const __m128i a,
                                             const __m128i b) {
    const __m128i m1 = _mm_set1_epi16(-1);
    const __m128i t1 = _mm_sign_epi16(b, m1);
    return a & t1;
}

/// \param a
/// \param b in gf2 already expanded
/// \return
static inline __m256i gf2to128v_mul_gf2_u256_v2(const __m256i a,
                                                const __m256i b) {
    return a & b;
}

/// \param a
/// \param b in gf2 already expanded
/// \return
static inline __m128i gf2to128v_mul_gf2_u128_v2(const __m128i a,
                                                const __m128i b) {
    return a & b;
}



#elif defined(USE_NEON)
#else
#error "not implemented"
#endif
