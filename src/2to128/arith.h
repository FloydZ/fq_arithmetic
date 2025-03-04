#include <stdint.h>
#include <stdio.h>

/// Representation:
/// the generic base type is `uint64_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.

typedef __uint128_t eff_t;
typedef __uint128_t gf2to128;

// using irreducible polynomial x^128+x^7+x^2+x+1
// We need only the last word
#define MODULUS 0b10000111

eff_t add(const eff_t a, 
          const eff_t b) {
	return a ^ b;
}

eff_t sub(const eff_t a, 
          const eff_t b) {
	return a ^ b;
}

eff_t mul(const eff_t a,
          const eff_t b) {
    eff_t shifted=a, result=0,m=MODULUS;

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

#ifdef USE_AVX2
#include <immintrin.h>

eff_t mul_avx(const eff_t a,
              const eff_t b) {
    const __m128i a_ = _mm_loadu_si128((const __m128i*) &(a));
    const __m128i b_ = _mm_loadu_si128((const __m128i*) &(b)); 
    const __m128i modulus = _mm_setr_epi32(MODULUS, 0, 0, 0);

    /* compute the 256-bit result of a * b with the 4x64-bit multiplication
       intrinsic */
    __m128i mul256_high = _mm_clmulepi64_si128(a, b, 0x11); /* high of both */
    __m128i mul256_low = _mm_clmulepi64_si128(a, b, 0x00); /* low of both */

    __m128i mul256_mid1 = _mm_clmulepi64_si128(a, b, 0x01); /* low of a, high of b */
    __m128i mul256_mid2 = _mm_clmulepi64_si128(a, b, 0x10); /* high of a, low of b */

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
    return (eff_t)mul256_low;
}

/// \param []
__m256i gf2to128v_mul_u256(const __m256i a, 
                           const __m256i b) {
#ifdef USE_AVX512 

#else
    __m256i ret;
    const __m128i modulus = _mm_setr_epi32(MODULUS, 0, 0, 0);
    const __m128i a1 = _mm256_castsi256_si128(a);
    const __m128i a2 = _mm256_extracti128_si256(a, 0x1);
    const __m128i b1 = _mm256_castsi256_si128(b);
    const __m128i b2 = _mm256_extracti128_si256(a, 0x1);

    __m128i m1_high = _mm_clmulepi64_si128(a1, b1, 0x11); // high of both
    __m128i m2_high = _mm_clmulepi64_si128(a2, b2, 0x11); // high of both
    __m128i m1_low  = _mm_clmulepi64_si128(a1, b1, 0x00); // low of both
    __m128i m2_low  = _mm_clmulepi64_si128(a2, b2, 0x00); // low of both
    __m128i m1_mid  = _mm_clmulepi64_si128(a1, b1, 0x01); // low of a, high of b
    __m128i m2_mid  = _mm_clmulepi64_si128(a2, b2, 0x01); // low of a, high of b
    m2_mid ^= _mm_clmulepi64_si128(a2, b2, 0x10); // high of a, low of b
    m1_mid ^= _mm_clmulepi64_si128(a1, b1, 0x10); // high of a, low of b
    
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
#endif
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
eff_t mul_avx2(const eff_t a,
               const eff_t b) {
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

	return *((eff_t*)tmp);
}
#elif defined(USE_NEON)
#else
#error "not implemented"
#endif
