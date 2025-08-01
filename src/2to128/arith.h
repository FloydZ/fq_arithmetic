#pragma once

#include <stdint.h>
#include <stdio.h>

#include "../2/arith.h"

/// GF(2^128) field representation using polynomial basis
/// The generic base type is `__uint128_t` called a `limb`. Each `limb` can store
/// up to 128 bits or coordinates mod 2.
/// Each field element is represented as a 128-bit unsigned integer
typedef __uint128_t gf2to128;

/// The irreducible polynomial used to define the field
/// Two options are provided (uncomment the one you prefer):
/// 
/// Option 1: Using irreducible polynomial x^128 + x^7 + x^2 + x + 1
#define MODULUS 0b10000111
/// Option 2: Using irreducible polynomial x^128 + x^7 + 1
///#define MODULUS 0b10000001

/// Addition in GF(2^128)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b in GF(2^128), which is a XOR b since characteristic is 2
static inline
gf2to128 gf2to128_add(const gf2to128 a, 
                      const gf2to128 b) {
	return a ^ b;
}

/// Subtraction in GF(2^128)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b in GF(2^128), which is the same as a+b (XOR) since we're in characteristic 2
static inline
gf2to128 gf2to128_sub(const gf2to128 a, 
                      const gf2to128 b) {
	return a ^ b;
}



/// Multiplication in GF(2^128) - constant time version
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^128)
/// \note This implementation is constant-time (resistant to timing attacks)
/// \note Implements multiplication using bit-masking to avoid branches
static inline
gf2to128 gf2to128_mul(const gf2to128 a,
                      const gf2to128 b) {
    gf2to128 r = (-(b>>127u) & a);
    for (int i = 126; i >= 0; --i) {
        r = (-(b>>i & 1u) & a) ^ (-(r>>127) & MODULUS) ^ (r+r);
    }
    return r;
}

/// Multiplication in GF(2^128) - non-constant time version
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^128)
/// \note This implementation is NOT constant-time (vulnerable to timing attacks)
/// \note Uses the shift-and-add algorithm with conditional reduction by the modulus
static inline
gf2to128 gf2to128_mul_v2(const gf2to128 a,
                         const gf2to128 b) {
    gf2to128 shifted=a, result=0;

    for (size_t i = 0; i < 2; ++i) {
        const uint64_t bb = (b >> (i*64));
        for (size_t j = 0; j < 64; ++j) {
            if (bb & (1ull << j)) {
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

/// Multiplication of GF(2^128) element by GF(2) element
/// \param a[in]: GF(2^128) element
/// \param b[in]: GF(2) element (0 or 1)
/// \return a*b in GF(2^128)
/// \note Optimized for the case when the second operand is from GF(2)
static inline
gf2to128 gf2to128_mul_gf2(const gf2to128 a,
                          const gf2 b) {
    const gf2to128 c = ((gf2to128)-1ull)*b;
    return a & c;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// Expand 8 GF(2) elements into 4 256-bit vectors for vectorized operations
/// \param out[out]: Output array of 4 256-bit vectors
/// \param in[in]: Byte containing 8 GF(2) elements (bits)
/// \note Distributes the 8 bits from the input byte across 4 256-bit vectors
/// \note Must have 4 registers allocated for output
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

/// Expand 2 GF(2) elements into a 256-bit vector
/// \param in[in]: Byte containing at least 2 GF(2) elements (bits)
/// \return 256-bit vector with format [0, in_1, 0, in_0]
/// \note Extracts the lowest 2 bits from the input and places them in specific positions
static inline __m256i gf2to128v_expand_gf2_x2_u256(const uint8_t in) {
    const __m256i mask = _mm256_setr_epi32(0x0, 0x2, 0x2, 0x2, 0x1, 0x2, 0x2, 0x2);
    const uint64_t t1 = _pdep_u64(in, 0x100000001);
    const __m256i t2 = _mm256_setr_epi64x(t1, 0, 0, 0);
    const __m256i t3 = _mm256_permutevar8x32_epi32(t2, mask);
    return t3;
}

/// Expand 1 GF(2) element into a GF(2^128) element
/// \param in[in]: Byte containing at least 1 GF(2) element (bit)
/// \return GF(2^128) element with the value 0 or 1
/// \note Extracts the lowest bit from the input
static inline
gf2to128 gf2to128v_expand_gf2_x1(const uint8_t in) {
    return (gf2to128)(in & 1u);
}

/// Multiplication in GF(2^128) using PCLMULQDQ instruction
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^128)
/// \note Uses hardware acceleration via PCLMULQDQ for carry-less multiplication
/// \note Works for any modulus defined by MODULUS
/// \note More efficient than the software implementations
static inline
gf2to128 gf2to128_mul_u128(const gf2to128 a,
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

/// Vectorized multiplication of 2 GF(2^128) elements
/// \param a[in]: 256-bit vector containing 2 GF(2^128) elements
/// \param b[in]: 256-bit vector containing 2 GF(2^128) elements
/// \return 256-bit vector with element-wise products
/// \note Uses PCLMULQDQ instruction for carry-less multiplication
/// \note Processes 2 multiplications in parallel using AVX2 and CLMUL instructions
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

/// Alternative multiplication in GF(2^128) using PCLMULQDQ and manual reduction
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^128)
/// \note IMPORTANT: Only works with MODULUS 0b10000111 (x^128 + x^7 + x^2 + x + 1)
/// \note Roughly 5 times slower than gf2to128_mul_u128 (Intel N97 benchmarks):
///       BM_gf2to128_mul_avx        40.2 ns         40.1 ns     16424693 cycles=19.9797
///       BM_gf2to128_mul_avx2       80.7 ns         80.6 ns      8657807 cycles=99.4822
/// \note Uses a different approach to polynomial reduction than gf2to128_mul_u128
static inline
gf2to128 gf2to128_mul_u128_slow(const gf2to128 a,
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


/// Vectorized multiplication of GF(2^128) elements by GF(2) elements
/// \param a[in]: 256-bit vector containing GF(2^128) elements
/// \param b[in]: 256-bit vector containing GF(2) elements (each as a single bit)
/// \return 256-bit vector with element-wise products
/// \note Optimized for the case when the second operand is from GF(2)
/// \note The GF(2) elements in b are not compressed - each is a single bit
static inline __m256i gf2to128v_mul_gf2_u256(const __m256i a,
                                             const __m256i b) {
    const __m256i m1 = _mm256_set1_epi16(-1);
    const __m256i t1 = _mm256_sign_epi16(b, m1);
    return a & t1;
}

/// Multiplication of GF(2^128) element by GF(2) element using SSE instructions
/// \param a[in]: GF(2^128) element
/// \param b[in]: GF(2) element (as a 128-bit value with a single bit set)
/// \return a*b in GF(2^128)
/// \note Optimized version using SSE instructions
/// \note The GF(2) element in b is not compressed - it's a single bit in a 128-bit value
static inline gf2to128 gf2to128v_mul_gf2_u128(const gf2to128 a,
                                              const gf2to128 b) {
    const __m128i m1 = _mm_set1_epi16(-1);
    const __m128i t1 = _mm_sign_epi16((__m128i)b, m1);
    return (gf2to128)(a & (gf2to128)t1);
}

/// Vectorized multiplication of GF(2^128) elements by expanded GF(2) elements
/// \param a[in]: 256-bit vector containing GF(2^128) elements
/// \param b[in]: 256-bit vector containing expanded GF(2) elements
/// \return 256-bit vector with element-wise products
/// \note Optimized version for when b contains expanded GF(2) elements
/// \note In this case, the GF(2) elements in b are already expanded to full 128-bit masks
static inline __m256i gf2to128v_mul_gf2_u256_v2(const __m256i a,
                                                const __m256i b) {
    return a & b;
}

/// Multiplication of GF(2^128) element by expanded GF(2) element
/// \param a[in]: 128-bit vector containing a GF(2^128) element
/// \param b[in]: 128-bit vector containing an expanded GF(2) element
/// \return 128-bit vector with the product
/// \note Simplified version that uses AND when b is already expanded
/// \note The GF(2) element in b must be pre-expanded to a full 128-bit mask
static inline __m128i gf2to128v_mul_gf2_u128_v2(const __m128i a,
                                                const __m128i b) {
    return a & b;
}



#elif defined(USE_NEON)
#include <arm_neon.h>

/// TODO test
/// Multiplication in GF(2^128) using ARM PMULL instruction
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^128)
/// \note Uses PMULL instruction for carry-less multiplication
/// \note Requires ARMv8 with Crypto extension
/// \note Works for any modulus defined by MODULUS
static inline
gf2to128 gf2to128_mul_u128_neon(const gf2to128 a,
                               const gf2to128 b) {
    // Load modulus into a 128-bit vector
    uint32x4_t modulus_v = vdupq_n_u32(0);
    modulus_v = vsetq_lane_u32(MODULUS, modulus_v, 0);
    
    // Split inputs into 64-bit halves
    uint64x2_t a_v = vreinterpretq_u64_u128(a);
    uint64x2_t b_v = vreinterpretq_u64_u128(b);
    
    uint64_t a_lo = vgetq_lane_u64(a_v, 0);
    uint64_t a_hi = vgetq_lane_u64(a_v, 1);
    uint64_t b_lo = vgetq_lane_u64(b_v, 0);
    uint64_t b_hi = vgetq_lane_u64(b_v, 1);
    
    // Compute the 256-bit result of a * b using four 64x64->128 bit multiplications
    
    // Low 64 bits of a * low 64 bits of b
    poly128_t mul_lo_lo = vmull_p64((poly64_t)a_lo, (poly64_t)b_lo);
    uint64x2_t mul256_low = vreinterpretq_u64_p128(mul_lo_lo);
    
    // High 64 bits of a * high 64 bits of b
    poly128_t mul_hi_hi = vmull_p64((poly64_t)a_hi, (poly64_t)b_hi);
    uint64x2_t mul256_high = vreinterpretq_u64_p128(mul_hi_hi);
    
    // Low 64 bits of a * high 64 bits of b
    poly128_t mul_lo_hi = vmull_p64((poly64_t)a_lo, (poly64_t)b_hi);
    uint64x2_t mul256_mid1 = vreinterpretq_u64_p128(mul_lo_hi);
    
    // High 64 bits of a * low 64 bits of b
    poly128_t mul_hi_lo = vmull_p64((poly64_t)a_hi, (poly64_t)b_lo);
    uint64x2_t mul256_mid2 = vreinterpretq_u64_p128(mul_hi_lo);
    
    // Add the 4 terms together (XOR for GF(2^128))
    uint64x2_t mul256_mid = veorq_u64(mul256_mid1, mul256_mid2);
    
    // lower 64 bits of mid don't intersect with high, and upper 64 bits don't intersect with low
    uint64x2_t high_adj = vextq_u64(vdupq_n_u64(0), mul256_mid, 1);
    uint64x2_t low_adj = vextq_u64(mul256_mid, vdupq_n_u64(0), 1);
    
    mul256_high = veorq_u64(mul256_high, high_adj);
    mul256_low = veorq_u64(mul256_low, low_adj);
    
    // Done computing mul256_low and mul256_high, time to reduce
    
    // Extract the high 64-bit parts
    uint64_t high_high = vgetq_lane_u64(mul256_high, 1);
    uint64_t high_low = vgetq_lane_u64(mul256_high, 0);
    
    // Reduce w.r.t. high half of mul256_high
    poly128_t tmp_hi = vmull_p64((poly64_t)high_high, (poly64_t)vgetq_lane_u32(modulus_v, 0));
    uint64x2_t tmp_hi_v = vreinterpretq_u64_p128(tmp_hi);
    
    // Adjust mul256_low and mul256_high
    uint64x2_t low_adj_hi = vextq_u64(tmp_hi_v, vdupq_n_u64(0), 1);
    uint64x2_t high_adj_hi = vextq_u64(vdupq_n_u64(0), tmp_hi_v, 1);
    
    mul256_low = veorq_u64(mul256_low, low_adj_hi);
    mul256_high = veorq_u64(mul256_high, high_adj_hi);
    
    // Reduce w.r.t. low half of mul256_high
    poly128_t tmp_lo = vmull_p64((poly64_t)high_low, (poly64_t)vgetq_lane_u32(modulus_v, 0));
    uint64x2_t tmp_lo_v = vreinterpretq_u64_p128(tmp_lo);
    
    // Final result is in mul256_low XOR tmp_lo_v
    mul256_low = veorq_u64(mul256_low, tmp_lo_v);
    
    // Convert back to gf2to128
    return (gf2to128)vreinterpretq_u128_u64(mul256_low);
}

/// Vectorized multiplication of 2 GF(2^128) elements using NEON
/// \param a[in]: vector of 2 GF(2^128) elements [a_0, a_1]
/// \param b[in]: vector of 2 GF(2^128) elements [b_0, b_1]
/// \param result[out]: vector to store element-wise products [a_0*b_0, a_1*b_1]
/// \note Uses PMULL instruction for carry-less multiplication
/// \note Processes 2 multiplications in parallel using NEON
static inline
void gf2to128v_mul_u256_neon(const gf2to128 a[2],
                            const gf2to128 b[2],
                            gf2to128 result[2]) {
    // Multiply each element pair separately
    result[0] = gf2to128_mul_u128_neon(a[0], b[0]);
    result[1] = gf2to128_mul_u128_neon(a[1], b[1]);
}
#endif
