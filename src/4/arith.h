#pragma once

#include <stdint.h>
#include <stddef.h>

#include "../helper.h"

/// GF(4) field representation with X^2 + X + 1 as the irreducible polynomial
/// Field elements: {0, 1, X, X+1}, encoded as 2-bit values {0, 1, 2, 3}
typedef uint8_t ff_t;
typedef uint8_t gf4;

/// Multiplication lookup table for GF(4)
/// Table stores values for {b*X^0, b*X^1} for each possible value of b
static const uint8_t gf4_mult_table[] = {0x00,0x00,0x01,0x02,0x02,0x03,0x03,0x01};

/// Addition in GF(4)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b in GF(4) (implemented as XOR)
static inline gf4 gf4_add(const gf4 a, const gf4 b) { return a ^ b; }

/// Subtraction in GF(4)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b in GF(4) (same as addition since we're in characteristic 2)
static inline gf4 gf4_sub(const gf4 a, const gf4 b) { return a ^ b; }

/// Multiplication in GF(4) using lookup table
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(4)
static inline gf4 gf4_mul(const gf4 a, const gf4 b) {
    return ((a&1u )*gf4_mult_table[b*2 + 0]) ^
           ((a>>1u)*gf4_mult_table[b*2 + 1]);
}

/// Alternative multiplication in GF(4) without using multiplication operations
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(4)
static inline
gf4 gf4_mul_v2(const gf4 a,
               const gf4 b) {
    const gf4 tmpl1 = a & b;
    const gf4 tmpl2 = ((tmpl1 ^ (tmpl1 >> 1)) & 1u);

    gf4 tmph1 = (((a<<1) & b) ^ ((b<<1) & a));
    gf4 tmph3 = (tmpl1 ^ tmph1) & 2u;
    return  tmpl2 ^ tmph3;
}

/// Negation in GF(4)
/// \param a[in]: element to negate
/// \return -a = a (characteristic 2)
static inline gf4 gf4_neg(const gf4 a) { return a; }

/// Multiplicative inverse in GF(4): a^-1 = a^2 (= a^(4-2)), inv(0) = 0
/// squaring is the frobenius: (a1*X + a0)^2 = a1*X + (a0 + a1)
/// \param a[in]: element to invert
/// \return a^-1 in GF(4)
static inline gf4 gf4_inv(const gf4 a) {
    return (a & 2u) | ((a ^ (a >> 1u)) & 1u);
}

/// Vectorized multiplication of 4 GF(4) elements packed in a byte
/// \param a[in]: first vector of 4 GF(4) elements
/// \param b[in]: second vector of 4 GF(4) elements
/// \return vector of a*b in each 2-bit position
static inline uint8_t gf4v_mul_u8(const uint8_t a,
                                  const uint8_t b) {
    const uint8_t and_ = a & b;
    const uint8_t l = ((uint8_t)(a << 1) & b) ^ (a & (uint8_t)(b << 1));
    return ((and_ ^ (and_ >> 1u)) & 0x55u) ^ ((l ^ and_) & 0xAAu);
}

/// Vectorized addition of 32 GF(4) elements packed in 64-bit integers
/// \param a[in]: first vector of 32 GF(4) elements
/// \param b[in]: second vector of 32 GF(4) elements
/// \return vector of a+b in each 2-bit position
static inline uint64_t gf4v_add_u64(const uint64_t a, const uint64_t b) { return a ^ b; }

/// Vectorized subtraction of 32 GF(4) elements packed in 64-bit integers
/// \param a[in]: vector of 32 GF(4) minuends
/// \param b[in]: vector of 32 GF(4) subtrahends
/// \return vector of a-b in each 2-bit position
static inline uint64_t gf4v_sub_u64(const uint64_t a, const uint64_t b) { return a ^ b; }

/// NOTE: not constant time.
/// Vectorized scalar multiplication of 32 GF(4) elements by a single value
/// \param a[in]: vector of 32 GF(4) elements
/// \param b[in]: scalar GF(4) element to multiply by
/// \return vector of a*b in each 2-bit position
static inline uint64_t gf4v_scalar_u64(const uint64_t a, const gf4 b) {
	return (( a&0x5555555555555555    )*gf4_mult_table[b*2 + 0]) ^
		   (((a&0xaaaaaaaaaaaaaaaa)>>1)*gf4_mult_table[b*2 + 1]);
}

/// Vectorized multiplication of 32 GF(4) elements packed in 64-bit integers
/// \param a[in]: first vector of 32 GF(4) elements
/// \param b[in]: second vector of 32 GF(4) elements
/// \return vector of a*b in each 2-bit position
static inline
uint64_t gf4v_mul_u64(const uint64_t a,
					  const uint64_t b) {
    const uint64_t mask = 0b11;
    uint64_t ret = 0;
    for (uint32_t i = 0; i < 32u; ++i) {
        const gf4 a1 = (a >> (i*2u))&mask;
        const gf4 b1 = (b >> (i*2u))&mask;
        const gf4 t1 = gf4_mul(a1, b1);
        const uint64_t t2 = ((uint64_t)t1) << (i*2u);
        ret ^= t2;
    }
    return ret;
}

/// \param a[in]: GF(4) element
/// \return byte containing `a` in all four 2-bit positions
static inline uint8_t gf4_broadcast_u8(const gf4 a) {
    return (a & 3u) * 0x55u;
}

/// horizontal addition of all 2-bit elements of a 64-bit integer
/// \param a[in]: vector of 32 GF(4) elements
/// \return sum of all elements
static inline gf4 gf4_hadd_u64(uint64_t a) {
    a ^= a >> 32u;
    a ^= a >> 16u;
    a ^= a >> 8u;
    a ^= a >> 4u;
    a ^= a >> 2u;
    return a & 3u;
}

#ifdef USE_AVX2
#include <immintrin.h>

/// AVX2 vectorized addition of 128 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a+b in each 2-bit position
static inline
__m256i gf4v_add_u256(const __m256i a,
                      const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// AVX2 vectorized subtraction of 128 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a-b in each 2-bit position
static inline
__m256i gf4v_sub_u256(const __m256i a,
                      const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// AVX2 vectorized multiplication of 128 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a*b in each 2-bit position
static inline
__m256i gf4v_mul_u256(const __m256i a,
				      const __m256i b) {
    const __m256i mask1 = _mm256_set1_epi8(0b01010101);
    const __m256i mask2 = _mm256_set1_epi8((char)0b10101010);

    const __m256i and_ = a & b;
    const __m256i and2 = (a<<1) & b;
    const __m256i and1 = a & (b<<1);
    const __m256i l = and1 ^ and2;

    __m256i ret = (and_  ^ (and_ >> 1u)) & mask1;
    ret ^= (l ^ and_) & mask2;
    return ret;
}

/// AVX2 vectorized scalar multiplication of GF(4) elements by a single value
/// \param a[in]: vector of GF(4) elements
/// \param b[in]: scalar GF(4) element to multiply by
/// \return vector of a*b in each 2-bit position
static inline
__m256i gf4v_scalar_u256(const __m256i a,
						 const gf4 b) {
	const __m256i m0 = _mm256_set1_epi8(0x55);
	const __m256i m1 = _mm256_set1_epi8((char)0xaa);
	const __m256i s0 = _mm256_set1_epi16(gf4_mult_table[b*2 + 0]);
	const __m256i s1 = _mm256_set1_epi16(gf4_mult_table[b*2 + 1]);
	__m256i r0, r1;

	// each 16-bit lane contains 8 elements, the products of the single bits
	// with the (2-bit) scalars do not overlap and therefore do not carry.
	r0 = _mm256_and_si256(a, m0);
	r1 = _mm256_and_si256(a, m1);
	r1 = _mm256_srli_epi16(r1, 1);
	r0 = _mm256_mullo_epi16(r0, s0);
	r1 = _mm256_mullo_epi16(r1, s1);
	r0 = _mm256_xor_si256(r0, r1);
	return r0;
}

/// horizontal addition of all 2-bit elements
/// \param a[in]: vector of 128 GF(4) elements
/// \return sum of all elements
static inline gf4 gf4_hadd_u256(const __m256i a) {
    const __m128i t = _mm_xor_si128(_mm256_castsi256_si128(a),
                                    _mm256_extracti128_si256(a, 1));
    return gf4_hadd_u64((uint64_t)_mm_cvtsi128_si64(t) ^ (uint64_t)_mm_extract_epi64(t, 1));
}

#elif defined(USE_NEON)
#include <arm_neon.h>

/// NEON vectorized addition of 64 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a+b in each 2-bit position
static inline
uint8x16_t gf4v_add_u128(const uint8x16_t a,
                         const uint8x16_t b) {
    return veorq_u8(a, b);
}

/// NEON vectorized subtraction of 64 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a-b in each 2-bit position
static inline
uint8x16_t gf4v_sub_u128(const uint8x16_t a,
                         const uint8x16_t b) {
    return veorq_u8(a, b);
}

/// NEON vectorized multiplication of 64 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a*b in each 2-bit position
static inline
uint8x16_t gf4v_mul_u128(const uint8x16_t a,
                         const uint8x16_t b) {
    const uint8x16_t mask1 = vdupq_n_u8(0x55);
    const uint8x16_t mask2 = vdupq_n_u8(0xaa);

    const uint8x16_t and_ = vandq_u8(a, b);
    const uint8x16_t and2 = vandq_u8(vshlq_n_u8(a, 1), b);
    const uint8x16_t and1 = vandq_u8(a, vshlq_n_u8(b, 1));
    const uint8x16_t l = veorq_u8(and1, and2);

    const uint8x16_t r0 = vandq_u8(veorq_u8(and_, vshrq_n_u8(and_, 1)), mask1);
    const uint8x16_t r1 = vandq_u8(veorq_u8(l, and_), mask2);
    return veorq_u8(r0, r1);
}

/// horizontal addition of all 2-bit elements
/// \param a[in]: vector of 64 GF(4) elements
/// \return sum of all elements
static inline gf4 gf4_hadd_u128(const uint8x16_t a) {
    const uint64x2_t t = vreinterpretq_u64_u8(a);
    return gf4_hadd_u64(vgetq_lane_u64(t, 0) ^ vgetq_lane_u64(t, 1));
}

/// NEON vectorized scalar multiplication of GF(4) elements by a single value
/// \param a[in]: vector of GF(4) elements [a_0, ..., a_31], a_i \in GF(4)
/// \param b[in]: scalar GF(4) element
/// \return vector of a*b in each 2-bit position [a_0*b, ..., a_31*b]
static inline
uint8x8_t gf4v_scalar_neon_u64(const uint8x8_t a,
							   const uint8_t b) {
	const uint8x8_t m0 = vdup_n_u8(0x55);
	const uint8x8_t m1 = vdup_n_u8(0xaa);
	const uint8x8_t s0 = vdup_n_u8(gf4_mult_table[b*2 + 0]);
	const uint8x8_t s1 = vdup_n_u8(gf4_mult_table[b*2 + 1]);
	uint8x8_t r0, r1;

	r0 = vand_u8(a, m0);
	r1 = vand_u8(a, m1);
	r1 = vshr_n_u8(r1, 1);
	r0 = vmul_u8(r0, s0);
	r1 = vmul_u8(r1, s1);
	r0 = veor_u8(r0, r1);
	return r0;
}
#endif

#ifdef USE_AVX512
#include <immintrin.h>

/// AVX-512 vectorized addition of 256 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a+b in each 2-bit position
static inline
__m512i gf4v_add_u512(const __m512i a,
                      const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// AVX-512 vectorized subtraction of 256 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a-b in each 2-bit position
static inline
__m512i gf4v_sub_u512(const __m512i a,
                      const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// AVX-512 vectorized multiplication of 256 GF(4) elements
/// \param a[in]: first vector of GF(4) elements
/// \param b[in]: second vector of GF(4) elements
/// \return vector of a*b in each 2-bit position
static inline
__m512i gf4v_mul_u512(const __m512i a,
                      const __m512i b) {
    const __m512i mask1 = _mm512_set1_epi8(0x55);
    const __m512i mask2 = _mm512_set1_epi8((char)0xaa);

    const __m512i and_ = _mm512_and_si512(a, b);
    const __m512i and2 = _mm512_and_si512(_mm512_slli_epi64(a, 1), b);
    const __m512i and1 = _mm512_and_si512(a, _mm512_slli_epi64(b, 1));
    const __m512i l = _mm512_xor_si512(and1, and2);

    const __m512i r0 = _mm512_and_si512(_mm512_xor_si512(and_, _mm512_srli_epi64(and_, 1)), mask1);
    const __m512i r1 = _mm512_and_si512(_mm512_xor_si512(l, and_), mask2);
    return _mm512_xor_si512(r0, r1);
}

/// horizontal addition of all 2-bit elements
/// \param a[in]: vector of 256 GF(4) elements
/// \return sum of all elements
static inline gf4 gf4_hadd_u512(const __m512i a) {
    const __m256i t1 = _mm256_xor_si256(_mm512_castsi512_si256(a),
                                        _mm512_extracti64x4_epi64(a, 1));
    const __m128i t2 = _mm_xor_si128(_mm256_castsi256_si128(t1),
                                     _mm256_extracti128_si256(t1, 1));
    return gf4_hadd_u64((uint64_t)_mm_cvtsi128_si64(t2) ^ (uint64_t)_mm_extract_epi64(t2, 1));
}
#endif
