#pragma once

#include <stdint.h>
#include <stddef.h>

/// GF(2) field representation
/// The generic base type is `uint8_t` called a `limb`. Each `limb` can store 
/// up to 8 bits or coordinates mod 2. Field elements are simply 0 and 1.
typedef uint8_t gf2;

/// Single bit operations for GF(2) field arithmetic
/// Addition in GF(2)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b in GF(2) (implemented as XOR)
static gf2 gf2_add(const gf2 a, const gf2 b) { return (a&1u) ^ (b&1u); }

/// Subtraction in GF(2)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b in GF(2) (same as addition since we're in characteristic 2)
static gf2 gf2_sub(const gf2 a, const gf2 b) { return (a&1u) ^ (b&1u); }

/// Multiplication in GF(2)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2) (implemented as AND)
static gf2 gf2_mul(const gf2 a, const gf2 b) { return (a&1u) & (b&1u); }

/// Multiplicative inverse in GF(2)
/// \param a[in]: element to invert
/// \return 1/a in GF(2) (note: 1^-1 = 1, and 0^-1 is undefined)
static gf2 gf2_inv(const gf2 a) { return a; }

/// Vectorized operations for 32 GF(2) elements packed in 32-bit integers
/// Vectorized addition of 32 GF(2) elements packed in 32-bit integers
/// \param a[in]: first vector of 32 GF(2) elements
/// \param b[in]: second vector of 32 GF(2) elements
/// \return vector of a+b in each bit position
static uint32_t gf2v_add_u32(const uint32_t a, const uint32_t b) { return a ^ b; }

/// Vectorized subtraction of 32 GF(2) elements packed in 32-bit integers
/// \param a[in]: vector of 32 GF(2) minuends
/// \param b[in]: vector of 32 GF(2) subtrahends
/// \return vector of a-b in each bit position
static uint32_t gf2v_sub_u32(const uint32_t a, const uint32_t b) { return a ^ b; }

/// Vectorized multiplication of 32 GF(2) elements packed in 32-bit integers
/// \param a[in]: first vector of 32 GF(2) elements
/// \param b[in]: second vector of 32 GF(2) elements
/// \return vector of a*b in each bit position
static uint32_t gf2v_mul_u32(const uint32_t a, const uint32_t b) { return a & b; }

/// Vectorized operations for 64 GF(2) elements packed in 64-bit integers
/// Vectorized addition of 64 GF(2) elements packed in 64-bit integers
/// \param a[in]: first vector of 64 GF(2) elements
/// \param b[in]: second vector of 64 GF(2) elements
/// \return vector of a+b in each bit position
static uint64_t gf2v_add_u64(const uint64_t a, const uint64_t b) { return a ^ b; }

/// Vectorized subtraction of 64 GF(2) elements packed in 64-bit integers
/// \param a[in]: vector of 64 GF(2) minuends
/// \param b[in]: vector of 64 GF(2) subtrahends
/// \return vector of a-b in each bit position
static uint64_t gf2v_sub_u64(const uint64_t a, const uint64_t b) { return a ^ b; }

/// Vectorized multiplication of 64 GF(2) elements packed in 64-bit integers
/// \param a[in]: first vector of 64 GF(2) elements
/// \param b[in]: second vector of 64 GF(2) elements
/// \return vector of a*b in each bit position
static uint64_t gf2v_mul_u64(const uint64_t a, const uint64_t b) { return a & b; }

#ifdef USE_AVX2
#include <immintrin.h>

/// AVX2 vectorized addition of 256 GF(2) elements
/// \param a[in]: first vector of 256 GF(2) elements
/// \param b[in]: second vector of 256 GF(2) elements
/// \return vector of a+b in each bit position
static __m256i gf2v_add_u256(const __m256i a, const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// AVX2 vectorized subtraction of 256 GF(2) elements
/// \param a[in]: vector of 256 GF(2) minuends
/// \param b[in]: vector of 256 GF(2) subtrahends
/// \return vector of a-b in each bit position
static __m256i gf2v_sub_u256(const __m256i a, const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// AVX2 vectorized multiplication of 256 GF(2) elements
/// \param a[in]: first vector of 256 GF(2) elements
/// \param b[in]: second vector of 256 GF(2) elements
/// \return vector of a*b in each bit position
static __m256i gf2v_mul_u256(const __m256i a, const __m256i b) {
	return _mm256_and_si256(a, b);
}

/// AVX2 vectorized scalar multiplication of 256 GF(2) elements by a single value
/// \param a[in]: vector of 256 GF(2) elements
/// \param b[in]: scalar GF(2) element to multiply by
/// \return vector of a*b in each bit position
static __m256i gf2v_scalar_u256(const __m256i a, const gf2 b) {
    const __m256i _b = _mm256_set1_epi8(-b);
    return gf2v_mul_u256(a, _b);
}

/// Horizontal XOR reduction of 256 GF(2) elements to a single element
/// \param in[in]: vector of 256 GF(2) elements
/// \return XOR of all bits in the input vector
static inline gf2 gf2_hadd_u256(const __m256i in) {
    __m256i ret = in;
    ret = ret ^ _mm256_permute2x128_si256(ret, ret, 129); // 0b10000001
    ret = ret ^ _mm256_srli_si256(ret, 8);
    const uint64_t r1 = _mm256_extract_epi64(ret, 0);
    const uint64_t r2 = __builtin_popcountll(r1);
    return r2&1;
}

#elif defined(USE_NEON)
#include <arm_neon.h>
#include "../helper.h"

/// NEON vectorized addition of 128 GF(2) elements
/// \param a[in]: first vector of 128 GF(2) elements
/// \param b[in]: second vector of 128 GF(2) elements
/// \return vector of a+b in each bit position
static uint8x16_t gf2v_add_u128(const uint8x16_t a, const uint8x16_t b) {
    return a ^ b;
}

/// NEON vectorized subtraction of 128 GF(2) elements
/// \param a[in]: vector of 128 GF(2) minuends
/// \param b[in]: vector of 128 GF(2) subtrahends
/// \return vector of a-b in each bit position
static uint8x16_t gf2v_sub_u128(const uint8x16_t a, const uint8x16_t b) {
	return a ^ b;
}

/// NEON vectorized multiplication of 128 GF(2) elements
/// \param a[in]: first vector of 128 GF(2) elements
/// \param b[in]: second vector of 128 GF(2) elements
/// \return vector of a*b in each bit position
static uint8x16_t gf2v_mul_u128(const uint8x16_t a, const uint8x16_t b) {
	return a & b;
}


/// NEON vectorized addition of 256 GF(2) elements (using NEON 128-bit register pairs)
/// \param a[in]: first vector of 256 GF(2) elements
/// \param b[in]: second vector of 256 GF(2) elements
/// \return vector of a+b in each bit position
static uint8x16x2_t gf2v_add_u256(const uint8x16x2_t a, const uint8x16x2_t b) {
    return veorq_u8_x2(a, b);
}

/// NEON vectorized subtraction of 256 GF(2) elements (using NEON 128-bit register pairs)
/// \param a[in]: vector of 256 GF(2) minuends
/// \param b[in]: vector of 256 GF(2) subtrahends
/// \return vector of a-b in each bit position
static uint8x16x2_t gf2v_sub_u256(const uint8x16x2_t a, const uint8x16x2_t b) {
    return veorq_u8_x2(a, b);
}

/// NEON vectorized multiplication of 256 GF(2) elements (using NEON 128-bit register pairs)
/// \param a[in]: first vector of 256 GF(2) elements
/// \param b[in]: second vector of 256 GF(2) elements
/// \return vector of a*b in each bit position
static uint8x16x2_t gf2v_mul_u256(const uint8x16x2_t a, const uint8x16x2_t b) {
    return vandq_u8_x2(a, b);
}
#endif
