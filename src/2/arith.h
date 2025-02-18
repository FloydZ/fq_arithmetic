#pragma once

#include <stdint.h>
#include <stddef.h>

/// Representation:
/// the generic base type is `uint8_t` called a `limb`. Each `limb` can store 
/// up to 8 bits or coordinates mod 2.
typedef uint8_t gf2;

// single bit operations
static gf2 gf2_add(const gf2 a, const gf2 b) { return (a&1u) ^ (b&1u); }
static gf2 gf2_sub(const gf2 a, const gf2 b) { return (a&1u) ^ (b&1u); }
static gf2 gf2_mul(const gf2 a, const gf2 b) { return (a&1u) & (b&1u); }
static gf2 gf2_inv(const gf2 a) { return a; }

/// add of 32 variables in parallel
static uint32_t gf2_add_u32(const uint32_t a, const uint32_t b) { return a ^ b; }
static uint32_t gf2_sub_u32(const uint32_t a, const uint32_t b) { return a ^ b; }
static uint32_t gf2_mul_u32(const uint32_t a, const uint32_t b) { return a & b; }

/// add of 64 variables in parallel
static uint64_t gf2_add_u64(const uint64_t a, const uint64_t b) { return a ^ b; }
static uint64_t gf2_sub_u64(const uint64_t a, const uint64_t b) { return a ^ b; }
static uint64_t gf2_mul_u64(const uint64_t a, const uint64_t b) { return a & b; }

#ifdef USE_AVX2
#include <immintrin.h>

static __m256i gf2_add_u256(const __m256i a, const __m256i b) {
    return _mm256_xor_si256(a, b);
}

static __m256i gf2_sub_u256(const __m256i a, const __m256i b) {
    return _mm256_xor_si256(a, b);
}

static __m256i gf2_mul_u256(const __m256i a, const __m256i b) {
	return _mm256_and_si256(a, b);
}

static __m256i gf2_scalar_u256(const __m256i a, const gf2 b) {
    const __m256i _b = _mm256_set1_epi8(b);
    return gf2_mul_u256(a, _b);
}

/// horizontal xor
/// \param in
/// \return
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
static uint8x16_t add_simd256(const uint8x16_t a, const uint8x16_t b) { 
		return a ^ b;
}
static uint8x16_t sub_simd256(const uint8x16_t a, const uint8x16_t b) { 
	return a ^ b;
}
static uint8x16_t mul_simd256(const uint8x16_t a, const uint8x16_t b) { 
	return a & b;
}
#endif