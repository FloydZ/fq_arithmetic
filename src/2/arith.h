#include <stdint.h>
#include <stddef.h>

/// Representation:
/// the generic base type is `uint8_t` called a `limb`. Each `limb` can store 
/// up to 8 bits or coordinates mod 2.
typedef uint8_t ff_t;

// single bit operations
static ff_t gf2_add(const ff_t a, const ff_t b) { return (a&1u) ^ (b&1u); }
static ff_t gf2_sub(const ff_t a, const ff_t b) { return (a&1u) ^ (b&1u); }
static ff_t gf2_mul(const ff_t a, const ff_t b) { return (a&1u) & (b&1u); }
static ff_t gf2_inv(const ff_t a) { return a; }

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
static __m256i gf2_scalar_u256(const __m256i a, const ff_t b) {
    const __m256i _b = _mm256_set1_epi8(b);
    return gf2_mul_u256(a, _b);
}
#elif defined(USE_NEON)
static uint8x16_t add_simd256(const uint8x16_t a, const uint8x16_t b) { 
		return a ^ b;
}
static uint8x16_t sub_simd256(const uint8x16_t a, const uint8x16_t b) { 
	return a ^ b
}
static uint8x16_t mul_simd256(const uint8x16_t a, const uint8x16_t b) { 
	return a & b
}
#endif

/// C = A*B, where are `A` is in row-major form and `B` in `colum`-major
/// NOTE: trivial bitwise implementation
static void matrix_mul(uint8_t *C, const uint8_t *A, const uint8_t *B,
		const size_t nrows_A, const size_t ncols_A, const size_t ncols_B) {
	for (uint32_t i = 0; i < nrows_A; i++) {
		for (uint32_t j = 0; j < ncols_B; j++) {
			uint8_t t = 0;
			for (uint32_t k = 0; k < ncols_A; k++) {
				// TODO
			}
		}
	}
}
