#include <stdint.h>

/// Representation:
/// the generic base type is `uint8_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.
typedef uint8_t ff_t;

// single bit operations
uint64_t add(const uint64_t a, const uint64_t b) { return (a&1) ^ (b&1); }
uint64_t sub(const uint64_t a, const uint64_t b) { return (a&1) ^ (b&1); }
uint64_t mul(const uint64_t a, const uint64_t b) { return (a&1) & (b&1); }

/// add of 64 variables in parallel
uint64_t add_T(const uint64_t a, const uint64_t b) { return a ^ b; }
uint64_t sub_T(const uint64_t a, const uint64_t b) { return a ^ b; }
uint64_t mul_T(const uint64_t a, const uint64_t b) { return a & b; }

#ifdef USE_AVX2
#include <immintrin.h>
__m256i add_simd256(const __m256i a, const __m256i b) { 
		return _mm256_xor_si256(a, b);
}
__m256i sub_simd256(const __m256i a, const __m256i b) { 
	return add_simd256(a, b);
}
__m256i mul_simd256(const __m256i a, const __m256i b) { 
	return _mm256_and_si256(a, b);
}

#elif defined(USE_NEON)
#else
	#error "at least avx or neon must be enabled"
#endif

/// C = A*B, where are `A` is in row-major form and `B` in `colum`-major
/// NOTE: trivial bitwise implementation
void matrix_mul(uint8_t *C, uint8_t *A, uint8_t *B,
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

int main() {
	return 1;
}
