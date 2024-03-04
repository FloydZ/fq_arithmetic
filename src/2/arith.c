#include <stdint.h>

/// Representation:
/// the generic base type is `uint64_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.
///
/// 
///
///
///
///

#ifdef USE_AVX2
#elif defined(USE_NEON)
#else

/// add of 64 variables in parallel
uint64_t add(const uint64_t a, const uint64_t b) { return a ^ b; }
uint64_t sub(const uint64_t a, const uint64_t b) { return a ^ b; }
uint64_t mul(const uint64_t a, const uint64_t b) { return a & b; }
#endif


int main() {
	return 1;
}
