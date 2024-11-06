#include <stdint.h>

typedef uint16_t eff_t;


#define MODULUS 0x805

eff_t add(const eff_t a, const eff_t b) {
	return a ^ b;
}

eff_t mul(const eff_t a, const eff_t b) {
    eff_t result = -(a & 1) & b;
    eff_t tmp = b;
    for(int i=1 ; i<11 ; i++) {
        tmp = ((tmp << 1) ^ (-(tmp >> 10) & MODULUS));
        result = result ^ (-(a >> i & 1) & tmp);
    }
    return result;
}

/// \return a^-1
static inline eff_t mirath_ff_mu_inv(const eff_t a) {
    eff_t result = a;
    for(int i=0 ; i<9 ; i++) {
        result = mul(result, result);
        result = mul(result, a);
    }
    result = mul(result, result);
    return result;
}

#ifdef USE_AVX2
#include <immintrin.h>
#include <wmmintrin.h>

#elif defined(USE_NEON)
#else
#endif


int main() {
	return 1;
}
