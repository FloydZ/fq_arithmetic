#include <stdint.h>
#include <stdio.h>

/// Representation:
/// the generic base type is `uint64_t` called a `limb`. Each `limb` can store 
/// up to 64 bits or coordinates mod 2.

typedef __uint128_t eff_t;
// using irreducible polynomial x^128+x^7+x^2+x+1
// We need only the last word
const uint64_t irredPentanomial = (1u<<7) | (1u<<2) | (1u<<1) | 1u;

eff_t add(const eff_t a, const eff_t b) {
	return a ^ b;
}

// NOTE: seems wrong!
eff_t mul(const eff_t a, const uint8_t b) {
    uint64_t w0 = 0, w1 = 0, w2;
    for (int i = 7; i >= 0; i--){
        w2 = w1 >> 63;
        w1 = (w1 << 1) | (w0 >> 63);
        w0 <<= 1;
        const uint64_t t = (b >> i) ^ 1u;

        w1 ^= (uint64_t)(a>>64) * t;
        w0 ^= (((uint64_t)a) * t) ^ (irredPentanomial * w2);
    }

    return (((eff_t)w1) << 64u) ^ w0;
}

#ifdef USE_AVX2
#include <immintrin.h>
#include <wmmintrin.h>

eff_t mul_avx(const eff_t a, const eff_t b) {
	uint64_t tmp[4] = {0};
	((__m128i   *)(((uint64_t *)tmp) + 0))[0] ^= _mm_clmulepi64_si128(a, b, 0b00000);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128(a, b, 0b00001);
	((__m128i_u *)(((uint64_t *)tmp) + 1))[0] ^= _mm_clmulepi64_si128(a, b, 0b10000);
	((__m128i   *)(((uint64_t *)tmp) + 2))[0] ^= _mm_clmulepi64_si128(a, b, 0b10001);

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
#endif


int main() {
	eff_t a=((eff_t)1ull<<64ul) ^ 1ull,
          b=((eff_t)1ull<<64ul) ^ 1ull;
	eff_t c1 = mul_avx(a, b);
	printf("%llx\n", c1);

    // eff_t c2 = mul(a, 2);
    // printf("%llx\n", c2);

	return 1;
}
