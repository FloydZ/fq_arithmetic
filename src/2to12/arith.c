#include <stdint.h>
#include <stdio.h>

typedef uint16_t eff_t;

eff_t add(const eff_t a, const eff_t b) {
	return a ^ b;
}

// sage gave me this polynomial
// x^12 + x^7 + x^6 + x^5 + x^3 + x + 1
#define MODULUS 0b1000011101011

eff_t mul(const eff_t a, const eff_t b) {
	eff_t r;
    r = (-(b>>11u     ) & a);
    r = (-(b>>10u & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>9u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>8u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>7u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>6u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>5u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>4u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>3u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>2u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>1u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
 return (-(b      & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
}

#ifdef USE_AVX2
#include <immintrin.h>
#include <wmmintrin.h>

#elif defined(USE_NEON)
#else
#endif


int main() {
	eff_t a=1u << 11u,
          b=1u << 11u;
	eff_t c1 = mul(a, b);
	printf("%u\n", c1);
	return 1;
}
