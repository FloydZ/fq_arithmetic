#include <stdint.h>
#include <stdlib.h>
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


#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x800 ? '1' : '0'), \
  ((byte) & 0x400 ? '1' : '0'), \
  ((byte) & 0x200 ? '1' : '0'), \
  ((byte) & 0x100 ? '1' : '0'), \
  ((byte) & 0x080 ? '1' : '0'), \
  ((byte) & 0x040 ? '1' : '0'), \
  ((byte) & 0x020 ? '1' : '0'), \
  ((byte) & 0x010 ? '1' : '0'), \
  ((byte) & 0x008 ? '1' : '0'), \
  ((byte) & 0x004 ? '1' : '0'), \
  ((byte) & 0x002 ? '1' : '0'), \
  ((byte) & 0x001 ? '1' : '0') 


int main() {
	//printf("[");
	//for (eff_t a = 0; a < (1u<<12u); a++) {
	//	for (eff_t b = 0; b < (1u<<12u); b++) {
	//		printf("%u,", mul(a, b)&1);
	//	}
	//}
	//printf("]");

	// print dnf
	for (eff_t a = 0; a < (1u<<12u); a++) {
		for (eff_t b = 0; b < (1u<<12u); b++) {
			if (mul(a, b)&1u){ 
				printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(a));
				printf(BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(b));
				printf("\n");
			}
		}
	}
}
