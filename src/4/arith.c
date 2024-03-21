#include <stdint.h>

/// Representation:
typedef uint8_t ff_t;

const uint8_t gf4_mult_table[] = {0x00,0x00,0x01,0x02,0x02,0x03,0x03,0x01};

ff_t add(const ff_t a, const ff_t b) { return a ^ b; }
ff_t sub(const ff_t a, const ff_t b) { return a ^ b; }
ff_t mul(const ff_t a, const ff_t b) {
	return ((a&1 )*gf4_mult_table[b*2 + 0]) ^ 
		   ((a>>1)*gf4_mult_table[b*2 + 1]);
}

ff_t add_T(const ff_t a, const ff_t b) { return a ^ b; }
ff_t sub_T(const ff_t a, const ff_t b) { return a ^ b; }
ff_t mul_T(const ff_t a, const ff_t b) {
	return (( a&0x5555555555555555    )*gf4_mult_table[b*2 + 0]) ^ 
		   (((a&0xaaaaaaaaaaaaaaaa)>>1)*gf4_mult_table[b*2 + 1]);
}
#ifdef USE_AVX2

#include <immintrin.h>
__m256i scalar_simd(const __m256i a, const uint8_t b){
	const __m256i m0 = _mm256_set1_epi8(0x55);
	const __m256i m1 = _mm256_set1_epi8(0xaa);
	const __m256i s0 = _mm256_set1_epi16(gf4_mult_table[b*2 + 0]);
	const __m256i s1 = _mm256_set1_epi16(gf4_mult_table[b*2 + 1]);
	__m256i r0, r1;

	r0 = _mm256_and_si256(a, m1);
	r1 = _mm256_and_si256(a, m1);
	r1 = _mm256_srli_epi16(r1, 1);
	r0 = _mm256_mullo_epi16(r0, s0);
	r1 = _mm256_mullo_epi16(r1, s1);
	r0 = _mm256_xor_si256(r0, r1);
	return r0;
}

#elif defined(USE_NEON)

uint8x8_t scalar_simd(const uint8x8_t a, const uint8_t b) {
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
#else
# error "at least avx2 or neon"
#endif


int main() {
	return 1;
}
