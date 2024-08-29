#include <stdint.h>
#include <stddef.h>

/// Representation:
/// 0= X^2 + X + 1
typedef uint8_t ff_t;

// = { b*X^0, b*X^1 }
const uint8_t gf4_mult_table[] = {0x00,0x00,0x01,0x02,0x02,0x03,0x03,0x01};

static ff_t gf4_add(const ff_t a, const ff_t b) { return a ^ b; }
static ff_t gf4_sub(const ff_t a, const ff_t b) { return a ^ b; }
static ff_t gf4_mul(const ff_t a, const ff_t b) {
    return ((a&1u )*gf4_mult_table[b*2 + 0]) ^
           ((a>>1u)*gf4_mult_table[b*2 + 1]);
}

/// same as `gf4_mul` but without a multiplication
static ff_t gf4_mul_v2(const ff_t a, const ff_t b) {
    const ff_t tmpl1 = a & b;
    const ff_t tmpl2 = ((tmpl1 ^ (tmpl1 >> 1)) & 1u);

    ff_t tmph1 = (((a<<1) & b) ^ ((b<<1) & a));
    ff_t tmph3 = (tmpl1 ^ tmph1) & 2u;
    return  tmpl2 ^ tmph3;
}

static uint64_t gf4v_add_u64(const uint64_t a, const uint64_t b) { return a ^ b; }
static uint64_t gf4v_sub_u64(const uint64_t a, const uint64_t b) { return a ^ b; }
static uint64_t gf4v_scalar_u64(const uint64_t a, const ff_t b) {
	return (( a&0x5555555555555555    )*gf4_mult_table[b*2 + 0]) ^ 
		   (((a&0xaaaaaaaaaaaaaaaa)>>1)*gf4_mult_table[b*2 + 1]);
}
static uint64_t gf4v_mul_u64(const uint64_t a, const uint64_t b) {
    const uint64_t mask = 0b11;
    uint64_t ret = 0;
    for (uint32_t i = 0; i < 32u; ++i) {
        const ff_t a1 = (a >> (i*2u))&mask;
        const ff_t b1 = (b >> (i*2u))&mask;
        const ff_t t1 = gf4_mul(a1, b1);
        const uint64_t t2 = ((uint64_t)t1) << (i*2u);
        ret ^= t2;
    }
    return ret;
}

#ifdef USE_AVX2
#include <immintrin.h>

static __m256i gf4v_mul_u256(const __m256i a,
                             const __m256i b){
    const __m256i mask1 = _mm256_set1_epi8(0b01010101);
    const __m256i mask2 = _mm256_set1_epi8(0b10101010);

    const __m256i and_ = a & b;
    const __m256i and2 = (a<<1) & b;
    const __m256i and1 = a & (b<<1);
    const __m256i l = and1 ^ and2;

    __m256i ret = (and_  ^ (and_ >> 1u)) & mask1;
    ret ^= (l ^ and_) & mask2;
    return ret;
}

static __m256i gf4v_scalar_u256(const __m256i a,
		                        const uint8_t b){
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

static uint8x8_t scalar_u256(const uint8x8_t a,
		const uint8_t b) {
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
#endif
