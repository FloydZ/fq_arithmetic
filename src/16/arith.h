#pragma once

#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "../helper.h"

#define SWAP(a, b) { a^=b; b^=a; a^=b; }

#define MODULUS 3

/// Representation: two F16 elements are stored in a single `uint8_t`
// 	gf16 := gf2[x]/ (x^4+x+1)
typedef uint8_t ff_t;
typedef uint8_t gf16;

/// table
const uint8_t gf16_mult_table[256] = {
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f, 
    0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e,0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d, 
    0x00,0x03,0x06,0x05,0x0c,0x0f,0x0a,0x09,0x0b,0x08,0x0d,0x0e,0x07,0x04,0x01,0x02, 
    0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f,0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09, 
    0x00,0x05,0x0a,0x0f,0x07,0x02,0x0d,0x08,0x0e,0x0b,0x04,0x01,0x09,0x0c,0x03,0x06, 
    0x00,0x06,0x0c,0x0a,0x0b,0x0d,0x07,0x01,0x05,0x03,0x09,0x0f,0x0e,0x08,0x02,0x04, 
    0x00,0x07,0x0e,0x09,0x0f,0x08,0x01,0x06,0x0d,0x0a,0x03,0x04,0x02,0x05,0x0c,0x0b, 
    0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d,0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01, 
    0x00,0x09,0x01,0x08,0x02,0x0b,0x03,0x0a,0x04,0x0d,0x05,0x0c,0x06,0x0f,0x07,0x0e, 
    0x00,0x0a,0x07,0x0d,0x0e,0x04,0x09,0x03,0x0f,0x05,0x08,0x02,0x01,0x0b,0x06,0x0c, 
    0x00,0x0b,0x05,0x0e,0x0a,0x01,0x0f,0x04,0x07,0x0c,0x02,0x09,0x0d,0x06,0x08,0x03, 
    0x00,0x0c,0x0b,0x07,0x05,0x09,0x0e,0x02,0x0a,0x06,0x01,0x0d,0x0f,0x03,0x04,0x08, 
    0x00,0x0d,0x09,0x04,0x01,0x0c,0x08,0x05,0x02,0x0f,0x0b,0x06,0x03,0x0e,0x0a,0x07, 
    0x00,0x0e,0x0f,0x01,0x0d,0x03,0x02,0x0c,0x09,0x07,0x06,0x08,0x04,0x0a,0x0b,0x05, 
    0x00,0x0f,0x0d,0x02,0x09,0x06,0x04,0x0b,0x01,0x0e,0x0c,0x03,0x08,0x07,0x05,0x0a, 
};

/// NOTE: normally 16 elements would be sufficient, but for easy use in avx
/// it's extended to 32 elements
/// table[i] = i^{-1} mod GF(16)
const uint8_t gf16_inverse_tab[32] __attribute__((aligned(32)))= {
    0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8,
    0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8,
};

static inline ff_t gf16_inv(const ff_t a) {
    return gf16_inverse_tab[a];
}

static inline ff_t gf16_add(const ff_t a,
                            const ff_t b) {
	return a ^ b;
}

/// the same as add: char = 2 = best char
static inline ff_t gf16_sub(const ff_t a,
              const ff_t b) {
	return a ^ b;
}

/// a*a
static inline ff_t gf16_sqr(const ff_t a) {
    ff_t r4 = a&1u;         // constant term
    r4 ^=  (a<<1u)&4u;      // x -> x^2
    r4 ^= ((a>>2u)&1u)*3u;  // x^2 -> x^4 -> x+1
    r4 ^= ((a>>3u)&1u)*12u; // x^3 -> x^6 -> x^3+x^2
    return r4;
}

/// a*b
static inline 
ff_t gf16_mul(const ff_t a,
              const ff_t b) {
    return gf16_mult_table[a * 16 + b];
}

/// NOTE: b must be a single number
/// NOTE: multiplication with a much smaller lookup table
static inline 
ff_t gf16_mul_v2(const ff_t a,
                 const ff_t b) {
    /// row0 : 0*1, ..., 0*4
    /// row1 : 1*1, ..., 1*4
    ///     ...
    /// rowf : f*1, ..., f*4
    const static uint8_t __gf16_mulbase2[64] __attribute__((aligned(64))) = {
            0x00,0x00,0x00,0x00,
            0x01,0x02,0x04,0x08,
            0x02,0x04,0x08,0x03,
            0x03,0x06,0x0c,0x0b,
            0x04,0x08,0x03,0x06,
            0x05,0x0a,0x07,0x0e,
            0x06,0x0c,0x0b,0x05,
            0x07,0x0e,0x0f,0x0d,
            0x08,0x03,0x06,0x0c,
            0x09,0x01,0x02,0x04,
            0x0a,0x07,0x0e,0x0f,
            0x0b,0x05,0x0a,0x07,
            0x0c,0x0b,0x05,0x0a,
            0x0d,0x09,0x01,0x02,
            0x0e,0x0f,0x0d,0x09,
            0x0f,0x0d,0x09,0x01,
    };

    const ff_t *p = __gf16_mulbase2 + b*4;
    const ff_t tmp1 = ((a & 0x11) >> 0) * p[0];
    const ff_t tmp2 = ((a & 0x22) >> 1) * p[1];
    const ff_t tmp3 = ((a & 0x44) >> 2) * p[2];
    const ff_t tmp4 = ((a & 0x88) >> 3) * p[3];
    return tmp1 ^ tmp2 ^ tmp3 ^ tmp4;
}

uint8_t gf16_mul_v3(const uint8_t a,
                    const uint8_t b) {
    uint8_t r;
    r = (-(b>>3    ) & a);
    r = (-(b>>2 & 1) & a) ^ (-(r>>3) & MODULUS) ^ ((r+r) & 0x0F);
    r = (-(b>>1 & 1) & a) ^ (-(r>>3) & MODULUS) ^ ((r+r) & 0x0F);
 return (-(b    & 1) & a) ^ (-(r>>3) & MODULUS) ^ ((r+r) & 0x0F);
}
/// NOTE: two multiplications at once
/// a*b
static inline ff_t gf16v_mul(const ff_t a,
                             const ff_t b) {
    return gf16_mult_table[(a & 0xffff) * 16u + (b & 0xffff)] ^
          (gf16_mult_table[(a >>     4) * 16u + (b >>    4u)] << 4u);
}

/// a + b*c
static inline ff_t gf16_addmul(const ff_t a,
                               const ff_t b,
                               const ff_t c) {
    return gf16_add(a, gf16_mul(b, c));
}

/// vectorized sqr(a) = a**2
static inline uint64_t gf16_sqrv_u64(const uint64_t a) {
    uint64_t a01 = (      a&0x1111111111111111ULL) + ((a<<1)&0x4444444444444444ULL);
    uint64_t a23 = (((a>>2)&0x1111111111111111ULL) + ((a>>1)&0x4444444444444444ULL))*3;
    return a01^a23;
}

#ifdef USE_AVX2

/// M[i, j] = 16**i * j mod 16 for i = row, j = column
const uint8_t __gf16_mulbase[128] __attribute__((aligned(32))) = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f, 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
        0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e, 0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d, 0x00,0x02,0x04,0x06,0x08,0x0a,0x0c,0x0e, 0x03,0x01,0x07,0x05,0x0b,0x09,0x0f,0x0d,
        0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f, 0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09, 0x00,0x04,0x08,0x0c,0x03,0x07,0x0b,0x0f, 0x06,0x02,0x0e,0x0a,0x05,0x01,0x0d,0x09,
        0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d, 0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01, 0x00,0x08,0x03,0x0b,0x06,0x0e,0x05,0x0d, 0x0c,0x04,0x0f,0x07,0x0a,0x02,0x09,0x01,
};


#include <immintrin.h>
typedef struct __xmm_x2 { __m128i v0; __m128i v1; } xmmx2_t;

/// horizontal xor, but not withing a single limb, but over the 8 -32bit limbs
/// \param in
/// \return
static inline __m256i gf16_hadd_avx2_32(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 4));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(ret, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return ret;
}

/// horizontal xor, but over 4-64 bit limbs
static inline uint64_t gf16_hadd_u256_64(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return _mm256_extract_epi64(ret, 0);
}

/// vectorized squaring of 64 values at once.
static inline __m256i gf16_sqrv_u256(const __m256i a) {
    const __m256i mask1 = _mm256_set1_epi8(0x11);
    const __m256i mask4 = _mm256_set1_epi8(0x44);
    const __m256i three = _mm256_set1_epi64x(0x3);

    const __m256i a01 = (a&mask1) + (_mm256_slli_epi64(a, 1) &mask4);
    const __m256i a23 = (_mm256_srli_epi64(a, 2)&mask1) + (_mm256_srli_epi64(a, 1) &mask4);
    const __m256i a02 = _mm256_mullo_epi16(a23, three);
    const __m256i ret = a01 ^ a02;
    return ret;
}

/// special multiplication by `x`
static inline __m256i gf16_mulvx_u256(const __m256i a) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i zero = _mm256_set1_epi8(0x00);
    const __m256i xp1  = _mm256_set1_epi8(0x03);  // x + 1 
	__m256i h1 = _mm256_srli_epi16(a, 4);
	__m256i l1 = a & mask;

	// times x 
	h1 = _mm256_slli_epi16(h1, 1);
	l1 = _mm256_slli_epi16(l1, 1);

	const __m256i selector_h = _mm256_slli_epi16(h1, 2);
	const __m256i selector_l = _mm256_slli_epi16(l1, 2);

	const __m256i add_h = _mm256_blendv_epi8(zero, xp1, selector_h);
	const __m256i add_l = _mm256_blendv_epi8(zero, xp1, selector_l);

	h1 = h1^add_h;
	l1 = l1^add_l;

	return l1 ^ (_mm256_slli_epi16(h1, 4));
}

/// computes a^31, a^30, ..., a^1
static inline __m256i gf16_powers(const ff_t a) {
	ff_t tmp[32];
	tmp[0] = a;
	for (uint32_t i = 1; i < 32; i++) {
		tmp[i] = gf16_mul(a, tmp[i - 1]);
	}

	const __m256i ret = _mm256_load_si256((__m256i *)tmp);
	return ret;
}

/// Full multiplication
/// NOTE: assumes that in every byte the two nibbles are the same in b
/// \return a*b \in \F_16 for all 64 nibbles in the
static inline __m128i gf16v_mul_u128(const __m128i a,
                                     const __m128i b) {
    const __m128i mask_lvl2 = _mm_load_si128((__m128i const *) (__gf16_mulbase +   32));
    const __m128i mask_lvl3 = _mm_load_si128((__m128i const *) (__gf16_mulbase + 32*2));
    const __m128i mask_lvl4 = _mm_load_si128((__m128i const *) (__gf16_mulbase + 32*3));
    const __m128i zero = _mm_setzero_si128();

    __m128i low_lookup = b;
    __m128i high_lookup = _mm_slli_epi16(low_lookup, 4);
    __m128i tmp1l = _mm_slli_epi16(a, 7);
    __m128i tmp2h = _mm_slli_epi16(a, 3);
    __m128i tmp_mul_0_1 = _mm_blendv_epi8(zero, low_lookup , tmp1l);
    __m128i tmp_mul_0_2 = _mm_blendv_epi8(zero, high_lookup, tmp2h);
    __m128i tmp = _mm_xor_si128(tmp_mul_0_1, tmp_mul_0_2);
    __m128i tmp1;

    /// 1
    low_lookup = _mm_shuffle_epi8(mask_lvl2, b);
    high_lookup = _mm_slli_epi16(low_lookup, 4);
    tmp1l = _mm_slli_epi16(a, 6);
    tmp2h = _mm_slli_epi16(a, 2);
    tmp_mul_0_1 = _mm_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm_xor_si128(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm_xor_si128(tmp, tmp1);

    /// 2
    low_lookup = _mm_shuffle_epi8(mask_lvl3, b);
    high_lookup = _mm_slli_epi16(low_lookup, 4);
    tmp1l = _mm_slli_epi16(a, 5);
    tmp2h = _mm_slli_epi16(a, 1);
    tmp_mul_0_1 = _mm_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm_xor_si128(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm_xor_si128(tmp, tmp1);

    /// 3
    low_lookup = _mm_shuffle_epi8(mask_lvl4, b);
    high_lookup = _mm_slli_epi16(low_lookup, 4);
    tmp1l = _mm_slli_epi16(a, 4);
    tmp_mul_0_1 = _mm_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm_blendv_epi8(zero, high_lookup, a );
    tmp1 = _mm_xor_si128(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm_xor_si128(tmp, tmp1);
    return tmp;
}

/// NOTE: assumes that in every byte the two nibbles are the same in b
/// \return a*b \in \F_16 for all 64 nibbles in the
static inline __m256i gf16v_mul_u256(const __m256i a,
                                     const __m256i b) {
    const __m256i mask_lvl2 = _mm256_load_si256((__m256i const *) (__gf16_mulbase +   32));
    const __m256i mask_lvl3 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2));
    const __m256i mask_lvl4 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3));
    const __m256i zero = _mm256_setzero_si256();

    __m256i low_lookup = b;
    __m256i high_lookup = _mm256_slli_epi16(low_lookup, 4);
    __m256i tmp1l = _mm256_slli_epi16(a, 7);
    __m256i tmp2h = _mm256_slli_epi16(a, 3);
    __m256i tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    __m256i tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    __m256i tmp = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    __m256i tmp1;

    /// 1
    low_lookup = _mm256_shuffle_epi8(mask_lvl2, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 6);
    tmp2h = _mm256_slli_epi16(a, 2);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 2
    low_lookup = _mm256_shuffle_epi8(mask_lvl3, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 5);
    tmp2h = _mm256_slli_epi16(a, 1);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 3
    low_lookup = _mm256_shuffle_epi8(mask_lvl4, b);
    high_lookup = _mm256_slli_epi16(low_lookup, 4);
    tmp1l = _mm256_slli_epi16(a, 4);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, a );
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);
    return tmp;
}

/// Full multiplication
/// \return a*b \in \F_16 for all 64 nibbles in the
static inline __m256i gf16v_mul_full_u256(const __m256i a,
                                          const __m256i _b) {
    const __m256i mask_lvl2 = _mm256_load_si256((__m256i const *) (__gf16_mulbase +   32));
    const __m256i mask_lvl3 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2));
    const __m256i mask_lvl4 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3));
    const __m256i zero = _mm256_setzero_si256();
    const __m256i mask1 = _mm256_set1_epi8(0x0F);

    const __m256i b = _b & mask1;
    const __m256i b2 = _mm256_srli_epi16(_b, 4) & mask1;
    __m256i low_lookup  = b;
    __m256i high_lookup = _mm256_slli_epi16(b2, 4);
    __m256i tmp1l = _mm256_slli_epi16(a, 7);
    __m256i tmp2h = _mm256_slli_epi16(a, 3);
    __m256i tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    __m256i tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    __m256i tmp = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    __m256i tmp1;

    /// 1
    low_lookup = _mm256_shuffle_epi8(mask_lvl2, b);
    high_lookup = _mm256_slli_epi16(_mm256_shuffle_epi8(mask_lvl2, b2), 4);
    tmp1l = _mm256_slli_epi16(a, 6);
    tmp2h = _mm256_slli_epi16(a, 2);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 2
    low_lookup = _mm256_shuffle_epi8(mask_lvl3, b);
    high_lookup = _mm256_slli_epi16(_mm256_shuffle_epi8(mask_lvl3, b2), 4);
    tmp1l = _mm256_slli_epi16(a, 5);
    tmp2h = _mm256_slli_epi16(a, 1);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, tmp2h);
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);

    /// 3
    low_lookup = _mm256_shuffle_epi8(mask_lvl4, b);
    high_lookup = _mm256_slli_epi16(_mm256_shuffle_epi8(mask_lvl4, b2), 4);
    tmp1l = _mm256_slli_epi16(a, 4);
    tmp_mul_0_1 = _mm256_blendv_epi8(zero, low_lookup , tmp1l);
    tmp_mul_0_2 = _mm256_blendv_epi8(zero, high_lookup, a );
    tmp1 = _mm256_xor_si256(tmp_mul_0_1, tmp_mul_0_2);
    tmp  = _mm256_xor_si256(tmp, tmp1);
    return tmp;
}

/// SRC: https://github.com/pqov/pqov-paper/blob/main/src/avx2/gf16_avx2.h
/// Caution: multabs are different from ssse3 version
/// ssse3:  [multab_low] [ multab_high ]
///         <-   16  ->  <-    16     ->
/// avx2:   [         multab_low       ]
///         <---        32          --->
/// NOTE: if you have the choice use `mult_simd` its faster.
void gf16v_generate_multab_16_avx2(__m256i *multabs,
									__m128i a,
									unsigned len ) {
    __m256i broadcast_x1 = _mm256_set_epi8( 0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16,  0,-16,0,-16, 0,-16,0,-16 );
    __m256i broadcast_x2 = _mm256_set_epi8( 0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16,  0,0,-16,-16, 0,0,-16,-16 );
    __m256i broadcast_x4 = _mm256_set_epi8( 0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16,  0,0,0,0, -16,-16,-16,-16 );
    __m256i broadcast_x8 = _mm256_set_epi8( 0,0,0,0, 0,0,0,0,  -16,-16,-16,-16, -16,-16,-16,-16,  0,0,0,0, 0,0,0,0,  -16,-16,-16,-16, -16,-16,-16,-16 );

    //__m256i bx1 = _mm256_inserti128_si256( _mm256_castsi128_si256(a), a , 1 );
    __m256i bx1 = _mm256_setr_m128i( a , a );
    __m256i bx2 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32 ))   , bx1 );
    __m256i bx4 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2 )) , bx1 );
    __m256i bx8 = _mm256_shuffle_epi8( _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3 )) , bx1 );


    multabs[0] =  _mm256_shuffle_epi8(bx1,broadcast_x1) ^ _mm256_shuffle_epi8(bx2,broadcast_x2)
                  ^ _mm256_shuffle_epi8(bx4,broadcast_x4) ^ _mm256_shuffle_epi8(bx8,broadcast_x8);
    for(uint32_t i=1;i<len;i++) {
        bx1 = _mm256_srli_si256(bx1, 1);
        bx2 = _mm256_srli_si256(bx2, 1);
        bx4 = _mm256_srli_si256(bx4, 1);
        bx8 = _mm256_srli_si256(bx8, 1);

        multabs[i] = _mm256_shuffle_epi8(bx1,broadcast_x1) ^
					 _mm256_shuffle_epi8(bx2,broadcast_x2) ^
					 _mm256_shuffle_epi8(bx4,broadcast_x4) ^
					 _mm256_shuffle_epi8(bx8,broadcast_x8);
    }
}

///  TODO use `mult_simd`
void mat_simd_16x16(uint8_t *c, 
					const uint8_t *matA,
					const __m256i * multab_b){
    __m256i mask_f = _mm256_set1_epi8(0xf);
    __m256i ma0123 = _mm256_loadu_si256((const __m256i*) (matA) );    // 4 rows
    __m256i ma4567 = _mm256_loadu_si256((const __m256i*) (matA+32) ); // 4 rows
    __m256i mb02 = _mm256_permute2x128_si256(multab_b[0], multab_b[2] , 0x20 );
    __m256i mb13 = _mm256_permute2x128_si256(multab_b[1], multab_b[3] , 0x20 );
    __m256i mb46 = _mm256_permute2x128_si256(multab_b[4], multab_b[6] , 0x20 );
    __m256i mb57 = _mm256_permute2x128_si256(multab_b[5], multab_b[7] , 0x20 );
    __m256i ma0123l = ma0123&mask_f;
    __m256i ma0123h = _mm256_srli_epi16(ma0123,4)&mask_f;
    __m256i ma4567l = ma4567&mask_f;
    __m256i ma4567h = _mm256_srli_epi16(ma4567,4)&mask_f;
    __m256i ma02 = _mm256_unpacklo_epi64(ma0123l, ma0123h);
    __m256i ma13 = _mm256_unpackhi_epi64(ma0123l, ma0123h);
    __m256i ma46 = _mm256_unpacklo_epi64(ma4567l, ma4567h);
    __m256i ma57 = _mm256_unpackhi_epi64(ma4567l, ma4567h);
    __m256i r = _mm256_shuffle_epi8(mb02,ma02) ^
				_mm256_shuffle_epi8(mb13,ma13) ^
				_mm256_shuffle_epi8(mb46,ma46) ^
				_mm256_shuffle_epi8(mb57,ma57);

    __m256i ma89ab = _mm256_loadu_si256((const __m256i*) (matA+64) ); // 4 rows
    __m256i macdef = _mm256_loadu_si256((const __m256i*) (matA+96) ); // 4 rows
    __m256i mb8a = _mm256_permute2x128_si256(multab_b[8], multab_b[10], 0x20);
    __m256i mb9b = _mm256_permute2x128_si256(multab_b[9], multab_b[11], 0x20);
    __m256i mbce = _mm256_permute2x128_si256(multab_b[12], multab_b[14], 0x20);
    __m256i mbdf = _mm256_permute2x128_si256(multab_b[13], multab_b[15], 0x20);
    __m256i ma89abl = ma89ab&mask_f;
    __m256i ma89abh = _mm256_srli_epi16(ma89ab,4)&mask_f;
    __m256i macdefl = macdef&mask_f;
    __m256i macdefh = _mm256_srli_epi16(macdef,4)&mask_f;
    __m256i ma8a = _mm256_unpacklo_epi64(ma89abl,ma89abh);
    __m256i ma9b = _mm256_unpackhi_epi64(ma89abl,ma89abh);
    __m256i mace = _mm256_unpacklo_epi64(macdefl,macdefh);
    __m256i madf = _mm256_unpackhi_epi64(macdefl,macdefh);
    r ^= _mm256_shuffle_epi8(mb8a,ma8a) ^
		 _mm256_shuffle_epi8(mb9b,ma9b) ^
		 _mm256_shuffle_epi8(mbce,mace) ^
		 _mm256_shuffle_epi8(mbdf,madf);

    __m128i rr = _mm256_castsi256_si128(r) ^ _mm256_extractf128_si256(r, 1);
    __m128i rr2 = rr ^ _mm_srli_si128(_mm_slli_epi16(rr, 4), 8);

    uint8_t temp[16] __attribute__((aligned(16)));
    _mm_store_si128( (__m128i*)temp , rr2 );
    for(uint32_t i = 0; i < 8; i++) {
        c[i] = temp[i];
    }
}

// input a:          0x12 0x34 0x56 0x78 ......
// output x_align:   0x02 0x01 0x04 0x03 0x06 0x05 .........
static inline
void gf16v_split_16to32_sse(__m128i *x_align, __m128i a) {
    __m128i mask_f = _mm_set1_epi8(0xf);
    __m128i al = a&mask_f;
    __m128i ah = _mm_srli_epi16( a,4 )&mask_f;

    __m128i a0 = _mm_unpacklo_epi8( al , ah );
    __m128i a1 = _mm_unpackhi_epi8( al , ah );

    _mm_store_si128( x_align , a0 );
    _mm_store_si128( x_align + 1 , a1 );
}

static inline
xmmx2_t gf16v_split_16to32_sse2( __m128i a) {
    __m128i mask_f = _mm_set1_epi8(0xf);
    __m128i al = a & mask_f;
    __m128i ah = _mm_srli_epi16(a, 4) & mask_f;
    xmmx2_t r;
    r.v0 = _mm_unpacklo_epi8(al, ah);
    r.v1 = _mm_unpackhi_epi8(al, ah);
    return r;
}


// SOURCE: https://github.com/pqov/pqov-paper/blob/main/src/avx2/gf16_avx2.h
// generate multiplication table for '4-bit' variable 'b'
static inline __m256i gf16v_tbl32_multab(const ff_t b) {
    __m256i bx = _mm256_set1_epi16( b&0xf );
    __m256i b1 = _mm256_srli_epi16( bx , 1 );

    __m256i tab0 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*0));
    __m256i tab1 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*1));
    __m256i tab2 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*2));
    __m256i tab3 = _mm256_load_si256((__m256i const *) (__gf16_mulbase + 32*3));

    __m256i mask_1  = _mm256_set1_epi16(1);
    __m256i mask_4  = _mm256_set1_epi16(4);
    __m256i mask_0  = _mm256_setzero_si256();

    return (tab0 & _mm256_cmpgt_epi16(bx&mask_1  , mask_0))
         ^ (tab1 & _mm256_cmpgt_epi16(b1&mask_1  , mask_0))
         ^ (tab2 & _mm256_cmpgt_epi16(bx&mask_4  , mask_0))
         ^ (tab3 & _mm256_cmpgt_epi16(b1&mask_4  , mask_0));
}

/// \param a[in]:
/// \param _num_byte[in]:
/// \return 
static inline
__m128i _load_xmm(const uint8_t *a,
                  const unsigned _num_byte ) {
    uint8_t temp[32] __attribute__((aligned(32)));
    for(unsigned i=0;i<_num_byte;i++) { 
        temp[i] = a[i]; 
    }
    return _mm_load_si128((__m128i*)temp);
}

/// \param a[in]:
/// \param _num_byte[in]:
/// \param [in]:
/// \return 
static inline
void _store_xmm(uint8_t *a ,
                const unsigned _num_byte , 
                const __m128i data ) {
    uint8_t temp[32] __attribute__((aligned(32)));
    _mm_store_si128((__m128i*)temp,data);
    for(unsigned i=0;i<_num_byte;i++) a[i] = temp[i];
}
static inline __m128i linear_transform_8x8_128b( __m128i tab_l , __m128i tab_h , __m128i v , __m128i mask_f ){
    return _mm_shuffle_epi8(tab_l,v&mask_f)^_mm_shuffle_epi8(tab_h,_mm_srli_epi16(v,4)&mask_f);
}

static inline
void linearmap_8x8_accu_sse(uint8_t *accu_c, 
		const uint8_t *a,
		const __m128i ml,
		const __m128i mh,
		const __m128i mask,
		const unsigned _num_byte ) {

    unsigned n_16 = _num_byte>>4;
    unsigned rem = _num_byte&15;
    while(n_16--) {
        __m128i inp = _mm_loadu_si128( (__m128i*)(a) );
        __m128i out = _mm_loadu_si128( (__m128i*)(accu_c) );
        __m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
        r0 ^= out;
        _mm_storeu_si128( (__m128i*)(accu_c) , r0 );
        a += 16;
        accu_c += 16;
    }
    if( rem ) {
        __m128i inp = _load_xmm( a , rem );
        __m128i out = _load_xmm( accu_c , rem );
        __m128i r0 = linear_transform_8x8_128b( ml , mh , inp , mask );
        r0 ^= out;
        _store_xmm( accu_c , rem , r0 );
    }
}

/// SOURCE: https://github.com/pqov/pqov-paper/blob/main/src/avx2/gf16_avx2.h
static inline __m256i linear_transform_8x8_256b(__m256i tab_l,
                                                __m256i tab_h,
                                                __m256i v,
                                                __m256i mask_f){
    return _mm256_shuffle_epi8(tab_l,v&mask_f) ^ _mm256_shuffle_epi8(tab_h,_mm256_srli_epi16(v,4)&mask_f);
}


// SOURCE: https://github.com/pqov/pqov-paper/blob/main/src/avx2/gf16_avx2.h
// multiplication by generating multiplication tables
static inline __m256i gf16v_mul_avx2( __m256i a , uint8_t b ) {
    __m256i multab_l = gf16v_tbl32_multab( b );
    __m256i multab_h = _mm256_slli_epi16( multab_l , 4 );

    return linear_transform_8x8_256b( multab_l , multab_h , a , _mm256_set1_epi8(0xf) );
}



static inline
void linearmap_8x8_accu_ymm( uint8_t * accu_c , const uint8_t * a ,  __m256i ml , __m256i mh , __m256i mask , unsigned _num_byte ) {
    unsigned n_32 = _num_byte>>5;
    unsigned rem = _num_byte&31;
    while(n_32--){
        __m256i inp = _mm256_loadu_si256( (__m256i*)a );
        __m256i out = _mm256_loadu_si256( (__m256i*)accu_c );
        __m256i r0 = out ^ linear_transform_8x8_256b( ml , mh , inp , mask );
        _mm256_storeu_si256( (__m256i*)accu_c , r0 );
        a += 32;
        accu_c += 32;
    }
    if( rem ) {
        linearmap_8x8_accu_sse( accu_c , a , _mm256_castsi256_si128(ml) , _mm256_castsi256_si128(mh) , _mm256_castsi256_si128(mask) , rem );
    }
}

/// internal function
/// \param accu_c
/// \param a
/// \param gf16_b
/// \param _num_byte
static inline
void gf16v_madd_avx2( uint8_t * accu_c, const uint8_t * a , uint8_t gf16_b, unsigned _num_byte ) {
    __m256i ml = gf16v_tbl32_multab( gf16_b );
    __m256i mh = _mm256_slli_epi16( ml , 4 );
    __m256i mask = _mm256_set1_epi8(0xf);
    linearmap_8x8_accu_ymm( accu_c , a , ml , mh , mask , _num_byte );
}

#elif defined(USE_NEON)
#include <arm_neon.h>
const unsigned char __gf16_reduce[16] __attribute__((aligned(16))) = {
        0x00,0x13,0x26,0x35,0x4c,0x5f,0x6a,0x79, 0x8b,0x98,0xad,0xbe,0xc7,0xd4,0xe1,0xf2
};

// NOTE: not a full multiplication (only the lower 4 bits in each limb are used in b)
uint8x16_t gf16v_mul_u128(const uint8x16_t a,
                          const uint8x16_t b) {
    uint8x16_t mask_f = vdupq_n_u8( 0xf );
    uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);

    uint8x16_t al0 = a&mask_f;
    uint8x16_t ah0 = vshrq_n_u8( a , 4 );

	// mul
    poly8x16_t abl = vmulq_p8(al0, b);
    poly8x16_t abh = vmulq_p8(ah0, b);

    poly8x16_t rl = abl ^ vqtbl1q_u8(tab_reduce, vshrq_n_u8(abl, 4));
    poly8x16_t rh = abh ^ vqtbl1q_u8(tab_reduce, vshrq_n_u8(abh, 4));
    return vsliq_n_u8( rl , rh , 4 );
}

uint8x16_t gf16v_mul_full_u128(const uint8x16_t a,
                               const uint8x16_t b) {
    const uint8x16_t mask_f = vdupq_n_u8( 0xf );
    const uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);

    const uint8x16_t al0 = a&mask_f;
    const uint8x16_t bl0 = b&mask_f;
    const uint8x16_t ah0 = vshrq_n_u8(a, 4);
    const uint8x16_t bh0 = vshrq_n_u8(b, 4);

	// mul
    poly8x16_t abl = vmulq_p8(al0, bl0);
    poly8x16_t abh = vmulq_p8(ah0, bh0);

    poly8x16_t rl = abl ^ vqtbl1q_u8(tab_reduce, vshrq_n_u8(abl, 4));
    poly8x16_t rh = abh ^ vqtbl1q_u8(tab_reduce, vshrq_n_u8(abh, 4));

    return vsliq_n_u8(rl, rh, 4);
}

// NOTE: not a full multiplication
// only computes the multiplication on the lower 4 bits within
// each 8 bit limb
uint8x16_t gf16v_mul_u128_lower(uint8x16_t a,
                                uint8x16_t b) {
    const uint8x16_t m   = vdupq_n_u8( 0xf );
    const uint8x16_t tr  = vld1q_u8(__gf16_reduce);
    const uint8x16_t al0 = a&m;

	// mul
    const poly8x16_t abl = vmulq_p8(al0 , b);
    const poly8x16_t rl = abl ^ vqtbl1q_u8(tr, vshrq_n_u8(abl, 4));
    return rl;
}

// only computes the multiplication on the upper 4 bits within
// each 8 bit limb
uint8x16_t gf16v_mul_u128_upper(uint8x16_t a,
                                uint8x16_t b) {
    const uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);
    const uint8x16_t ah0 = vshrq_n_u8(a, 4);

    // mul
    const poly8x16_t abh = vmulq_p8(ah0, b);
    const poly8x16_t rh = abh ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abh,4) );
    return rh;
}
#else
// any other architecture
#endif

#undef MODULUS
