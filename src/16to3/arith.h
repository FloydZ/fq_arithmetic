#pragma once

#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../helper.h"
#include "../16/arith.h"
#include "../16/matrix.h"


// lookup table for switching between gf16 and gf16to3
static const uint8_t gf16_to_gf16to3[16] __attribute__((aligned(32))) = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
};

/// Representation: a single F16to3 elements is stored in uin16_t
///  GF16to3:uint16_t: X^3 + x + 1;
typedef uint16_t gf16to3;

/// \return a+b
static inline gf16to3 gf16to3_add(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a+b for 5 elements in parallel
static inline gf16to3 gf16to3v_add(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// the same as add: char = 2 = best char
/// \return a-b = a+b
static inline gf16to3 gf16to3_sub(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a-b for 5 elements in parallel
static inline gf16to3 gf16to3v_sub(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// vector multiplication
/// \return a*b
static inline gf16to3 gf16to3_mul_gf16(const gf16to3 a,
                                       const gf16 b) {
    return gf16_mul(a&0xF, b) ^
          (gf16_mul((a>>4)&0xF, b) << 4u) ^
          (gf16_mul((a>>8)&0xF, b) << 8u);
}

/// implemented using karatsuba
/// \return a*b
static inline gf16to3 gf16to3_mul(const gf16to3 a,
                                  const gf16to3 b) {
    gf16to3 r;

    const gf16 a0 = a&0xF, a1 = (a>>4)&0XF, a2 = (a>>8)&0xF;
    const gf16 b0 = b&0xF, b1 = (b>>4)&0XF, b2 = (b>>8)&0xF;

    const gf16 p0 = gf16_mul(a0, b0);
    const gf16 p1 = gf16_mul(a1, b1);
    const gf16 p2 = gf16_mul(a2, b2);

    const gf16 a01 = gf16_add(a0, a1);
    const gf16 a12 = gf16_add(a1, a2);
    const gf16 b01 = gf16_add(b0, b1);
    const gf16 b12 = gf16_add(b1, b2);
    const gf16 p01 = gf16_mul(a01, b01);
    const gf16 p12 = gf16_mul(a12, b12);

    const gf16 a012 = gf16_add(a01, a2);
    const gf16 b012 = gf16_add(b01, b2);
    const gf16 p012 = gf16_mul(a012, b012);

    // compute lowest limb
    r = gf16_add(p1, p2);
    r = gf16_add(r, p12);
    r = gf16_add(r, p0);

    r^= p2 << 4;
    r^= p01 << 4;
    r^= p0 << 4;
    r^= p1 << 4;

    r^= p012 << 8;
    r^= p01 << 8;
    r^= p12 << 8;
    return r;
}

static inline gf16to3 gf16to3v_mul(const uint64_t a,
                                   const uint64_t b) {
    const uint64_t m0 = 0x00F00F00F00F, m1 = 0x0F00F00F00F0,
                   m2 = 0xF00F00F00F00;
    uint64_t r;

    // TODO

    return r;
}


#ifdef USE_AVX2

/// vector multiplication
/// \param a
/// \param b
/// \return a*b;
static inline __m256i gf16to3v_mul_gf16_u256(const __m256i a,
                                             const uint8_t b) {
    const __m256i bb = _mm256_set1_epi16(b);
    return gf16v_mul_u256(a, bb);
}


static inline __m128i gf16to3v_mul_gf16_u128(const __m128i a,
                                             const uint8_t b) {
    const __m128i bb = _mm_set1_epi16(b);
    return gf16v_mul_u128(a, bb);
}

/// NOTE there are multiple ways to implement this:
///     - assumes that each 1.5byte element is stored in 2 bytes,
///         thus there are 16 elements in a register
///     - or that that the elements are compressed together, meaning
///         we have 21 elements in a register
/// NOTE: this implementation follows the first approach
static inline __m256i gf16to3v_mul_u256(const __m256i a,
                                        const __m256i b) {
    __m256i r;
    const __m256i m0     = _mm256_set1_epi16(0x00F);
    const __m256i m1     = _mm256_set1_epi16(0x0F0);
    const __m256i m2     = _mm256_set1_epi16(0xF00);
    const __m256i m      = _mm256_set1_epi16(0xFFF);
    const __m256i pi     = gf16v_mul_full_u256(a, b);

    // bit    0     4      8     12     16
    // a01_12=[a0   , a0^a1, a1^a2,  a2]
    // b01_12=[b0   , b0^b1, b1^b2,  b2]
    const __m256i a01_12 = a ^ _mm256_slli_epi16(a, 4);
    const __m256i b01_12 = b ^ _mm256_slli_epi16(b, 4);

    // bit    0     4      8     12        16
    // a012 = [a0   , a0^a1, a1^a2, a2^a1^a0]
    // b012 = [b0   , b0^b1, b1^b2, b2^b1^b0]
    const __m256i a012 = a01_12 ^ _mm256_slli_epi16(a01_12&m1 , 8);
    const __m256i b012 = b01_12 ^ _mm256_slli_epi16(b01_12&m1 , 8);

    // bit    0    4    8    12    16
    // p012 = [p0 , p01, p12 , p012]
    // p012 = [p01, p12, p012, 0000]
    __m256i p012 = gf16v_mul_full_u256(a012, b012);
    p012 = _mm256_srli_epi16(p012, 4);

    // bit 0         4         8    16
    // r = [p0^p1^p2, p0^p1^p2, 0, 0]
    r = pi ^ (_mm256_srli_epi16(pi, 4));
    r ^= _mm256_srli_epi16(pi, 8);
    r ^= _mm256_slli_epi16(pi&m0, 4);     // ^= p0
    r &= (m0 ^ m1);

    r ^= _mm256_srli_epi16(p012, 4) & m0;   //^= p12
    r ^= _mm256_slli_epi16(p012, 4);        // ^= p01, ^= p12
    r ^= _mm256_slli_epi16(p012, 8);
    r ^= p012&m2; // ^= p012

    return r&m;
}

#endif

#ifdef USE_NEON

// TODO neon
uint16x8_t gf16to3v_mul_u128(uint16x8_t a, uint16x8_t b) {
    uint16x8_t m0 = vdupq_n_u8(0x00f);
    uint16x8_t m1 = vdupq_n_u8(0x0f0);
    uint16x8_t m2 = vdupq_n_u8(0xf00);
    uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);
    uint8x16_t bp = vdupq_n_u8(b);

    uint8x16_t a0 = a&m0;
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
	// mul
    poly8x16_t abl = vmulq_p8( al0 , bp );
    poly8x16_t abh = vmulq_p8( ah0 , bp );

    poly8x16_t rl = abl ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abl,4) );
    poly8x16_t rh = abh ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abh,4) );

    return vsliq_n_u8( rl , rh , 4 );
}
#endif
#undef MODULUS
