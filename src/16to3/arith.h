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

    const gf16 a0 = a&0xF, a1 = (a>>4)&0xF, a2 = (a>>8)&0xF;
    const gf16 b0 = b&0xF, b1 = (b>>4)&0xF, b2 = (b>>8)&0xF;

    const gf16 p0 = gf16_mul(a0, b0);
    const gf16 p1 = gf16_mul(a1, b1);
    const gf16 p2 = gf16_mul(a2, b2);

    const ff_t a01 = gf16_add(a0, a1);
    const ff_t a12 = gf16_add(a1, a2);
    const ff_t a02 = gf16_add(a0, a2);
    const ff_t b01 = gf16_add(b0, b1);
    const ff_t b12 = gf16_add(b1, b2);
    const ff_t b02 = gf16_add(b0, b2);
    const ff_t p01 = gf16_mul(a01, b01);
    const ff_t p12 = gf16_mul(a12, b12);
    const ff_t p02 = gf16_mul(a02, b02);

    r = gf16_add(p1, p2);
    r = gf16_add(r, p12);
    r = gf16_add(r, p0);

    r^= p0 << 4;
    r^= p01 << 4;
    r^= p12 << 4;

    r^= p02 << 8;
    r^= p0 << 8;
    r^= p1 << 8;
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
///     - or that the elements are compressed together, meaning
///         we have 21 elements in a register
/// NOTE: this implementation follows the first approach
/// NOTE: we actually need that the upper 4 bits in each limb in
///     a and b are zerod out.
static inline __m256i gf16to3v_mul_u256(const __m256i a,
                                        const __m256i b) {
    __m256i r,t;
    const __m256i m0     = _mm256_set1_epi16(0x00F);
    const __m256i m1     = _mm256_set1_epi16(0x0F0);
    const __m256i m12    = _mm256_set1_epi16(0xFF0);
    const __m256i m      = _mm256_set1_epi16(0xFFF);

    // bit  0     4      8     12     16
    // pi = [a0*b0, a1*b1, a2*b2,  0  ]
    // pi = [  p0 ,   p1 ,   p2 ,  0  ]
    const __m256i pi = gf16v_mul_full_u256(a, b);

    // bit    0     4      8     12    16
    // a01_12=[a0^a1, a1^a2, a0^a2,   0 ]  = [a01, a12, a02, 0]
    // b01_12=[b0^b1, b1^b2, a0^b2,   0 ]  = [b01, b12, b02, 0]
    const __m256i a01_12 = a ^ _mm256_srli_epi16(a, 4) ^ _mm256_slli_epi16(a, 8);
    const __m256i b01_12 = b ^ _mm256_srli_epi16(b, 4) ^ _mm256_slli_epi16(b, 8);

    // bit    0   4    8   12    16
    // p012 = [p01, p12, p02,   0]
    __m256i p012 = gf16v_mul_full_u256(a01_12, b01_12);

    // bit 0   4           8         12    16
    // r = [  0, p0^p12^p01, p1^p0^p02,   0]
    r  = _mm256_slli_epi16(pi, 4);
    r ^= _mm256_slli_epi16(pi, 8);
    r ^= p012 & m12;
    r ^= _mm256_slli_epi16(p012, 4) & m1;

    t = pi ^ (_mm256_srli_epi16(pi, 4));
    t ^= _mm256_srli_epi16(pi, 8);
    t ^= _mm256_srli_epi16(p012, 4);
    t &= m0;
    r ^= t;
    return r&m;
}

// parallel hadd of 2 gf16to3 elements
static inline uint32_t gf16to3_hadd_x2_x32_u256(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 4));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(ret, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return _mm256_extract_epi32(ret, 0);
}

#endif

#ifdef USE_NEON

// lookup table for switching between gf16 and gf16to3
static const uint8_t gf16to3_neon_table1[16] __attribute__((aligned(32))) = {
    2,3,   4,5,   0,1,  16,16, // 1 2 0 inv
    10,11, 12,13, 8,9,  16,16, // 1 2 0 inv
};
static const uint8_t gf16to3_neon_table2[16] __attribute__((aligned(32))) = {
    4,5,   0,1, 0,1,  16,16,   // 2 0 0 inv
    12,11, 8,9, 8,9,  16,16,   // 2 0 0 inv
};
static const uint8_t gf16to3_neon_table3[16] __attribute__((aligned(32))) = {
    2,3,   16,16, 2,3,   16,16,// 1 inv 1 inv
    10,11, 16,16, 10,11, 16,16,// 1 inv 1 inv
};

uint16x8_t gf16to3v_mul_u128(const uint16x8_t a,
                             const uint16x8_t b) {
    const uint8x16_t ti1 = vld1q_u8(gf16to3_neon_table1);
    const uint8x16_t ti2 = vld1q_u8(gf16to3_neon_table2);
    const uint8x16_t ti3 = vld1q_u8(gf16to3_neon_table3);

    const uint16x8_t pi = gf16v_mul_u128_full(a, b);
    const uint16x8_t p2 = vqtbl1q_u8(pi, ti2);
    const uint16x8_t p3 = vqtbl1q_u8(pi, ti3);
    const uint16x8_t pp = pi ^ p2 ^ p3;

    // NOTE: replaceable with a rotate instruction
    const uint16x8_t k1 = vshrq_n_u64(a, 32);
    const uint16x8_t k2 = vshrq_n_u64(b, 32);
    const uint16x8_t a01_12 = a ^ vshl_n_u64(b, 32);
    const uint16x8_t b01_12 = b ^ tb ^ k2;
    const uint16x8_t p012 = gf16v_mul_u128_full(a01_12, b01_12);
// todo not finished
    const uint16x8_t ret = p012 ^ pp;
    return ret;
}

static inline uint16x8x2_t gf16to3v_mul_u256(const uint16x8x2_t a,
                                             const uint16x8x2_t b) {
    uint16x8x2_t r;
    r.val[0] = gf16to3v_mul_u128(a.val[0], b.val[0]);
    r.val[1] = gf16to3v_mul_u128(a.val[1], b.val[1]);
    return r;
}
#endif
#undef MODULUS
