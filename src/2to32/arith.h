#pragma once

#include <stdint.h>
#include "../2/arith.h"

/// GF(2^32) field representation using polynomial basis
/// Main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
/// 
/// Initially defined with polynomial: GF(2)/[x^32 + x^22 + x^2 + x^1 + 1]
/// 
/// Now using a more efficient polynomial for faster reduction in 'gf2to32v_mul_u256_v2':
/// GF(2)/[x^32 + x^15 + x^7 + x^1 + 1]
#define MODULUS 0b000000000000000001000000010000011ull
typedef uint32_t gf2to32;

/// Addition in GF(2^32)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a + b mod irreducible polynomial
static inline
gf2to32 gf2to32_add(const gf2to32 a,
                    const gf2to32 b) {
    return a ^ b;
}

/// Subtraction in GF(2^32)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a - b mod irreducible polynomial (same as addition in characteristic 2)
static inline
gf2to32 gf2to32_sub(const gf2to32 a,
                    const gf2to32 b) {
    return a ^ b;
}

/// Multiplication in GF(2^32) - non constant-time implementation
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a * b mod irreducible polynomial
static inline
gf2to32 gf2to32_mul_v2(const gf2to32 a,
                       const gf2to32 b) {
    uint32_t result = 0;
    uint32_t shifted = a;

    for (uint32_t i = 0; i < 32; ++i) {
        if ((b & (1u << i)) != 0u) {
            result ^= shifted;
        }

        if ((shifted & (1u << 31u)) != 0u) {
            shifted <<= 1u;
            shifted ^= MODULUS;
        } else {
            shifted <<= 1u;
        }
    }

    return result;
}

/// Multiplication in GF(2^32) - constant-time implementation
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a * b mod irreducible polynomial
static inline
gf2to32 gf2to32_mul(const gf2to32 a,
                    const gf2to32 b) {
    gf2to32 r = (-(b>>31u) & a);
    for (int i = 30; i >= 0; --i) {
        r = (-(b>>i & 1u) & a) ^ (-(r>>31) & MODULUS) ^ (r+r);
    }

    return r;
}

/// Negation in GF(2^32)
/// \param a[in]: value to negate
/// \return -a = a (characteristic 2)
static inline
gf2to32 gf2to32_neg(const gf2to32 a) {
    return a;
}

/// Inversion in GF(2^32), constant time: a^(2^32 - 2)
/// \param a[in]: value to invert
/// \return a^-1, and 0 if a == 0
static inline
gf2to32 gf2to32_inv(const gf2to32 a) {
    // t = a^(2^k - 1) for k = 1, ..., 31
    gf2to32 t = a;
    for (uint32_t k = 1; k < 31; k++) {
        t = gf2to32_mul(gf2to32_mul(t, t), a);
    }

    return gf2to32_mul(t, t);
}

/// Multiplication of GF(2^32) element by GF(2) element
/// \param a[in]: GF(2^32) element
/// \param b[in]: GF(2) element (0 or 1)
/// \return a*b in GF(2^32)
static inline
gf2to32 gf2to32_mul_gf2(const gf2to32 a,
                        const gf2 b) {
    return a & (-b);
}

#ifdef USE_AVX2
#include <immintrin.h>

/// AVX2 vectorized addition of 8 GF(2^32) elements
/// \param a[in]: first vector of 8 GF(2^32) elements
/// \param b[in]: second vector of 8 GF(2^32) elements
/// \return vector of a+b in each 32-bit position
static inline
__m256i gf2to32v_add_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// AVX2 vectorized subtraction of 8 GF(2^32) elements
/// \param a[in]: vector of 8 GF(2^32) minuends
/// \param b[in]: vector of 8 GF(2^32) subtrahends
/// \return vector of a-b in each 32-bit position
static inline
__m256i gf2to32v_sub_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// AVX2 vectorized multiplication of 8 GF(2^32) elements
/// Uses constant-time approach with 188 instructions (2 + 6*31)
/// See: https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1AB9U8lJL6yAngGVG6AMKpaAVxYM9DgDJ4GmADl3ACNMYhAAZlIAB1QFQlsGZzcPPVj4mwFffyCWUPCoy0xrRKECJmICZPdPLgtMK0yGMoqCbMCQsMiLcsrq1LqFXra/DryuiIBKC1RXYmR2DgBSACYIv2Q3LABqJYjHPBYWPwJiPwA6BD3sJY0AQVWIrBp/bYBZAHkAETlvOSFthpglwNKCwWCQeCodCIVwuLcHvcEaYWCsAKwANjw22AVBWRAiKwAbsYWK5aMZXOiMRA0AwhtsUdTsUxSAjthzOVzuTzuXSGUzMdjgpNdgB2ABC7M5/IIjNJzO2AC8wqhdhEvoyjtTjEoCFxjJhongABwQDSTPZS%2B5c2Xy1FC7YsDDqzWkh0Y3WYfWG42EqAnSafH5/ISWiLWu5cwVY7nEV3bCAxvCTd064L0BjoEnRBRQGOTFXEVCkRMF1llhWYyYiq3SjnJ7aiYjEACeCaTVaxqe1mOMGYcObznY9hdVpZH1MmzvQE4LxHDkfr2wA9Cvthjtn4hsRXCUBApttEwttaKhUNFl/x4xATluExENFb79c9pqnxHtgBaL8p8WR3kmwPOVG31Dtkx7D1%2B0zbNjFzfMuzHYs50QitJ2rNM%2BwUWhaDwX08H9YJSwiLhf0mRdl1tYD7UVAgVgTMCljRbAmwqNs6xtXk7VAiIGPopiWMwz0mHQdB8P9MhtgXDio15eM322AgIhkwDmzbcCuxTIToMHODhwLIsS0rUcZxQ0dpIjZcljFL4EWXYhvTmBgpJk6yvg4aZaE4NFeE8DgtFIVBOEcbYFFmeZMF2NYeFIAhNA86YAGsQEkABOc4xQ0OETQiDE0RWDQVlSjF9E4SRfPiwLOF4BQQA0WL4umOBYCQNAWGNegyAoWlUHaugumQTZDGAAqQT4OgCDCWqIGCSrgj8CpW04GL5uYNsPmCbRiji7heDathBA%2BBhaCW/zeCwYJXGARwxFoWrdtILAWGG8QzsevAHJKIlMHugLMFUYpXEm5beBOBpKtw4JiEW5wsEq05DhB0hvuIYI4kwL5MGeoxcKMRq%2BAMYAFAANTwTAAHcPmPPyYv4QQRDEdgpBkQRFBUdQ3t0FZ9GGkBTGMcxIdqyBpgvJp7q/D5eK/Z6iVUb9ZdUejeFQFGziwYWIGmIp908CAHH6WpSB8UZcnyNI4gSARDYtjJEnaM2ukGBpttKYYbedxo3daB3OnCQZ3ZcGo9CGH3Tb9iRtfChY9HJwwCCpggAEkGH4EHPO8iq3qCjhtlUE0MS/DFJCbAwjG2Arzi4c4NETXBCBIKKSMmXgdq0cjSGStENHOIqTRNLgMQiSQxVSzLJEH0qOHK0g/ICnOarqhqzo7ryOBWLP5%2Bq5f2%2BmFH4jsSQgA%3D%3D%3D
/// \param a[in]: first vector of 8 GF(2^32) elements
/// \param b[in]: second vector of 8 GF(2^32) elements
/// \return vector of a*b mod irreducible polynomial in each 32-bit position
static inline
__m256i gf2to32v_mul_u256(const __m256i a,
                          const __m256i b) {
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i mod = _mm256_set1_epi32((int)MODULUS);
    __m256i     r = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)a, (__m256)b);
    __m256i carry = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)mod, (__m256)r);

    // 6 instructions per loop
    for (int i = 30; i >= 0; --i) {
        const __m256i t1 = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)a, (__m256)_mm256_slli_epi32(b, 31-i));
        const __m256i t2 = t1 ^ carry;
        const __m256i t3 = t2 ^ _mm256_add_epi32(r, r);
        r = t3;
        carry = (__m256i)_mm256_blendv_ps((__m256)zero, (__m256)mod, (__m256)r);
    }

    return r;
}

/// Helper function for GF(2^32) multiplication and reduction using CLMUL
/// Computes multiplication and reduction of two input values < 2^32
/// \param a1[in]: 128-bit vector containing two 32-bit values at positions [0,64]
///                Format: [a1_1, 0, a1_2, 0] at bit positions [0,32,64,96]
/// \param c1[in]: 128-bit vector containing two 32-bit values at positions [0,64]
///                Format: [b1_1, 0, b1_2, 0] at bit positions [0,32,64,96]
/// \return 128-bit vector with reduced products in positions [0,64]
static inline
__m128i gf2to32v_mul_u256_helper(__m128i a1,
                                 __m128i c1) {
    const __m128i mod = _mm_setr_epi64((__m64)MODULUS, (__m64)0ll);
    const __m128i m1  = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);

    __m128i a1c1t0 = _mm_clmulepi64_si128(a1, c1, 0);
    a1c1t0 ^= _mm_bslli_si128(_mm_clmulepi64_si128(a1, c1, 0x11), 8);
    __m128i a1c1t1 = _mm_srli_epi64(a1c1t0, 32);
    a1c1t0 = a1c1t0&m1;
    __m128i a1c1t2 = _mm_clmulepi64_si128(mod, a1c1t1, 0);
    a1c1t2 ^= _mm_bslli_si128(_mm_clmulepi64_si128(mod, a1c1t1, 0x10), 8);
    __m128i a1c1t3 = (a1c1t0 ^ a1c1t2) & m1;

    __m128i a1c1t4 = _mm_srli_epi64(a1c1t2, 32);
    __m128i a1c1t5 = _mm_clmulepi64_si128(mod, a1c1t4, 0);
    a1c1t5 ^= _mm_bslli_si128(_mm_clmulepi64_si128(mod, a1c1t4, 0x10), 8);
    return a1c1t3 ^ a1c1t5;
}

/// AVX2 vectorized multiplication of 8 GF(2^32) elements using CLMUL (optimized version)
/// Uses the PCLMULQDQ instruction for faster multiplication
/// \param a[in]: vector of 8 GF(2^32) elements [a_0, ..., a_7]
/// \param b[in]: vector of 8 GF(2^32) elements [b_0, ..., b_7]
/// \return vector of [a_0*b_0 mod irreducible polynomial, ..., a_7*b_7 mod irreducible polynomial]
static inline
__m256i gf2to32v_mul_u256_v2(const __m256i a,
                             const __m256i b) {
    const __m128i m1  = _mm_setr_epi32(0xFFFFFFFF, 0, 0xFFFFFFFF, 0);

    const __m128i al1 = _mm256_castsi256_si128(a);
    const __m128i bl1 = _mm256_castsi256_si128(b);
    const __m128i ah1 = _mm256_extracti128_si256(a, 0x1);
    const __m128i bh1 = _mm256_extracti128_si256(b, 0x1);

    const __m128i a1 = al1 & m1;
    const __m128i a3 = _mm_srli_epi64(al1, 32) & m1;
    const __m128i b1 = ah1 & m1;
    const __m128i b3 = _mm_srli_epi64(ah1, 32) & m1;

    const __m128i c1 = bl1 & m1;
    const __m128i c3 = _mm_srli_epi64(bl1, 32) & m1;
    const __m128i d1 = bh1 & m1;
    const __m128i d3 = _mm_srli_epi64(bh1, 32) & m1;

    __m128i a1c1 = gf2to32v_mul_u256_helper(a1, c1);
    __m128i a3c3 = _mm_slli_epi64(gf2to32v_mul_u256_helper(a3, c3), 32);
    __m128i b1d1 = gf2to32v_mul_u256_helper(b1, d1);
    __m128i b3d3 = _mm_slli_epi64(gf2to32v_mul_u256_helper(b3, d3), 32);

    a1c1 ^= a3c3;
    b1d1 ^= b3d3;
    const __m256i ret = _mm256_set_m128i(b1d1, a1c1);
    return ret;
}

/// Expand 8 GF(2) elements to 8 GF(2^32) elements
/// \param in[in]: 8-bit value containing 8 GF(2) elements [in_0, ..., in_7]
/// \return 256-bit vector containing 8 GF(2^32) elements derived from input bits
static inline
__m256i gf2to32v_expand_gf2_x8_u256(const uint8_t in) {
    const uint64_t t1 = _pdep_u64(in, 0x0101010101010101ull);
    const __m128i t2 = _mm_set1_epi64x(t1);
    return _mm256_cvtepi8_epi32(t2);
}

/// Expand 8 GF(2) elements to 8 GF(2^32) elements with bit extension
/// \param in[in]: 8-bit value containing 8 GF(2) elements [in_0, ..., in_7]
/// \return 256-bit vector containing 8 GF(2^32) elements with full bit extension
static inline
__m256i gf2to32v_expand_gf2_x8_ext_u256(const uint8_t in) {
    const __m256i t = gf2to32v_expand_gf2_x8_u256(in);
    return _mm256_cmpgt_epi32(t, _mm256_setzero_si256());
}

/// Vectorized multiplication of GF(2^32) elements by GF(2) elements
/// \param a[in]: vector of 8 GF(2^32) elements [a_0, ..., a_7]
/// \param b[in]: vector of 8 GF(2) elements [b_0, ..., b_7] already bit-extended
/// \return vector of products [a_0*b_0, ..., a_7*b_7]
static inline
__m256i gf2to32v_mul_gf2_u256(const __m256i a,
                              const __m256i b) {
    return a & b;
}
#endif

#ifdef USE_AVX512
#include <immintrin.h>

/// AVX512 vectorized addition of 16 GF(2^32) elements
/// \param a[in]: first vector of 16 GF(2^32) elements
/// \param b[in]: second vector of 16 GF(2^32) elements
/// \return vector of a+b in each 32-bit position
static inline
__m512i gf2to32v_add_u512(const __m512i a,
                          const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// AVX512 vectorized subtraction of 16 GF(2^32) elements
/// \param a[in]: vector of 16 GF(2^32) minuends
/// \param b[in]: vector of 16 GF(2^32) subtrahends
/// \return vector of a-b in each 32-bit position
static inline
__m512i gf2to32v_sub_u512(const __m512i a,
                          const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// Reduces in each 64-bit position a polynomial of degree < 63 modulo the
/// irreducible polynomial x^32 + MODULUS.
/// \param p[in]: 8 unreduced products, one per 64-bit position
/// \return 8 reduced elements in the lower 32 bits of each 64-bit position
static inline
__m512i gf2to32v_red_u512(const __m512i p) {
    const __m512i mod = _mm512_set1_epi64(MODULUS);
    const __m512i m32 = _mm512_set1_epi64(0xFFFFFFFFull);

    // hi * MODULUS has degree < 47
    const __m512i h1 = _mm512_srli_epi64(p, 32);
    const __m512i t1 = _mm512_xor_si512(_mm512_clmulepi64_epi128(h1, mod, 0x00),
                       _mm512_bslli_epi128(_mm512_clmulepi64_epi128(h1, mod, 0x11), 8));

    // the remaining overflow has degree < 15, its product degree < 30
    const __m512i h2 = _mm512_srli_epi64(t1, 32);
    const __m512i t2 = _mm512_xor_si512(_mm512_clmulepi64_epi128(h2, mod, 0x00),
                       _mm512_bslli_epi128(_mm512_clmulepi64_epi128(h2, mod, 0x11), 8));

    return _mm512_and_si512(_mm512_xor_si512(_mm512_xor_si512(p, t1), t2), m32);
}

/// AVX512 vectorized multiplication of 16 GF(2^32) elements using VPCLMULQDQ
/// \param a[in]: vector of 16 GF(2^32) elements
/// \param b[in]: vector of 16 GF(2^32) elements
/// \return vector of a*b mod irreducible polynomial in each 32-bit position
static inline
__m512i gf2to32v_mul_u512(const __m512i a,
                          const __m512i b) {
    const __m512i m32 = _mm512_set1_epi64(0xFFFFFFFFull);
    const __m512i ae = _mm512_and_si512(a, m32);
    const __m512i be = _mm512_and_si512(b, m32);
    const __m512i ao = _mm512_srli_epi64(a, 32);
    const __m512i bo = _mm512_srli_epi64(b, 32);

    // products of the even/odd 32-bit positions, one per 64-bit position
    const __m512i pe = _mm512_xor_si512(_mm512_clmulepi64_epi128(ae, be, 0x00),
                       _mm512_bslli_epi128(_mm512_clmulepi64_epi128(ae, be, 0x11), 8));
    const __m512i po = _mm512_xor_si512(_mm512_clmulepi64_epi128(ao, bo, 0x00),
                       _mm512_bslli_epi128(_mm512_clmulepi64_epi128(ao, bo, 0x11), 8));

    const __m512i re = gf2to32v_red_u512(pe);
    const __m512i ro = gf2to32v_red_u512(po);
    return _mm512_xor_si512(re, _mm512_slli_epi64(ro, 32));
}
#endif

#ifdef USE_NEON
#include <arm_neon.h>

/// NEON vectorized addition of 4 GF(2^32) elements
/// \param a[in]: first vector of 4 GF(2^32) elements
/// \param b[in]: second vector of 4 GF(2^32) elements
/// \return vector of a+b in each 32-bit position
static inline
uint32x4_t gf2to32v_add_u128(const uint32x4_t a,
                             const uint32x4_t b) {
    return veorq_u32(a, b);
}

/// NEON vectorized subtraction of 4 GF(2^32) elements
/// \param a[in]: vector of 4 GF(2^32) minuends
/// \param b[in]: vector of 4 GF(2^32) subtrahends
/// \return vector of a-b in each 32-bit position
static inline
uint32x4_t gf2to32v_sub_u128(const uint32x4_t a,
                             const uint32x4_t b) {
    return veorq_u32(a, b);
}

/// NEON vectorized multiplication of 4 GF(2^32) elements, constant time
/// shift-and-add (same algorithm as `gf2to32_mul`)
/// \param a[in]: vector of 4 GF(2^32) elements
/// \param b[in]: vector of 4 GF(2^32) elements
/// \return vector of a*b mod irreducible polynomial in each 32-bit position
static inline
uint32x4_t gf2to32v_mul_u128(const uint32x4_t a,
                             const uint32x4_t b) {
    const uint32x4_t mod = vdupq_n_u32((uint32_t)MODULUS);
    // all-ones in each lane where the top bit of the lane is set
    uint32x4_t r = vandq_u32(a, vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(b), 31)));
    for (int i = 30; i >= 0; --i) {
        const uint32x4_t bi = vshlq_u32(b, vdupq_n_s32(31 - i));
        const uint32x4_t mb = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(bi), 31));
        const uint32x4_t mr = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(r), 31));
        r = veorq_u32(veorq_u32(vandq_u32(a, mb), vandq_u32(mod, mr)), vshlq_n_u32(r, 1));
    }

    return r;
}
#endif
