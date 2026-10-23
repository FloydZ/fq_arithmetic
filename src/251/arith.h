#pragma once

#include <stdint.h>

#define PRIME 251

/// Type definition for GF(251) field elements, one element per byte
typedef uint8_t gf251;
/// legacy name
typedef uint8_t ff_t;

/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b mod 251
static inline
gf251 gf251_add(const gf251 a, const gf251 b) {
    return (a+b) % PRIME;
}

/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b mod 251
static inline
gf251 gf251_sub(const gf251 a, const gf251 b) {
    return (a+PRIME-b) % PRIME;
}

/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b mod 251
static inline
gf251 gf251_mul(const gf251 a, const gf251 b) {
    return (a*b)%PRIME;
}

/// \param a[in]: value to negate
/// \return -a mod 251
static inline
gf251 gf251_neg(const gf251 a) {
    return (PRIME - a) % PRIME;
}

/// Inversion in GF(251), computed as a^(251-2) (constant time)
/// \param a[in]: value to invert
/// \return a^-1 mod 251, and 0 if a == 0
static inline
gf251 gf251_inv(const gf251 a) {
    const uint32_t e = PRIME - 2;
    gf251 r = 1;
    for (int i = 7; i >= 0; i--) {
        r = gf251_mul(r, r);
        const gf251 t = gf251_mul(r, a);
        const gf251 m = -(gf251)((e >> i) & 1u);
        r = (t & m) ^ (r & ~m);
    }
    return r;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// reduces every byte from [0, 256) to [0, 251)
/// NOTE: assumes that each gf251 element is in a single uint8_t
static inline
__m256i gf251v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    const __m256i t = _mm256_sub_epi8(a, Q);
    return _mm256_min_epu8(a, t);
}

/// NOTE: assumes that each gf251 element is in a single uint8_t, and reduced
static inline
__m256i gf251v_add_u256(const __m256i a,
                        const __m256i b) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    // a + b >= 251 <=> a >= 251 - b
    const __m256i d = _mm256_sub_epi8(Q, b);
    const __m256i m = _mm256_cmpeq_epi8(_mm256_max_epu8(a, d), a);
    return _mm256_blendv_epi8(_mm256_add_epi8(a, b), _mm256_sub_epi8(a, d), m);
}

/// NOTE: assumes that each gf251 element is in a single uint8_t, and reduced
/// \return a-b mod 251 in each byte
static inline
__m256i gf251v_sub_u256(const __m256i a,
                        const __m256i b) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    const __m256i d = _mm256_sub_epi8(a, b);
    // m = 0xFF <=> a >= b
    const __m256i m = _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
    return _mm256_blendv_epi8(_mm256_add_epi8(d, Q), d, m);
}

/// \param c[in]: [c_0, ..., c_15], c_i < 251^2
/// \return [c_0 mod 251, ..., c_15 mod 251]
static inline
__m256i gf251v_red_u16_u256(const __m256i c) {
    const __m256i Q = _mm256_set1_epi16(251);
    // ceil(2^23 / 251), exact for c < 133152
    const __m256i M = _mm256_set1_epi16((short)33421);
    __m256i t = _mm256_mulhi_epu16(c, M);
    t = _mm256_srli_epi16(t, 7);
    t = _mm256_mullo_epi16(t, Q);
    return _mm256_sub_epi16(c, t);
}

/// NOTE: assumes that each gf251 element is in a single uint8_t, and reduced
static inline
__m256i gf251v_mul_u256(const __m256i a,
                        const __m256i b) {
    const __m256i m = _mm256_set1_epi16(0xFF);
    const __m256i al = a & m;
    const __m256i ah = _mm256_srli_epi16(a, 8);
    const __m256i bl = b & m;
    const __m256i bh = _mm256_srli_epi16(b, 8);

    // actual multiplication
    const __m256i cl = _mm256_mullo_epi16(al, bl);
    const __m256i ch = _mm256_mullo_epi16(ah, bh);

    // reduction
    const __m256i rl = gf251v_red_u16_u256(cl);
    const __m256i rh = gf251v_red_u16_u256(ch);
    return rl ^ _mm256_slli_epi16(rh, 8);
}
#endif

#ifdef USE_AVX512
#include <immintrin.h>

/// AVX512 vectorized addition of 64 GF(251) elements
/// \param a[in]: first vector of 64 reduced elements
/// \param b[in]: second vector of 64 reduced elements
/// \return a+b mod 251 in each byte
static inline
__m512i gf251v_add_u512(const __m512i a,
                        const __m512i b) {
    const __m512i Q = _mm512_set1_epi8((char)251);
    // a + b >= 251 <=> a >= 251 - b
    const __mmask64 m = _mm512_cmpge_epu8_mask(a, _mm512_sub_epi8(Q, b));
    const __m512i s = _mm512_add_epi8(a, b);
    return _mm512_mask_sub_epi8(s, m, s, Q);
}

/// AVX512 vectorized subtraction of 64 GF(251) elements
/// \param a[in]: vector of 64 reduced minuends
/// \param b[in]: vector of 64 reduced subtrahends
/// \return a-b mod 251 in each byte
static inline
__m512i gf251v_sub_u512(const __m512i a,
                        const __m512i b) {
    const __m512i Q = _mm512_set1_epi8((char)251);
    const __mmask64 m = _mm512_cmplt_epu8_mask(a, b);
    const __m512i d = _mm512_sub_epi8(a, b);
    return _mm512_mask_add_epi8(d, m, d, Q);
}

/// \param c[in]: 32 16-bit values, c_i < 251^2
/// \return c mod 251
static inline
__m512i gf251v_red_u16_u512(const __m512i c) {
    const __m512i Q = _mm512_set1_epi16(251);
    // ceil(2^23 / 251), exact for c < 133152
    const __m512i M = _mm512_set1_epi16((short)33421);
    __m512i t = _mm512_mulhi_epu16(c, M);
    t = _mm512_srli_epi16(t, 7);
    t = _mm512_mullo_epi16(t, Q);
    return _mm512_sub_epi16(c, t);
}

/// AVX512 vectorized multiplication of 64 GF(251) elements
/// \param a[in]: first vector of 64 reduced elements
/// \param b[in]: second vector of 64 reduced elements
/// \return a*b mod 251 in each byte
static inline
__m512i gf251v_mul_u512(const __m512i a,
                        const __m512i b) {
    const __m512i m  = _mm512_set1_epi16(0xFF);
    const __m512i cl = _mm512_mullo_epi16(_mm512_and_si512(a, m), _mm512_and_si512(b, m));
    const __m512i ch = _mm512_mullo_epi16(_mm512_srli_epi16(a, 8), _mm512_srli_epi16(b, 8));
    const __m512i rl = gf251v_red_u16_u512(cl);
    const __m512i rh = gf251v_red_u16_u512(ch);
    return _mm512_or_si512(rl, _mm512_slli_epi16(rh, 8));
}
#endif

#ifdef USE_NEON
#include <arm_neon.h>

/// NEON vectorized addition of 16 GF(251) elements
/// \param a[in]: first vector of 16 reduced elements
/// \param b[in]: second vector of 16 reduced elements
/// \return a+b mod 251 in each byte
static inline
uint8x16_t gf251v_add_u128(const uint8x16_t a,
                           const uint8x16_t b) {
    const uint8x16_t Q = vdupq_n_u8(251);
    // a + b >= 251 <=> a >= 251 - b
    const uint8x16_t m = vcgeq_u8(a, vsubq_u8(Q, b));
    const uint8x16_t s = vaddq_u8(a, b);
    return vsubq_u8(s, vandq_u8(m, Q));
}

/// NEON vectorized subtraction of 16 GF(251) elements
/// \param a[in]: vector of 16 reduced minuends
/// \param b[in]: vector of 16 reduced subtrahends
/// \return a-b mod 251 in each byte
static inline
uint8x16_t gf251v_sub_u128(const uint8x16_t a,
                           const uint8x16_t b) {
    const uint8x16_t Q = vdupq_n_u8(251);
    const uint8x16_t m = vcltq_u8(a, b);
    const uint8x16_t d = vsubq_u8(a, b);
    return vaddq_u8(d, vandq_u8(m, Q));
}

/// NEON helper: reduces 8 16-bit values c < 251^2 modulo 251
/// \param c[in]: 8 16-bit values
/// \return c mod 251
static inline
uint16x8_t gf251v_red_u16_u128(const uint16x8_t c) {
    const uint16x8_t Q  = vdupq_n_u16(251);
    const uint16x8_t m8 = vdupq_n_u16(0xFF);
    // 256 = 5 mod 251
    // c <= 62500 => t1 <= 255 + 5*244 = 1475
    const uint16x8_t t1 = vmlaq_n_u16(vandq_u16(c, m8), vshrq_n_u16(c, 8), 5);
    // t2 <= 255 + 5*5 = 280
    const uint16x8_t t2 = vmlaq_n_u16(vandq_u16(t1, m8), vshrq_n_u16(t1, 8), 5);
    return vminq_u16(t2, vsubq_u16(t2, Q));
}

/// NEON vectorized multiplication of 16 GF(251) elements
/// \param a[in]: first vector of 16 reduced elements
/// \param b[in]: second vector of 16 reduced elements
/// \return a*b mod 251 in each byte
static inline
uint8x16_t gf251v_mul_u128(const uint8x16_t a,
                           const uint8x16_t b) {
    const uint16x8_t lc = vmull_u8(vget_low_u8(a), vget_low_u8(b));
    const uint16x8_t hc = vmull_high_u8(a, b);
    const uint16x8_t lr = gf251v_red_u16_u128(lc);
    const uint16x8_t hr = gf251v_red_u16_u128(hc);
    return vcombine_u8(vmovn_u16(lr), vmovn_u16(hr));
}
#endif
