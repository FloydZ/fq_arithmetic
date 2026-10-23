#pragma once

#include <stdint.h>

/// The prime field modulus
#define PRIME 31

/// Fast reduction modulo 31 for 16-bit values
/// This constant-time formula works for inputs x <= 63487 = 2^15.95
/// Uses Barrett reduction technique with precomputed constants
#define _gf31_reduce16(x) ((x) - 31 * (((uint32_t)(x) * 67651) >> 21))

/// Fast reduction modulo 31 for 32-bit values
/// This constant-time formula works for inputs x <= 4160749567 = 2^31.95
/// Uses Barrett reduction technique with precomputed constants
#define _gf31_reduce32(x) ((x) - 31 * (((uint64_t)(x) * 4433514629) >> 37))


/// Type definition for GF(31) field elements
typedef uint8_t gf31;
/// legacy name
typedef uint8_t ff_t;

/// Addition in GF(31)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b mod 31
static inline
gf31 gf31_add(const gf31 a, const gf31 b) {
    return (a+b) % PRIME;
}

/// Subtraction in GF(31)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b mod 31
static inline
gf31 gf31_sub(const gf31 a, const gf31 b) {
    return (a+PRIME-b) % PRIME;
}

/// Multiplication in GF(31)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b mod 31
static inline
gf31 gf31_mul(const gf31 a, const gf31 b) {
    return (a*b)%PRIME;
}

/// Negation in GF(31)
/// \param a[in]: value to negate
/// \return -a mod 31
static inline
gf31 gf31_neg(const gf31 a) {
    return (PRIME - a) % PRIME;
}

/// Inversion in GF(31), computed as a^(31-2) (constant time)
/// \param a[in]: value to invert
/// \return a^-1 mod 31, and 0 if a == 0
static inline
gf31 gf31_inv(const gf31 a) {
    const uint32_t e = PRIME - 2;
    gf31 r = 1;
    for (int i = 4; i >= 0; i--) {
        r = gf31_mul(r, r);
        const gf31 t = gf31_mul(r, a);
        const gf31 m = -(gf31)((e >> i) & 1u);
        r = (t & m) ^ (r & ~m);
    }
    return r;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// AVX2 vectorized partial reduction of 32 GF(31) elements
/// Reduces values to range [0,31] but doesn't reduce 31 to 0
/// \param a[in]: vector of 32 GF(31) elements to reduce
/// \return vector of reduced elements
/// \note Assumes that each GF(31) element is stored in a single uint8_t
static inline
__m256i gf31v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8(31);
    const __m256i mask = _mm256_set1_epi8(7u);
    __m256i t = _mm256_srli_epi16(a, 5);
    t &= mask;
    t &= Q;
    __m256i ret = _mm256_add_epi8(a, t);
    ret &= Q;
    return ret;
}

/// AVX2 vectorized complete reduction of 32 GF(31) elements
/// Fully reduces values in [0, 61] to range [0,30] (proper GF(31) elements)
/// \param a[in]: vector of 32 GF(31) elements to reduce
/// \return vector of completely reduced elements
/// \note Assumes that each GF(31) element is stored in a single uint8_t
static inline
__m256i gf31v_full_red_u256(const __m256i a) {
    const __m256i mod = _mm256_set1_epi8(31u);
    const __m256i one = _mm256_set1_epi8(1u);
    __m256i t;

    t = _mm256_add_epi8(a, one);
    t = _mm256_srli_epi16(t, 5);
    t = _mm256_and_si256(t, one);
    t = _mm256_add_epi8(a, t);
    t = _mm256_and_si256(t, mod);
    return t;
}

/// AVX2 vectorized addition of 32 GF(31) elements with partial reduction
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of element-wise sums with partial reduction
/// \note Not a complete reduction, as 31 is not reduced to 0
static inline
__m256i gf31v_add_partial_u256(const __m256i a,
                               const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_red_u256(c);
}

/// AVX2 vectorized addition of 32 GF(31) elements with complete reduction
/// \param a[in]: first vector of 32 GF(31) elements, each < 31
/// \param b[in]: second vector of 32 GF(31) elements, each < 31
/// \return vector of completely reduced element-wise sums, each in [0,30]
static inline
__m256i gf31v_add_u256(const __m256i a,
                       const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_full_red_u256(c);
}

/// AVX2 vectorized subtraction of 32 GF(31) elements with complete reduction
/// \param a[in]: vector of 32 GF(31) minuends, each < 31
/// \param b[in]: vector of 32 GF(31) subtrahends, each < 31
/// \return vector of a-b mod 31 in each byte, each in [0,30]
static inline
__m256i gf31v_sub_u256(const __m256i a,
                       const __m256i b) {
    const __m256i Q = _mm256_set1_epi8(31);
    // a - b in [-30, 30], +31 => [1, 61]
    const __m256i c = _mm256_add_epi8(_mm256_sub_epi8(a, b), Q);
    return gf31v_full_red_u256(c);
}

/// AVX2 vectorized multiplication of 32 GF(31) elements with partial reduction
/// Implements optimized 16-bit multiplication and reduction for GF(31) elements
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of element-wise products, each in [0, 59]
/// \note Assumes inputs have already been reduced
static inline
__m256i gf31v_mul_partial_u256(const __m256i a,
                               const __m256i b) {
    __m256i tl, th, sl, sh, r;

    const __m256i m = _mm256_set1_epi16(0xFF);
    const __m256i al = a & m;
    const __m256i ah = _mm256_srli_epi16(a, 8);
    const __m256i bl = b & m;
    const __m256i bh = _mm256_srli_epi16(b, 8);

    // actual multiplication
    const __m256i cl = _mm256_mullo_epi16(al, bl);
    const __m256i ch = _mm256_mullo_epi16(ah, bh);

    // reduction: c = 32*t + s = t + s mod 31
    tl = _mm256_srli_epi16(cl, 5);
    th = _mm256_srli_epi16(ch, 5);
    sl = _mm256_add_epi16(tl, cl);
    sh = _mm256_add_epi16(th, ch);
    tl = _mm256_slli_epi16(tl, 5);
    th = _mm256_slli_epi16(th, 5);

    r  = _mm256_sub_epi16(sl, tl);
    sh = _mm256_sub_epi16(sh, th);
    sh = _mm256_slli_epi16(sh, 8);
    r = r ^ sh;
    return r;
}

/// AVX2 vectorized multiplication of 32 GF(31) elements with complete reduction
/// \param a[in]: first vector of 32 GF(31) elements, each < 31
/// \param b[in]: second vector of 32 GF(31) elements, each < 31
/// \return vector of a*b mod 31 in each byte, each in [0,30]
static inline
__m256i gf31v_mul_u256(const __m256i a,
                       const __m256i b) {
    return gf31v_full_red_u256(gf31v_mul_partial_u256(a, b));
}
#endif

#ifdef USE_AVX512
#include <immintrin.h>

/// AVX512 vectorized addition of 64 GF(31) elements
/// \param a[in]: first vector of 64 GF(31) elements, each < 31
/// \param b[in]: second vector of 64 GF(31) elements, each < 31
/// \return a+b mod 31 in each byte
static inline
__m512i gf31v_add_u512(const __m512i a,
                       const __m512i b) {
    const __m512i Q = _mm512_set1_epi8(31);
    const __m512i t = _mm512_add_epi8(a, b);
    return _mm512_min_epu8(t, _mm512_sub_epi8(t, Q));
}

/// AVX512 vectorized subtraction of 64 GF(31) elements
/// \param a[in]: vector of 64 GF(31) minuends, each < 31
/// \param b[in]: vector of 64 GF(31) subtrahends, each < 31
/// \return a-b mod 31 in each byte
static inline
__m512i gf31v_sub_u512(const __m512i a,
                       const __m512i b) {
    const __m512i Q = _mm512_set1_epi8(31);
    const __m512i t = _mm512_add_epi8(_mm512_sub_epi8(a, b), Q);
    return _mm512_min_epu8(t, _mm512_sub_epi8(t, Q));
}

/// AVX512 vectorized multiplication of 64 GF(31) elements
/// \param a[in]: first vector of 64 GF(31) elements, each < 31
/// \param b[in]: second vector of 64 GF(31) elements, each < 31
/// \return a*b mod 31 in each byte
static inline
__m512i gf31v_mul_u512(const __m512i a,
                       const __m512i b) {
    const __m512i m  = _mm512_set1_epi16(0xFF);
    const __m512i Q  = _mm512_set1_epi16(31);
    // floor(2^16/31); the quotient estimate is at most one too small for c <= 900
    const __m512i M  = _mm512_set1_epi16(2114);

    const __m512i cl = _mm512_mullo_epi16(_mm512_and_si512(a, m), _mm512_and_si512(b, m));
    const __m512i ch = _mm512_mullo_epi16(_mm512_srli_epi16(a, 8), _mm512_srli_epi16(b, 8));

    __m512i rl = _mm512_sub_epi16(cl, _mm512_mullo_epi16(_mm512_mulhi_epu16(cl, M), Q));
    __m512i rh = _mm512_sub_epi16(ch, _mm512_mullo_epi16(_mm512_mulhi_epu16(ch, M), Q));
    rl = _mm512_min_epu16(rl, _mm512_sub_epi16(rl, Q));
    rh = _mm512_min_epu16(rh, _mm512_sub_epi16(rh, Q));
    return _mm512_or_si512(rl, _mm512_slli_epi16(rh, 8));
}
#endif

#ifdef USE_NEON
#include <arm_neon.h>

/// NEON vectorized addition of 16 GF(31) elements
/// \param a[in]: first vector of 16 GF(31) elements, each < 31
/// \param b[in]: second vector of 16 GF(31) elements, each < 31
/// \return a+b mod 31 in each byte
static inline
uint8x16_t gf31v_add_u128(const uint8x16_t a,
                          const uint8x16_t b) {
    const uint8x16_t Q = vdupq_n_u8(31);
    const uint8x16_t t = vaddq_u8(a, b);
    return vminq_u8(t, vsubq_u8(t, Q));
}

/// NEON vectorized subtraction of 16 GF(31) elements
/// \param a[in]: vector of 16 GF(31) minuends, each < 31
/// \param b[in]: vector of 16 GF(31) subtrahends, each < 31
/// \return a-b mod 31 in each byte
static inline
uint8x16_t gf31v_sub_u128(const uint8x16_t a,
                          const uint8x16_t b) {
    const uint8x16_t Q = vdupq_n_u8(31);
    const uint8x16_t t = vaddq_u8(vsubq_u8(a, b), Q);
    return vminq_u8(t, vsubq_u8(t, Q));
}

/// NEON helper: reduces 8 16-bit values c < 1024 modulo 31
/// \param c[in]: 8 16-bit values
/// \return c mod 31, each in [0,30]
static inline
uint16x8_t gf31v_red_u16_u128(const uint16x8_t c) {
    const uint16x8_t Q  = vdupq_n_u16(31);
    const uint16x8_t m5 = vdupq_n_u16(31);
    // 32 = 1 mod 31
    const uint16x8_t t = vaddq_u16(vshrq_n_u16(c, 5), vandq_u16(c, m5));
    return vminq_u16(t, vsubq_u16(t, Q));
}

/// NEON vectorized multiplication of 16 GF(31) elements
/// \param a[in]: first vector of 16 GF(31) elements, each < 31
/// \param b[in]: second vector of 16 GF(31) elements, each < 31
/// \return a*b mod 31 in each byte
static inline
uint8x16_t gf31v_mul_u128(const uint8x16_t a,
                          const uint8x16_t b) {
    const uint16x8_t lc = vmull_u8(vget_low_u8(a), vget_low_u8(b));
    const uint16x8_t hc = vmull_high_u8(a, b);
    const uint16x8_t lr = gf31v_red_u16_u128(lc);
    const uint16x8_t hr = gf31v_red_u16_u128(hc);
    return vcombine_u8(vmovn_u16(lr), vmovn_u16(hr));
}
#endif
