#pragma once

#include <stdint.h>
#include "../2/arith.h"

/// GF(2^64) field representation using polynomial basis
/// Main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
///
/// Defined using the irreducible polynomial: GF(2)/[x^64 + x^4 + x^3 + x + 1]
/// Each field element is represented as a 64-bit unsigned integer
#define MODULUS 0b11011ul
typedef uint64_t gf2to64;

/// Addition in GF(2^64)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b in GF(2^64)
static inline
gf2to64 gf2to64_add(const gf2to64 a,
                    const gf2to64 b) {
    return a ^ b;
}

/// Subtraction in GF(2^64)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b in GF(2^64)
static inline
gf2to64 gf2to64_sub(const gf2to64 a,
                    const gf2to64 b) {
    return a ^ b;
}

/// Multiplication in GF(2^64) - non-constant time version
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^64)
/// \note This implementation is NOT constant-time
static inline
gf2to64 gf2to64_mul_v2(const gf2to64 a,
                       const gf2to64 b) {
    uint64_t result = 0;
    uint64_t shifted = a;

    for (uint32_t i = 0; i < 64; ++i) {
        if ((b & (1ul << i)) != 0u) {
            result ^= shifted;
        }

        if ((shifted & (1ul << 63u)) != 0u) {
            shifted <<= 1ul;
            shifted ^= MODULUS;
        } else {
            shifted <<= 1ul;
        }
    }

    return result;
}

/// Multiplication in GF(2^64) - constant time version
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^64)
/// \note This implementation is constant-time
static inline
gf2to64 gf2to64_mul(const gf2to64 a,
                    const gf2to64 b) {
    gf2to64 r = (-(b>>63u) & a);
    for (int i = 62; i >= 0; --i) {
        r = (-(b>>i & 1u) & a) ^ (-(r>>63) & MODULUS) ^ (r+r);
    }

    return r;
}

/// Negation in GF(2^64)
/// \param a[in]: value to negate
/// \return -a = a (characteristic 2)
static inline
gf2to64 gf2to64_neg(const gf2to64 a) {
    return a;
}

/// Inversion in GF(2^64), constant time: a^(2^64 - 2)
/// \param a[in]: value to invert
/// \return a^-1, and 0 if a == 0
static inline
gf2to64 gf2to64_inv(const gf2to64 a) {
    // t = a^(2^k - 1) for k = 1, ..., 63
    gf2to64 t = a;
    for (uint32_t k = 1; k < 63; k++) {
        t = gf2to64_mul(gf2to64_mul(t, t), a);
    }

    return gf2to64_mul(t, t);
}

/// Multiplication of GF(2^64) element by GF(2) element
/// \param a[in]: GF(2^64) element
/// \param b[in]: GF(2) element (0 or 1)
/// \return a*b in GF(2^64)
static inline
gf2to64 gf2to64_mul_gf2(const gf2to64 a,
                        const gf2 b) {
    return a & (-(gf2to64)b);
}

#ifdef USE_AVX2
#include <immintrin.h>

/// Vectorized addition of 4 GF(2^64) elements
/// \param a[in]: vector of 4 GF(2^64) elements [a_0, ..., a_3]
/// \param b[in]: vector of 4 GF(2^64) elements [b_0, ..., b_3]
/// \return vector of element-wise sums [a_0 + b_0, ..., a_3 + b_3]
/// \note Implemented as a simple XOR operation since we're in characteristic 2
static inline
__m256i gf2to64v_add_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// Vectorized subtraction of 4 GF(2^64) elements
/// \param a[in]: vector of 4 GF(2^64) elements [a_0, ..., a_3]
/// \param b[in]: vector of 4 GF(2^64) elements [b_0, ..., b_3]
/// \return vector of element-wise differences [a_0 - b_0, ..., a_3 - b_3]
/// \note Identical to addition (XOR) since we're in characteristic 2
static inline
__m256i gf2to64v_sub_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// Multiplication of a single pair of GF(2^64) elements using PCLMULQDQ
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^64)
/// \note Implements reduction using the field's modulus polynomial
static inline
gf2to64 gf2to64_mul_u128(const gf2to64 a,
                         const gf2to64 b) {
    const __m128i modulus = _mm_set_epi64x(0, (long long)MODULUS);
    const __m128i mul128 = _mm_clmulepi64_si128(_mm_set_epi64x(0, (long long)a),
                                                _mm_set_epi64x(0, (long long)b), 0);

    /* reduce the 64 higher order bits of mul128. Output is 96 bits since modulus < 2^64 */
    const __m128i mul96 = _mm_clmulepi64_si128(modulus, mul128, 0x10); /* use high half of mul128 */
    __m128i rem = _mm_xor_si128(mul128, mul96);

    /* reduce the 32 higher order bits of mul96 */
    const __m128i mul64 = _mm_clmulepi64_si128(modulus, mul96, 0x10); /* use high half of mul96 */

    rem = _mm_xor_si128(rem, mul64);
    return (uint64_t)_mm_cvtsi128_si64(rem);
}

/// Vectorized multiplication of 4 GF(2^64) elements
/// \param a[in]: vector of 4 GF(2^64) elements [a_0, ..., a_3]
/// \param b[in]: vector of 4 GF(2^64) elements [b_0, ..., b_3]
/// \return vector of element-wise products [a_0 * b_0, ..., a_3 * b_3]
/// \note Uses the PCLMULQDQ instruction for carry-less multiplication
/// \note Processes 4 multiplications in parallel using AVX2 and CLMUL instructions
static inline
__m256i gf2to64v_mul_u256(const __m256i a,
                          const __m256i b) {
    const __m128i modulus = _mm_set_epi64x(0, (long long)MODULUS);

    const __m128i a1 = _mm256_castsi256_si128(a);
    const __m128i a2 = _mm256_extracti128_si256(a, 0x1);
    const __m128i b1 = _mm256_castsi256_si128(b);
    const __m128i b2 = _mm256_extracti128_si256(b, 0x1);

    __m128i m11 = _mm_clmulepi64_si128(a1, b1, 0x00);
    __m128i m12 = _mm_clmulepi64_si128(a1, b1, 0x11);
    __m128i m21 = _mm_clmulepi64_si128(a2, b2, 0x00);
    __m128i m22 = _mm_clmulepi64_si128(a2, b2, 0x11);

    /* reduce the 64 higher order bits of mul128. Output is 96 bits since modulus < 2^64 */
    const __m128i m96_11 = _mm_clmulepi64_si128(modulus, m11, 0x10);
    __m128i r11 = _mm_xor_si128(m11, m96_11);
    const __m128i m96_12 = _mm_clmulepi64_si128(modulus, m12, 0x10);
    __m128i r12 = _mm_xor_si128(m12, m96_12);
    const __m128i m96_21 = _mm_clmulepi64_si128(modulus, m21, 0x10);
    __m128i r21 = _mm_xor_si128(m21, m96_21);
    const __m128i m96_22 = _mm_clmulepi64_si128(modulus, m22, 0x10);
    __m128i r22 = _mm_xor_si128(m22, m96_22);

    /* reduce the 32 higher order bits of mul96 */
    const __m128i m64_11 = _mm_clmulepi64_si128(modulus, m96_11, 0x10);
    r11 = _mm_xor_si128(r11, m64_11);
    const __m128i m64_12 = _mm_clmulepi64_si128(modulus, m96_12, 0x10);
    r12 = _mm_xor_si128(r12, m64_12);
    const __m128i m64_21 = _mm_clmulepi64_si128(modulus, m96_21, 0x10);
    r21 = _mm_xor_si128(r21, m64_21);
    const __m128i m64_22 = _mm_clmulepi64_si128(modulus, m96_22, 0x10);
    r22 = _mm_xor_si128(r22, m64_22);

    const __m128i t1 = _mm_unpacklo_epi64(r11, r12);
    const __m128i t2 = _mm_unpacklo_epi64(r21, r22);
    return _mm256_set_m128i(t2, t1);
}

/// Expand the 4 lowest GF(2) elements (bits) of `in` into 4 GF(2^64) elements
/// \param in[in]: byte containing at least 4 GF(2) elements [in_0, ..., in_3]
/// \return [in_0, ..., in_3], each 0 or 1
static inline
__m256i gf2to64v_expand_gf2_x4_u256(const uint8_t in) {
    const uint32_t t = (uint32_t)_pdep_u64(in, 0x01010101ull);
    return _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)t));
}

/// Expand the 4 lowest GF(2) elements (bits) of `in` into 4 64-bit masks
/// \param in[in]: byte containing at least 4 GF(2) elements [in_0, ..., in_3]
/// \return [-in_0, ..., -in_3], each 0 or 0xFF..FF
static inline
__m256i gf2to64v_expand_gf2_x4_ext_u256(const uint8_t in) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), gf2to64v_expand_gf2_x4_u256(in));
}

/// Vectorized multiplication of GF(2^64) elements by GF(2) elements
/// \param a[in]: vector of 4 GF(2^64) elements [a_0, ..., a_3]
/// \param b[in]: vector of 4 GF(2) elements [b_0, ..., b_3] already bit-extended
/// \return vector of products [a_0*b_0, ..., a_3*b_3]
static inline
__m256i gf2to64v_mul_gf2_u256(const __m256i a,
                              const __m256i b) {
    return a & b;
}
#endif

#ifdef USE_AVX512
#include <immintrin.h>

/// AVX512 vectorized addition of 8 GF(2^64) elements
static inline
__m512i gf2to64v_add_u512(const __m512i a,
                          const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// AVX512 vectorized subtraction of 8 GF(2^64) elements
static inline
__m512i gf2to64v_sub_u512(const __m512i a,
                          const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// AVX512 vectorized multiplication of 8 GF(2^64) elements using VPCLMULQDQ
/// \param a[in]: vector of 8 GF(2^64) elements
/// \param b[in]: vector of 8 GF(2^64) elements
/// \return vector of element-wise products
static inline
__m512i gf2to64v_mul_u512(const __m512i a,
                          const __m512i b) {
    const __m512i mod = _mm512_set1_epi64(MODULUS);

    // 128-bit products of the even/odd 64-bit positions
    const __m512i p0 = _mm512_clmulepi64_epi128(a, b, 0x00);
    const __m512i p1 = _mm512_clmulepi64_epi128(a, b, 0x11);

    // reduce the upper 64 bits: upper * MODULUS has degree < 68
    const __m512i t0 = _mm512_clmulepi64_epi128(p0, mod, 0x01);
    const __m512i t1 = _mm512_clmulepi64_epi128(p1, mod, 0x01);

    // reduce the remaining overflow of degree < 4
    const __m512i u0 = _mm512_clmulepi64_epi128(t0, mod, 0x01);
    const __m512i u1 = _mm512_clmulepi64_epi128(t1, mod, 0x01);

    // the results are in the lower 64 bit of each 128-bit lane
    const __m512i r0 = _mm512_xor_si512(_mm512_xor_si512(p0, t0), u0);
    const __m512i r1 = _mm512_xor_si512(_mm512_xor_si512(p1, t1), u1);
    return _mm512_unpacklo_epi64(r0, r1);
}
#endif

#ifdef USE_NEON
#include <arm_neon.h>

/// NEON vectorized addition of 2 GF(2^64) elements
static inline
uint64x2_t gf2to64v_add_u128(const uint64x2_t a,
                             const uint64x2_t b) {
    return veorq_u64(a, b);
}

/// NEON vectorized subtraction of 2 GF(2^64) elements
static inline
uint64x2_t gf2to64v_sub_u128(const uint64x2_t a,
                             const uint64x2_t b) {
    return veorq_u64(a, b);
}

#ifdef __ARM_FEATURE_CRYPTO
/// Multiplication in GF(2^64) using ARM PMULL instruction
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2^64)
/// \note Uses PMULL instruction for carry-less multiplication
/// \note Requires ARMv8 with Crypto extension
static inline
gf2to64 gf2to64v_mul_pmull(const gf2to64 a,
                           const gf2to64 b) {
    // Perform carry-less multiplication using PMULL
    poly128_t mul128 = vmull_p64((poly64_t)a, (poly64_t)b);

    // Extract low and high 64-bit parts
    uint64x2_t mul_result = vreinterpretq_u64_p128(mul128);
    uint64_t mul_lo = vgetq_lane_u64(mul_result, 0);
    uint64_t mul_hi = vgetq_lane_u64(mul_result, 1);

    // Reduce higher 64 bits using the modulus
    poly128_t mul96 = vmull_p64((poly64_t)MODULUS, (poly64_t)mul_hi);

    uint64x2_t mul96_result = vreinterpretq_u64_p128(mul96);
    uint64_t mul96_lo = vgetq_lane_u64(mul96_result, 0);
    uint64_t mul96_hi = vgetq_lane_u64(mul96_result, 1);

    // XOR with the original lower bits
    uint64_t rem = mul_lo ^ mul96_lo;

    // Reduce the remaining higher bits
    poly128_t mul64 = vmull_p64((poly64_t)MODULUS, (poly64_t)mul96_hi);

    uint64x2_t mul64_result = vreinterpretq_u64_p128(mul64);
    uint64_t mul64_lo = vgetq_lane_u64(mul64_result, 0);

    // Final reduction
    rem = rem ^ mul64_lo;

    return rem;
}

/// Vectorized multiplication of 2 GF(2^64) elements using PMULL
/// \param a[in]: vector of 2 GF(2^64) elements [a_0, a_1]
/// \param b[in]: vector of 2 GF(2^64) elements [b_0, b_1]
/// \return vector of element-wise products [a_0 * b_0, a_1 * b_1]
/// \note Uses PMULL instruction for carry-less multiplication
static inline
uint64x2_t gf2to64v_mul_u128_pmull(const uint64x2_t a,
                                   const uint64x2_t b) {
    // Extract individual elements
    uint64_t a0 = vgetq_lane_u64(a, 0);
    uint64_t a1 = vgetq_lane_u64(a, 1);
    uint64_t b0 = vgetq_lane_u64(b, 0);
    uint64_t b1 = vgetq_lane_u64(b, 1);

    // Perform multiplications using PMULL
    uint64_t r0 = gf2to64v_mul_pmull(a0, b0);
    uint64_t r1 = gf2to64v_mul_pmull(a1, b1);

    // Create result vector
    return vcombine_u64(vcreate_u64(r0), vcreate_u64(r1));
}

/// Vectorized multiplication of 4 GF(2^64) elements using PMULL
/// \param a[in]: array of 4 GF(2^64) elements [a_0, ..., a_3]
/// \param b[in]: array of 4 GF(2^64) elements [b_0, ..., b_3]
/// \param result[out]: array to store 4 products [a_0*b_0, ..., a_3*b_3]
/// \note Uses PMULL instruction for carry-less multiplication
static inline
void gf2to64v_mul_u256_pmull(const uint64_t a[4],
                             const uint64_t b[4],
                             uint64_t result[4]) {
    // Process first two elements
    uint64x2_t a_vec1 = vld1q_u64(&a[0]);
    uint64x2_t b_vec1 = vld1q_u64(&b[0]);
    uint64x2_t r_vec1 = gf2to64v_mul_u128_pmull(a_vec1, b_vec1);

    // Process next two elements
    uint64x2_t a_vec2 = vld1q_u64(&a[2]);
    uint64x2_t b_vec2 = vld1q_u64(&b[2]);
    uint64x2_t r_vec2 = gf2to64v_mul_u128_pmull(a_vec2, b_vec2);

    // Store results
    vst1q_u64(&result[0], r_vec1);
    vst1q_u64(&result[2], r_vec2);
}
#endif // __ARM_FEATURE_CRYPTO

/// NEON vectorized multiplication of 2 GF(2^64) elements. Uses PMULL if
/// available, otherwise a constant time shift-and-add (as `gf2to64_mul`)
/// \param a[in]: vector of 2 GF(2^64) elements
/// \param b[in]: vector of 2 GF(2^64) elements
/// \return vector of element-wise products
static inline
uint64x2_t gf2to64v_mul_u128(const uint64x2_t a,
                             const uint64x2_t b) {
#ifdef __ARM_FEATURE_CRYPTO
    return gf2to64v_mul_u128_pmull(a, b);
#else
    const uint64x2_t mod = vdupq_n_u64(MODULUS);
    // all-ones in each lane where the top bit of the lane is set
    uint64x2_t r = vandq_u64(a, vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_u64(b), 63)));
    for (int i = 62; i >= 0; --i) {
        const uint64x2_t bi = vshlq_u64(b, vdupq_n_s64(63 - i));
        const uint64x2_t mb = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_u64(bi), 63));
        const uint64x2_t mr = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_u64(r), 63));
        r = veorq_u64(veorq_u64(vandq_u64(a, mb), vandq_u64(mod, mr)), vshlq_n_u64(r, 1));
    }

    return r;
#endif
}

#endif // USE_NEON
