#pragma once 

#include <stdint.h>
#include <stdlib.h>

#include "../2/arith.h"

#if USE_AVX2 
#include <immintrin.h>
#endif

#ifdef USE_NEON
#include <arm_neon.h>

#ifdef __ARM_FEATURE_CRYPTO
/// Vectorized multiplication of GF(2^192) elements using ARM NEON
/// \param a[in]: First factor (3 64-bit limbs stored in NEON registers)
/// \param b[in]: Second factor (3 64-bit limbs stored in NEON registers)
/// \param r[out]: Result of multiplication (3 64-bit limbs)
/// \note Uses PMULL instruction for carry-less multiplication
/// \note Implements Karatsuba algorithm to reduce the number of multiplications
/// \note Requires ARMv8 with Crypto extension
static inline
void gf2to192v_mul_u256_neon(const uint64_t a[3], 
                            const uint64_t b[3],
                            uint64_t r[3]) {
    // Load the modulus into a 64-bit scalar
    const uint64_t modulus = MODULUS;
    
    // Implement Karatsuba multiplication for 3-limb numbers:
    // a = a0 + B * a1 + B^2 * a2
    // b = b0 + B * b1 + B^2 * b2
    // Need to compute c = c0 + ... + B^4 * c4 with 6 multiplications
    
    // 1. c0 = a0 * b0
    poly128_t c0 = vmull_p64((poly64_t)a[0], (poly64_t)b[0]);
    uint64x2_t c0_v = vreinterpretq_u64_p128(c0);
    
    // 2. c4 = a2 * b2
    poly128_t c4 = vmull_p64((poly64_t)a[2], (poly64_t)b[2]);
    uint64x2_t c4_v = vreinterpretq_u64_p128(c4);
    
    // 3. t = a1 * b1
    poly128_t t_poly = vmull_p64((poly64_t)a[1], (poly64_t)b[1]);
    uint64x2_t t = vreinterpretq_u64_p128(t_poly);
    
    // 4. c1 = (a0 + a1) * (b0 + b1) - c0 - t
    uint64_t a01 = a[0] ^ a[1];
    uint64_t b01 = b[0] ^ b[1];
    poly128_t c1_poly = vmull_p64((poly64_t)a01, (poly64_t)b01);
    uint64x2_t c1 = vreinterpretq_u64_p128(c1_poly);
    c1 = veorq_u64(c1, c0_v);
    c1 = veorq_u64(c1, t);
    
    // 5. c3 = (a1 + a2) * (b1 + b2) - c4 - t
    uint64_t a12 = a[1] ^ a[2];
    uint64_t b12 = b[1] ^ b[2];
    poly128_t c3_poly = vmull_p64((poly64_t)a12, (poly64_t)b12);
    uint64x2_t c3 = vreinterpretq_u64_p128(c3_poly);
    c3 = veorq_u64(c3, c4_v);
    c3 = veorq_u64(c3, t);
    
    // 6. c2 = (a0 + a2) * (b0 + b2) - c0 - c4 + t
    uint64_t a02 = a[0] ^ a[2];
    uint64_t b02 = b[0] ^ b[2];
    poly128_t c2_poly = vmull_p64((poly64_t)a02, (poly64_t)b02);
    uint64x2_t c2 = vreinterpretq_u64_p128(c2_poly);
    c2 = veorq_u64(c2, c0_v);
    c2 = veorq_u64(c2, c4_v);
    c2 = veorq_u64(c2, t);
    
    // Now compute d = d0 + B^2 * d1 + B^4 * d2 where d = c and di < B^2
    
    // Extract individual values from the vectors
    uint64_t c0_lo = vgetq_lane_u64(c0_v, 0);
    uint64_t c0_hi = vgetq_lane_u64(c0_v, 1);
    uint64_t c1_lo = vgetq_lane_u64(c1, 0);
    uint64_t c1_hi = vgetq_lane_u64(c1, 1);
    uint64_t c2_lo = vgetq_lane_u64(c2, 0);
    uint64_t c2_hi = vgetq_lane_u64(c2, 1);
    uint64_t c3_lo = vgetq_lane_u64(c3, 0);
    uint64_t c3_hi = vgetq_lane_u64(c3, 1);
    uint64_t c4_lo = vgetq_lane_u64(c4_v, 0);
    uint64_t c4_hi = vgetq_lane_u64(c4_v, 1);
    
    // Combine into d0, d1, d2
    uint64_t d0_lo = c0_lo;
    uint64_t d0_hi = c0_hi ^ c1_lo;
    uint64_t d1_lo = c1_hi ^ c2_lo;
    uint64_t d1_hi = c2_hi ^ c3_lo;
    uint64_t d2_lo = c3_hi ^ c4_lo;
    uint64_t d2_hi = c4_hi;
    
    // Reduction phase
    
    // Reduce w.r.t. high half of d2
    poly128_t tmp_poly = vmull_p64((poly64_t)d2_hi, (poly64_t)modulus);
    uint64x2_t tmp = vreinterpretq_u64_p128(tmp_poly);
    uint64_t tmp_lo = vgetq_lane_u64(tmp, 0);
    uint64_t tmp_hi = vgetq_lane_u64(tmp, 1);
    d1_lo ^= tmp_lo;
    d1_hi ^= tmp_hi;
    
    // Reduce w.r.t. low half of d2
    tmp_poly = vmull_p64((poly64_t)d2_lo, (poly64_t)modulus);
    tmp = vreinterpretq_u64_p128(tmp_poly);
    tmp_lo = vgetq_lane_u64(tmp, 0);
    tmp_hi = vgetq_lane_u64(tmp, 1);
    d1_hi ^= tmp_lo;
    d0_lo ^= tmp_hi;
    
    // Reduce w.r.t. high half of d1
    tmp_poly = vmull_p64((poly64_t)d1_hi, (poly64_t)modulus);
    tmp = vreinterpretq_u64_p128(tmp_poly);
    tmp_lo = vgetq_lane_u64(tmp, 0);
    tmp_hi = vgetq_lane_u64(tmp, 1);
    d0_lo ^= tmp_lo;
    d0_hi ^= tmp_hi;
    
    // Store the final result
    r[0] = d0_lo;
    r[1] = d0_hi;
    r[2] = d1_lo;
}

/// Vectorized multiplication of multiple GF(2^192) elements using NEON
/// \param a[in]: Array of GF(2^192) elements, each represented as 3 64-bit limbs
/// \param b[in]: Array of GF(2^192) elements, each represented as 3 64-bit limbs
/// \param r[out]: Array to store results of multiplications
/// \param count[in]: Number of elements to process
/// \note Processes multiple multiplications by calling the single-element function
static inline
void gf2to192v_mul_batch_neon(const uint64_t a[][3], 
                             const uint64_t b[][3],
                             uint64_t r[][3],
                             size_t count) {
    for (size_t i = 0; i < count; i++) {
        gf2to192v_mul_u256_neon(a[i], b[i], r[i]);
    }
}
#endif // __ARM_FEATURE_CRYPTO

#endif // USE_NEON

/// GF(2^192) field representation using polynomial basis
/// Defined using the irreducible polynomial: GF(2)/[x^192 + x^7 + x^2 + x + 1]
/// Each field element is represented as an array of 3 64-bit unsigned integers
#define MODULUS 0b10000111
typedef uint64_t gf2to192[3];

/// Set a GF(2^192) element to zero
/// \param r[out]: GF(2^192) element to be set to zero
/// \note Initializes all limbs to 0
static inline
void gf2to192_set_zero(gf2to192 r) {
    for (uint32_t i = 0; i < 3; i++) {
        r[i] = 0;
    }
}

/// Set a GF(2^192) element to a random value
/// \param r[out]: GF(2^192) element to be set to random value
/// \note Uses standard rand() function to generate random bits
static inline
void gf2to192_set_random(gf2to192 r) {
    for (uint32_t i = 0; i < 3; i++) {
        r[i] = rand() ^ ((uint64_t)rand() << 32u);
    }
}

/// Addition in GF(2^192)
/// \param r[out]: result of addition
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \note Addition in GF(2^192) is simply bitwise XOR of each limb
static inline
void gf2to192_add(gf2to192 r, 
                  const gf2to192 a,
                  const gf2to192 b) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
    r[2] = a[2] ^ b[2];
}

/// Subtraction in GF(2^192)
/// \param r[out]: result of subtraction
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \note Subtraction in GF(2^192) is the same as addition (XOR) since we're in characteristic 2
static inline
void gf2to192_sub(gf2to192 r,
                  const gf2to192 a,
                  const gf2to192 b) {
    gf2to192_add(r, a, b);
}

/// Multiplication in GF(2^192) - non-constant time version
/// \param r[out]: result of multiplication
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \note This implementation is NOT constant-time (vulnerable to timing attacks)
/// \note Uses the shift-and-add algorithm with conditional reduction by the modulus
static inline
void gf2to192_mul_v2(gf2to192 r,
                     const gf2to192 a,
                     const gf2to192 b) {
    uint64_t shifted[3] = {a[0], a[1], a[2]};
    gf2to192_set_zero(r);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            if (b[i] & (1ull << j)) {
                r[0] ^= shifted[0];
                r[1] ^= shifted[1];
                r[2] ^= shifted[2];
            }

            if (shifted[2] & (1ull << 63)) {
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = (shifted[0] << 1) ^ MODULUS;
            } else {
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = shifted[0] << 1;
            }
        }
    }
}

/// Helper function for constant-time multiplication
/// Masks a into r if bit is set, otherwise sets r to zero
/// \param r[out]: output number (masked value of a or zero)
/// \param a[in]: value to mask
/// \param bit[in]: bit which selects the mask (0 or 1)
/// \note Used as part of the constant-time multiplication algorithm
static inline
void gf2to192_mul_helper(gf2to192 r,
                         const gf2to192 a,
                         const uint64_t bit) {
    const uint64_t b = -bit;
    for (uint32_t i = 0; i < 3; i++) {
        r[i] = a[i] & b;
    }
}

/// Addition helper for constant-time multiplication
/// Adds two 3-limb big integers a and b, stores result in r
/// \param r[in,out]: result of addition, modified in-place
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \note Handles carries between limbs using 128-bit arithmetic
static inline
void gf2to192_mul_addition(gf2to192 r,
                           const gf2to192 a,
                           const gf2to192 b) {
    __uint128_t carry = 0;

    for (uint32_t i = 0; i < 3; i++) {
        const __uint128_t sum = (__uint128_t)a[i] + (__uint128_t)b[i] + carry;
        r[i] ^= sum;
        carry = (sum >> 64);
    }
}

/// Multiplication in GF(2^192) - constant time version
/// \param r[out]: result of multiplication
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \note This implementation is constant-time (resistant to timing attacks)
/// \note Implements multiplication using bit-masking to avoid branches
static inline
void gf2to192_mul(gf2to192 r,
                  const gf2to192 a,
                  const gf2to192 b) {
    gf2to192 t1;
    gf2to192_mul_helper(r, a, b[2]>>63);
    for (int i = 190; i >= 0; --i) {
        const uint32_t limb = i / 64;
        const uint32_t pos = i % 64;

        gf2to192_mul_helper(t1, a, (b[limb] >> pos) & 1ull);
        t1[0] ^= (-(r[2] >> 63)) & MODULUS;
        gf2to192_mul_addition(t1, r, r);
        for (uint32_t j = 0; j < 3; j++) { r[j] = t1[j]; }
    }
}

/// Multiplication of GF(2^192) element by GF(2) element
/// \param r[out]: result of multiplication
/// \param a[in]: GF(2^192) element
/// \param b[in]: GF(2) element (0 or 1)
/// \note Optimized for the case when the second operand is from GF(2)
/// \note Effectively sets r to either 0 or a based on the value of b
static inline
void gf2to192_mul_gf2(gf2to192 r,
                      const gf2to192 a,
                      const gf2 b) {
    const uint64_t t = -b;
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = a[i] & t;
    }
}

#ifdef USE_AVX2
#include <immintrin.h>
/// AVX2 vectorized implementation for GF(2^192) operations
/// Main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary


/// Expand GF(2) elements into 256-bit vectors for vectorized operations
/// \param out[out]: Output array of 256-bit vectors (must be at least 'limit' registers)
/// \param in[in]: Byte containing GF(2) elements (bits)
/// \param limit[in]: Number of bits to expand from the input byte
/// \note Each bit of the input is expanded to a full 256-bit mask (all 1s or all 0s)
static inline void gf2to192v_expand_gf2_x8_u256(__m256i *out,
                                                const uint8_t in,
                                                const uint32_t limit) {
    for (uint32_t i = 0; i < limit; i++) {
        out[i] = _mm256_set1_epi64x(-((in >> i) & 1u));
    }
}

/// Vectorized addition of GF(2^192) elements
/// \param a[in]: 256-bit vector of GF(2^192) elements
/// \param b[in]: 256-bit vector of GF(2^192) elements
/// \return Vector of element-wise sums
/// \note Implemented as a simple XOR operation since we're in characteristic 2
static inline
__m256i gf2to192v_add_u256(const __m256i a,
                           const __m256i b) {
    return a ^ b;
}

/// Vectorized subtraction of GF(2^192) elements
/// \param a[in]: 256-bit vector of GF(2^192) elements
/// \param b[in]: 256-bit vector of GF(2^192) elements
/// \return Vector of element-wise differences
/// \note Identical to addition (XOR) since we're in characteristic 2
static inline
__m256i gf2to192v_sub_u256(const __m256i a,
                           const __m256i b) {
    return a ^ b;
}

/// Vectorized multiplication of GF(2^192) elements
/// \param a[in]: 256-bit vector of GF(2^192) elements
/// \param b[in]: 256-bit vector of GF(2^192) elements
/// \return Vector of element-wise products
/// \note Uses Karatsuba algorithm for optimized multiplication
/// \note Implements reduction using the field's modulus polynomial
static inline
__m256i gf2to192v_mul_u256(const __m256i a,
                           const __m256i b) {
    //const __m128i ab0 = _mm_set_epi64x(a[0], b[0]);
    //const __m128i ab1 = _mm_set_epi64x(a[1], b[1]);
    //const __m128i ab2 = _mm_set_epi64x(a[2], b[2]);

    const __m256i t1 = _mm256_unpacklo_epi64(a, b);
    const __m256i t2 = _mm256_unpackhi_epi64(a, b);
    const __m128i ab0 = _mm256_castsi256_si128(t1);
    const __m128i ab1 = _mm256_castsi256_si128(t2);
    const __m128i ab2 = _mm256_extracti128_si256(t1, 1);
    const __m128i ab3 = _mm256_extracti128_si256(t2, 1);
    const __m128i modulus = _mm_set_epi64x(MODULUS, MODULUS);

    /* here we implement a Karatsuba-like approach for multiplying 3-limb numbers.
    given
      a = a0 + B * a1 + B^2 * a2
      b = b0 + B * b1 + B^2 * b2
    we can compute
      c = c0 + ... + B^4 * c4
    (where ai and bi are < B, but ci are < B^2)
    with 6 multiplications as follows:
      1. c0 = a0 * b0
      2. c4 = a2 * b2
      3. t  = a1 * b1
      4. c1 = (a0 + a1) * (b0 + b1) - c0 - t
      5. c3 = (a1 + a2) * (b1 + b2) - c4 - t
      6. c2 = (a0 + a2) * (b0 + b2) - c0 - c4 + t */
    __m128i c0 = _mm_clmulepi64_si128(ab0, ab0, 0x01); /* multiply low and high halves */
    __m128i c4 = _mm_clmulepi64_si128(ab2, ab2, 0x01);

    __m128i t = _mm_clmulepi64_si128(ab1, ab1, 0x01);

    __m128i xor01 = _mm_xor_si128(ab0, ab1);
    __m128i c1 = _mm_clmulepi64_si128(xor01, xor01, 0x01);
    c1 = _mm_xor_si128(_mm_xor_si128(c1, c0), t);

    __m128i xor12 = _mm_xor_si128(ab1, ab2);
    __m128i c3 = _mm_clmulepi64_si128(xor12, xor12, 0x01);
    c3 = _mm_xor_si128(_mm_xor_si128(c3, c4), t);

    __m128i xor02 = _mm_xor_si128(ab0, ab2);
    __m128i c2 = _mm_clmulepi64_si128(xor02, xor02, 0x01);
    c2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(c2, c0), c4), t);

    /* now let's compute
         d = d0 + B^2 * d1 + B^4 d2
       where d = c and di < B^2 */
    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d2 = _mm_xor_si128(c4, _mm_srli_si128(c3, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)),
                                _mm_slli_si128(c3, 8));

    /* done with the multiplication, time to reduce */

    /* reduce w.r.t. high half of d2 */
    __m128i tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, tmp);

    /* reduce w.r.t. low half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. high half of d1 */
    tmp = _mm_clmulepi64_si128(d1, modulus, 0x01);
    d0 = _mm_xor_si128(d0, tmp);

    return _mm256_set_m128i(d1, d0);
}

/// Alternate implementation of multiplication in GF(2^192) using Karatsuba algorithm
/// \param r[out]: Result of multiplication
/// \param a[in]: First factor
/// \param b[in]: Second factor
/// \note Uses PCLMULQDQ instruction for carry-less multiplication
/// \note Implements Karatsuba algorithm to reduce the number of multiplications
static inline
void gf2to192v_mul_u256_(gf2to192 r,
                         const gf2to192 a,
                         const gf2to192 b) {
    /* load the two operands and the modulus into 128-bit registers.
       we load corresponding limbs of both operands into a single register,
       because it lets us implement Karatsuba (see below) with fewer 128-bit
       xors. */
    const __m128i ab0 = _mm_set_epi64x(a[0], b[0]);
    const __m128i ab1 = _mm_set_epi64x(a[1], b[1]);
    const __m128i ab2 = _mm_set_epi64x(a[2], b[2]);
    const __m128i modulus = _mm_set_epi64x(MODULUS, MODULUS);

    /* here we implement a Karatsuba-like approach for multiplying 3-limb numbers.
    given
      a = a0 + B * a1 + B^2 * a2
      b = b0 + B * b1 + B^2 * b2
    we can compute
      c = c0 + ... + B^4 * c4
    (where ai and bi are < B, but ci are < B^2)
    with 6 multiplications as follows:
      1. c0 = a0 * b0
      2. c4 = a2 * b2
      3. t  = a1 * b1
      4. c1 = (a0 + a1) * (b0 + b1) - c0 - t
      5. c3 = (a1 + a2) * (b1 + b2) - c4 - t
      6. c2 = (a0 + a2) * (b0 + b2) - c0 - c4 + t */
    __m128i c0 = _mm_clmulepi64_si128(ab0, ab0, 0x01); /* multiply low and high halves */
    __m128i c4 = _mm_clmulepi64_si128(ab2, ab2, 0x01);

    __m128i t = _mm_clmulepi64_si128(ab1, ab1, 0x01);

    __m128i xor01 = _mm_xor_si128(ab0, ab1);
    __m128i c1 = _mm_clmulepi64_si128(xor01, xor01, 0x01);
    c1 = _mm_xor_si128(_mm_xor_si128(c1, c0), t);

    __m128i xor12 = _mm_xor_si128(ab1, ab2);
    __m128i c3 = _mm_clmulepi64_si128(xor12, xor12, 0x01);
    c3 = _mm_xor_si128(_mm_xor_si128(c3, c4), t);

    __m128i xor02 = _mm_xor_si128(ab0, ab2);
    __m128i c2 = _mm_clmulepi64_si128(xor02, xor02, 0x01);
    c2 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(c2, c0), c4), t);

    /* now let's compute
         d = d0 + B^2 * d1 + B^4 d2
       where d = c and di < B^2 */
    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d2 = _mm_xor_si128(c4, _mm_srli_si128(c3, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)),
                                _mm_slli_si128(c3, 8));

    /* done with the multiplication, time to reduce */

    /* reduce w.r.t. high half of d2 */
    __m128i tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, tmp);

    /* reduce w.r.t. low half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. high half of d1 */
    tmp = _mm_clmulepi64_si128(d1, modulus, 0x01);
    d0 = _mm_xor_si128(d0, tmp);

    /* done, now just store everything back into this->value_ */
    _mm_storeu_si128((__m128i*) &r[0], d0);
    _mm_storel_epi64((__m128i*) &r[2], d1);
}

/// Vectorized multiplication of GF(2^192) elements by expanded GF(2) elements
/// \param a[in]: 256-bit vector containing GF(2^192) elements
/// \param b[in]: 256-bit vector containing expanded GF(2) elements (full masks)
/// \return 256-bit vector with element-wise products
/// \note Optimized version for when b contains expanded GF(2) elements
/// \note In this case, the GF(2) elements in b are already expanded to full 256-bit masks
static inline
__m256i gf2to192v_mul_gf2_u256(const __m256i a,
                               const __m256i b) {
    return a & b;
}

#endif

#ifdef USE_NEON
#include <arm_neon.h>

/// TODO untested
/// Vectorized multiplication of GF(2^192) elements using ARM NEON
/// \param a[in]: First factor (3 64-bit limbs stored in NEON registers)
/// \param b[in]: Second factor (3 64-bit limbs stored in NEON registers)
/// \param r[out]: Result of multiplication (3 64-bit limbs)
/// \note Uses PMULL instruction for carry-less multiplication
/// \note Implements Karatsuba algorithm to reduce the number of multiplications
/// \note Requires ARMv8 with Crypto extension
static inline
void gf2to192v_mul_u256_neon(const uint64_t a[3], 
                            const uint64_t b[3],
                            uint64_t r[3]) {
    // Load the modulus into a 64-bit scalar
    const uint64_t modulus = MODULUS;
    
    // Implement Karatsuba multiplication for 3-limb numbers:
    // a = a0 + B * a1 + B^2 * a2
    // b = b0 + B * b1 + B^2 * b2
    // Need to compute c = c0 + ... + B^4 * c4 with 6 multiplications
    
    // 1. c0 = a0 * b0
    poly128_t c0 = vmull_p64((poly64_t)a[0], (poly64_t)b[0]);
    uint64x2_t c0_v = vreinterpretq_u64_p128(c0);
    
    // 2. c4 = a2 * b2
    poly128_t c4 = vmull_p64((poly64_t)a[2], (poly64_t)b[2]);
    uint64x2_t c4_v = vreinterpretq_u64_p128(c4);
    
    // 3. t = a1 * b1
    poly128_t t_poly = vmull_p64((poly64_t)a[1], (poly64_t)b[1]);
    uint64x2_t t = vreinterpretq_u64_p128(t_poly);
    
    // 4. c1 = (a0 + a1) * (b0 + b1) - c0 - t
    uint64_t a01 = a[0] ^ a[1];
    uint64_t b01 = b[0] ^ b[1];
    poly128_t c1_poly = vmull_p64((poly64_t)a01, (poly64_t)b01);
    uint64x2_t c1 = vreinterpretq_u64_p128(c1_poly);
    c1 = veorq_u64(c1, c0_v);
    c1 = veorq_u64(c1, t);
    
    // 5. c3 = (a1 + a2) * (b1 + b2) - c4 - t
    uint64_t a12 = a[1] ^ a[2];
    uint64_t b12 = b[1] ^ b[2];
    poly128_t c3_poly = vmull_p64((poly64_t)a12, (poly64_t)b12);
    uint64x2_t c3 = vreinterpretq_u64_p128(c3_poly);
    c3 = veorq_u64(c3, c4_v);
    c3 = veorq_u64(c3, t);
    
    // 6. c2 = (a0 + a2) * (b0 + b2) - c0 - c4 + t
    uint64_t a02 = a[0] ^ a[2];
    uint64_t b02 = b[0] ^ b[2];
    poly128_t c2_poly = vmull_p64((poly64_t)a02, (poly64_t)b02);
    uint64x2_t c2 = vreinterpretq_u64_p128(c2_poly);
    c2 = veorq_u64(c2, c0_v);
    c2 = veorq_u64(c2, c4_v);
    c2 = veorq_u64(c2, t);
    
    // Now compute d = d0 + B^2 * d1 + B^4 * d2 where d = c and di < B^2
    
    // Extract individual values from the vectors
    uint64_t c0_lo = vgetq_lane_u64(c0_v, 0);
    uint64_t c0_hi = vgetq_lane_u64(c0_v, 1);
    uint64_t c1_lo = vgetq_lane_u64(c1, 0);
    uint64_t c1_hi = vgetq_lane_u64(c1, 1);
    uint64_t c2_lo = vgetq_lane_u64(c2, 0);
    uint64_t c2_hi = vgetq_lane_u64(c2, 1);
    uint64_t c3_lo = vgetq_lane_u64(c3, 0);
    uint64_t c3_hi = vgetq_lane_u64(c3, 1);
    uint64_t c4_lo = vgetq_lane_u64(c4_v, 0);
    uint64_t c4_hi = vgetq_lane_u64(c4_v, 1);
    
    // Combine into d0, d1, d2
    uint64_t d0_lo = c0_lo;
    uint64_t d0_hi = c0_hi ^ c1_lo;
    uint64_t d1_lo = c1_hi ^ c2_lo;
    uint64_t d1_hi = c2_hi ^ c3_lo;
    uint64_t d2_lo = c3_hi ^ c4_lo;
    uint64_t d2_hi = c4_hi;
    
    // Reduction phase
    
    // Reduce w.r.t. high half of d2
    poly128_t tmp_poly = vmull_p64((poly64_t)d2_hi, (poly64_t)modulus);
    uint64x2_t tmp = vreinterpretq_u64_p128(tmp_poly);
    uint64_t tmp_lo = vgetq_lane_u64(tmp, 0);
    uint64_t tmp_hi = vgetq_lane_u64(tmp, 1);
    d1_lo ^= tmp_lo;
    d1_hi ^= tmp_hi;
    
    // Reduce w.r.t. low half of d2
    tmp_poly = vmull_p64((poly64_t)d2_lo, (poly64_t)modulus);
    tmp = vreinterpretq_u64_p128(tmp_poly);
    tmp_lo = vgetq_lane_u64(tmp, 0);
    tmp_hi = vgetq_lane_u64(tmp, 1);
    d1_hi ^= tmp_lo;
    d0_lo ^= tmp_hi;
    
    // Reduce w.r.t. high half of d1
    tmp_poly = vmull_p64((poly64_t)d1_hi, (poly64_t)modulus);
    tmp = vreinterpretq_u64_p128(tmp_poly);
    tmp_lo = vgetq_lane_u64(tmp, 0);
    tmp_hi = vgetq_lane_u64(tmp, 1);
    d0_lo ^= tmp_lo;
    d0_hi ^= tmp_hi;
    
    // Store the final result
    r[0] = d0_lo;
    r[1] = d0_hi;
    r[2] = d1_lo;
}

/// Vectorized multiplication of multiple GF(2^192) elements using NEON
/// \param a[in]: Array of GF(2^192) elements, each represented as 3 64-bit limbs
/// \param b[in]: Array of GF(2^192) elements, each represented as 3 64-bit limbs
/// \param r[out]: Array to store results of multiplications
/// \param count[in]: Number of elements to process
/// \note Processes multiple multiplications by calling the single-element function
static inline
void gf2to192v_mul_batch_neon(const uint64_t a[][3], 
                             const uint64_t b[][3],
                             uint64_t r[][3],
                             size_t count) {
    for (size_t i = 0; i < count; i++) {
        gf2to192v_mul_u256_neon(a[i], b[i], r[i]);
    }
}
#endif // USE_NEON
