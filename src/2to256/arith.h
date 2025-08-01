#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include "../2/vector.h"

#if USE_AVX2
#include <immintrin.h>
#endif

// main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
// Implementation of finite field arithmetic over GF(2^256)
// The field is defined by the irreducible polynomial: x^256 + x^10 + x^5 + x^2 + 1
// Binary representation of the polynomial's coefficients (excluding the x^256 term)
#define MODULUS 0b10000100101


// Field element represented as an array of 4 uint64_t values (256 bits total)
typedef uint64_t gf2to256[4];

/// Set field element to zero
/// \param r[in/out] Field element to be set to zero
static inline
void gf2to256_set_zero(gf2to256 r) {
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = 0;
    }
}

/// Set field element to a random value
/// \param r[in/out] Field element to be set to random value
static inline
void gf2to256_set_random(gf2to256 r) {
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = rand() ^ ((uint64_t)rand() << 32u);
    }
}

/// Add two field elements: In GF(2^n), addition is performed using XOR operations
/// \param r[out] Result of addition
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_add(gf2to256 r, 
                  const gf2to256 a,
                  const gf2to256 b) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
    r[2] = a[2] ^ b[2];
    r[3] = a[3] ^ b[3];
}

/// Subtract two field elements: In GF(2^n), subtraction is the same as addition (XOR)
/// \param r[out] Result of subtraction
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_sub(gf2to256 r,
                  const gf2to256 a,
                  const gf2to256 b) {
    gf2to256_add(r, a, b);
}

/// Multiply two field elements - non constant-time implementation
/// This is a schoolbook multiplication with reduction
/// NOTE: This implementation is NOT constant-time (vulnerable to timing attacks)
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_v2(gf2to256 r,
                     const gf2to256 a,
                     const gf2to256 b) {
    /// Slow, but straight-forward implementation of polynomial multiplication
    uint64_t shifted[4] = {a[0], a[1], a[2], a[3]};
    gf2to256_set_zero(r);
    for (size_t i = 0; i < 4; ++i){
        for (size_t j = 0; j < 64; ++j) {
            if (b[i] & (1ull << j)) {
                r[0] ^= shifted[0];
                r[1] ^= shifted[1];
                r[2] ^= shifted[2];
                r[3] ^= shifted[3];
            }

            if (shifted[3] & (1ull << 63)) {
                shifted[3] = (shifted[3] << 1) | (shifted[2] >> 63);
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = (shifted[0] << 1) ^ MODULUS;
            } else {
                shifted[3] = (shifted[3] << 1) | (shifted[2] >> 63);
                shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
                shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
                shifted[0] = shifted[0] << 1;
            }
        }
    }
}

/// Helper function for constant-time multiplication
/// Conditionally copies 'a' into 'r' if the specified bit is set
/// If bit is 0, r will be set to zero
/// \param r[out] Output value (either a copy of a or zero)
/// \param a[in] Value to potentially copy
/// \param bit[in] Bit which controls whether a is copied (1) or not (0)
static inline
void gf2to256_mul_helper(gf2to256 r,
                         const gf2to256 a,
                         const uint64_t bit) {
    const uint64_t b = -bit;
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = a[i] & b;
    }
}

/// Specialized addition function used in the multiplication algorithm
/// Adds two 4-limb values with carry handling, and XORs the result into r
/// \param r[out] Result is XORed into this value
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_addition(gf2to256 r,
                           const gf2to256 a,
                           const gf2to256 b) {
    __uint128_t carry = 0;

    for (uint32_t i = 0; i < 4; i++) {
        const __uint128_t sum = (__uint128_t)a[i] + (__uint128_t)b[i] + carry;
        r[i] ^= sum;
        carry = (sum >> 64);
        assert(carry <= 1);
    }
}

/// Constant-time multiplication of two field elements
/// Uses a bit-by-bit multiplication approach that processes each bit of b
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul(gf2to256 r,
                  const gf2to256 a,
                  const gf2to256 b) {
    gf2to256 t1;
    gf2to256_mul_helper(r, a, b[3]>>63);
    for (int i = 254; i >= 0; --i) {
        const uint32_t limb = i / 64;
        const uint32_t pos = i % 64;

        gf2to256_mul_helper(t1, a, (b[limb] >> pos) & 1ull);
        t1[0] ^= (-(r[3] >> 63)) & MODULUS;
        gf2to256_mul_addition(t1, r, r);
        for (uint32_t j = 0; j < 4; j++) { r[j] = t1[j]; }
    }
}

/// Multiply a field element by a GF(2) element (0 or 1)
/// This is a simplified multiplication when one operand is in GF(2)
/// \param r[out] Result of multiplication
/// \param a[in] Field element in GF(2^256)
/// \param b[in] Element in GF(2) (either 0 or 1)
static inline
void gf2to256_mul_gf2(gf2to256 r,
                      const gf2to256 a,
                      const gf2 b) {
    const uint64_t t = -b;
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = a[i] & t;
    }
}

#ifdef USE_AVX2
#include <immintrin.h>


/// Expand a GF(2) byte into an array of AVX2 vectors for parallel operations
/// Each bit of the input is expanded to a full 256-bit mask (all 1s or all 0s)
/// \param out[out] Array of vectors to hold the expanded masks (at least 'limit' elements)
/// \param in[in] Input byte containing GF(2) elements
/// \param limit[in] Number of bits to process from the input
static inline void gf2to256v_expand_gf2_x8_u256(__m256i *out,
                                                const uint8_t in,
                                                const uint32_t limit) {
    for (uint32_t i = 0; i < limit; i++) {
        out[i] = _mm256_set1_epi64x(-((in >> i) & 1u));
    }
}

/// Add two field elements using AVX2 vector operations
/// \param a[in] First field element as 256-bit vector
/// \param b[in] Second field element as 256-bit vector
/// \return Result of addition
static inline
__m256i gf2to256v_add_u256(const __m256i a,
                           const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// Subtract two field elements using AVX2 vector operations
/// In GF(2^n), subtraction is the same as addition (XOR)
/// \param a[in] First field element as 256-bit vector
/// \param b[in] Second field element as 256-bit vector
/// \return Result of subtraction
static inline
__m256i gf2to256v_sub_u256(const __m256i a,
                           const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// Multiply two field elements using AVX2 vector operations
/// Uses Karatsuba multiplication algorithm to reduce the number of multiplications
/// \param a[in] First field element as 256-bit vector
/// \param b[in] Second field element as 256-bit vector
/// \return Result of multiplication
static inline
__m256i gf2to256v_mul_u256(const __m256i a,
                           const __m256i b) {
    /* depending on the manufacturer and generation of a CPU, the PCLMUL
       instruction might take different amounts of time.
       empirically, it appears that on recent Intel CPUs, PCLMUL is so fast that
       a naive multiplicator that uses 16 PCLMULs is faster than anything more
       complicated (because time spent doing non-PCLMUL operations dominates).
       on AMD CPUs, however, more complicated multiplicators (e.g. Karatsuba,
       which uses a total of 9 multiplications) can be faster.

       thus we use a preprocessor flag to choose between a naive and a Karatsuba
       multiplicator. */
    /* here we implement a Karatsuba-like approach for multiplying 4-limb numbers.

       given
         a = a0 + B * a1 + B^2 * a2 + B^3 * a3
         b = b0 + B * b1 + B^2 * b2 + B^3 * b3
       we can compute
         c = a * b = c0 + ... + B^6 * c6
       (where ai and bi are < B, but ci are < B^2)
       with 9 multiplications as follows:
         1. c0 = a0 * b0
         2. c6 = a3 * b3
         3. t  = a1 * b1
         4. u  = a2 * b2
         5. c1 = (a0 + a1) * (b0 + b1) - c0 - t
         6. c2 = (a0 + a2) * (b0 + b2) - c0 + t - u
         7. c5 = (a2 + a3) * (b2 + b3) - c6 - u
         8. c4 = (a1 + a3) * (b1 + b3) - c6 + u - t
         9. c3 = (a0 + a1 + a2 + a3) * (b0 + b1 + b2 + b3)
                 - c0 - c1 - c2 - c4 - c5 - c6 */

    /* load the two operands and the modulus into 128-bit registers.
       we load corresponding limbs of both operands into a single register,
       because it lets us implement Karatsuba with fewer 128-bit xors. */
    // const __m128i ab0 = _mm_set_epi64x(this->value_[0], other.value_[0]);
    // const __m128i ab1 = _mm_set_epi64x(this->value_[1], other.value_[1]);
    // const __m128i ab2 = _mm_set_epi64x(this->value_[2], other.value_[2]);
    // const __m128i ab3 = _mm_set_epi64x(this->value_[3], other.value_[3]);
    const __m256i t1 = _mm256_unpacklo_epi64(a, b);
    const __m256i t2 = _mm256_unpackhi_epi64(a, b);
    const __m128i ab0 = _mm256_castsi256_si128(t1);
    const __m128i ab1 = _mm256_castsi256_si128(t2);
    const __m128i ab2 = _mm256_extracti128_si256(t1, 1);
    const __m128i ab3 = _mm256_extracti128_si256(t2, 1);
    // const __m128i modulus = _mm_loadl_epi64((const __m128i*) &(this->modulus_));
    const __m128i modulus = _mm_set_epi64x(0, MODULUS);
    __m128i c0 = _mm_clmulepi64_si128(ab0, ab0, 0x01); /* multiply low and high halves */
    __m128i c6 = _mm_clmulepi64_si128(ab3, ab3, 0x01);

    __m128i t = _mm_clmulepi64_si128(ab1, ab1, 0x01);
    __m128i u = _mm_clmulepi64_si128(ab2, ab2, 0x01);

    __m128i xor01 = _mm_xor_si128(ab0, ab1);
    __m128i c1 = _mm_clmulepi64_si128(xor01, xor01, 0x01);
    __m128i xor_c0_t = _mm_xor_si128(c0, t);
    c1 = _mm_xor_si128(c1, xor_c0_t);

    __m128i xor02 = _mm_xor_si128(ab0, ab2);
    __m128i c2 = _mm_clmulepi64_si128(xor02, xor02, 0x01);
    c2 = _mm_xor_si128(_mm_xor_si128(c2, xor_c0_t), u);

    __m128i xor23 = _mm_xor_si128(ab2, ab3);
    __m128i c5 = _mm_clmulepi64_si128(xor23, xor23, 0x01);
    __m128i xor_c6_u = _mm_xor_si128(c6, u);
    c5 = _mm_xor_si128(c5, xor_c6_u);

    __m128i xor13 = _mm_xor_si128(ab1, ab3);
    __m128i c4 = _mm_clmulepi64_si128(xor13, xor13, 0x01);
    c4 = _mm_xor_si128(_mm_xor_si128(c4, xor_c6_u), t);

    __m128i xor0123 = _mm_xor_si128(xor02, xor13);
    __m128i c3 = _mm_clmulepi64_si128(xor0123, xor0123, 0x01);
    c3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(
         _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(
         c3, c0), c1), c2), c4), c5), c6);

    /* this part is common to both multiplication algorithms:
       given the 6 overlapping 128-bit limbs such that
       a * b = c0 + (c1 << 64) + (c2 << 128) + (c3 << 192) + ... (c6 << 384)
       merge them into non-overlapping 128-bit limbs
       a * b = d0 + (d1 << 128) + (d2 << 256) + (d3 << 384) */
    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)), _mm_slli_si128(c3, 8));
    __m128i d2 = _mm_xor_si128(_mm_xor_si128(c4, _mm_srli_si128(c3, 8)), _mm_slli_si128(c5, 8));
    __m128i d3 = _mm_xor_si128(c6, _mm_srli_si128(c5, 8));

    /* done with the multiplication, time to reduce */

    /* reduce w.r.t. high half of d3 */
    __m128i tmp = _mm_clmulepi64_si128(d3, modulus, 0x01);
    d2 = _mm_xor_si128(d2, _mm_srli_si128(tmp, 8));
    d1 = _mm_xor_si128(d1, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. low half of d3 */
    tmp = _mm_clmulepi64_si128(d3, modulus, 0x00);
    d1 = _mm_xor_si128(d1, tmp);

    /* reduce w.r.t. high half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. low half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d0 = _mm_xor_si128(d0, tmp);

    /* done, now just store everything back into this->value_ */
    // _mm_storeu_si128((__m128i*) &r[0], d0);
    // _mm_storeu_si128((__m128i*) &r[2], d1);
    return (__m256i)_mm256_set_m128((__m128)d1, (__m128)d0);
}

/// Alternative multiplication implementation using a naive approach
/// Computes all pairwise products of 64-bit limbs and combines them
/// \param a[in] First field element as 256-bit vector
/// \param b[in] Second field element as 256-bit vector
/// \return Result of multiplication
static inline
__m256i gf2to256v_mul_u256_v2(const __m256i a,
                              const __m256i b) {
    /* here we compute the same c as in Karatsuba, but by just naively
       multiplying all pairs of limbs of the operands and adding together
       the results that correspond to the same shift. */
    //const __m128i a_low     = _mm_loadu_si128((const __m128i*) &(a[0]));
    //const __m128i a_high    = _mm_loadu_si128((const __m128i*) &(a[2]));
    //const __m128i b_low     = _mm_loadu_si128((const __m128i*) &(b[0]));
    //const __m128i b_high    = _mm_loadu_si128((const __m128i*) &(b[2]));
    const __m128i a_low     = _mm256_castsi256_si128(a);
    const __m128i a_high    = _mm256_extracti128_si256(a, 1);
    const __m128i b_low     = _mm256_castsi256_si128(b);
    const __m128i b_high    = _mm256_extracti128_si256(b, 1);
    const __m128i modulus   = _mm_set_epi64x(0, MODULUS);

    __m128i m00 = _mm_clmulepi64_si128(a_low, b_low, 0x00);
    __m128i m01 = _mm_clmulepi64_si128(a_low, b_low, 0x10);
    __m128i m10 = _mm_clmulepi64_si128(a_low, b_low, 0x01);
    __m128i m11 = _mm_clmulepi64_si128(a_low, b_low, 0x11);
    __m128i m20 = _mm_clmulepi64_si128(a_high, b_low, 0x00);
    __m128i m21 = _mm_clmulepi64_si128(a_high, b_low, 0x10);
    __m128i m30 = _mm_clmulepi64_si128(a_high, b_low, 0x01);
    __m128i m31 = _mm_clmulepi64_si128(a_high, b_low, 0x11);
    __m128i m02 = _mm_clmulepi64_si128(a_low, b_high, 0x00);
    __m128i m03 = _mm_clmulepi64_si128(a_low, b_high, 0x10);
    __m128i m12 = _mm_clmulepi64_si128(a_low, b_high, 0x01);
    __m128i m13 = _mm_clmulepi64_si128(a_low, b_high, 0x11);
    __m128i m22 = _mm_clmulepi64_si128(a_high, b_high, 0x00);
    __m128i m23 = _mm_clmulepi64_si128(a_high, b_high, 0x10);
    __m128i m32 = _mm_clmulepi64_si128(a_high, b_high, 0x01);
    __m128i m33 = _mm_clmulepi64_si128(a_high, b_high, 0x11);

    __m128i c0 = m00;
    __m128i c1 = _mm_xor_si128(m01, m10);
    __m128i c2 = _mm_xor_si128(_mm_xor_si128(m02, m11), m20);
    __m128i c3 = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(m03, m12), m21), m30);
    __m128i c4 = _mm_xor_si128(_mm_xor_si128(m13, m22), m31);
    __m128i c5 = _mm_xor_si128(m23, m32);
    __m128i c6 = m33;

    /* this part is common to both multiplication algorithms:
       given the 6 overlapping 128-bit limbs such that
       a * b = c0 + (c1 << 64) + (c2 << 128) + (c3 << 192) + ... (c6 << 384)
       merge them into non-overlapping 128-bit limbs
       a * b = d0 + (d1 << 128) + (d2 << 256) + (d3 << 384) */
    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)), _mm_slli_si128(c3, 8));
    __m128i d2 = _mm_xor_si128(_mm_xor_si128(c4, _mm_srli_si128(c3, 8)), _mm_slli_si128(c5, 8));
    __m128i d3 = _mm_xor_si128(c6, _mm_srli_si128(c5, 8));

    /* done with the multiplication, time to reduce */

    /* reduce w.r.t. high half of d3 */
    __m128i tmp = _mm_clmulepi64_si128(d3, modulus, 0x01);
    d2 = _mm_xor_si128(d2, _mm_srli_si128(tmp, 8));
    d1 = _mm_xor_si128(d1, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. low half of d3 */
    tmp = _mm_clmulepi64_si128(d3, modulus, 0x00);
    d1 = _mm_xor_si128(d1, tmp);

    /* reduce w.r.t. high half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    /* reduce w.r.t. low half of d2 */
    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d0 = _mm_xor_si128(d0, tmp);

    /* done, now just store everything back into this->value_ */
    // _mm_storeu_si128((__m128i*) &r[0], d0);
    // _mm_storeu_si128((__m128i*) &r[2], d1);
    return (__m256i)_mm256_set_m128((__m128)d1, (__m128)d0);
}

/// Wrapper function for vector multiplication of field elements
/// Loads inputs from memory, calls the vector multiplication, and stores result
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256v_mul_u256_(gf2to256 r,
                         const gf2to256 a,
                         const gf2to256 b) {
    const __m256i aa = _mm256_loadu_si256((const __m256i *)a);
    const __m256i bb = _mm256_loadu_si256((const __m256i *)b);
    // choose which one is faster
    const __m256i cc = gf2to256v_mul_u256(aa, bb);
    // const __m256i cc = gf2to256v_mul_u256_v2(aa, bb);
    _mm256_storeu_si256((__m256i *)r, cc);
}

/// Multiply a field element by a GF(2) element using AVX2 vector operations
/// \param a Field element in GF(2^256) as 256-bit vector
/// \param b Expanded GF(2) element (all 0s or all 1s) as 256-bit vector
/// \return Result of multiplication
static inline
__m256i gf2to256v_mul_gf2_u256(const __m256i a,
                               const __m256i b) {
    return a & b;
}

#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
#include <arm_neon.h>

// Helper function to perform carryless multiplication of two 64-bit values
static inline
poly128_t clmul_neon(uint64_t a, uint64_t b) {
    return vmull_p64((poly64_t)a, (poly64_t)b);
}

// Helper function to XOR two 128-bit values and return a 128-bit result
static inline
uint64x2_t xor_u128(uint64x2_t a, uint64x2_t b) {
    return veorq_u64(a, b);
}

// Helper function to perform 128-bit shift left
static inline
uint64x2_t slli_u128(uint64x2_t a, int count) {
    if (count == 8) {
        // Shift left by 8 bytes (64 bits)
        return vextq_u64(vdupq_n_u64(0), a, 1);
    }
    // Other shift amounts would require more complex handling
    // Not implemented for simplicity
    return a;
}

// Helper function to perform 128-bit shift right
static inline
uint64x2_t srli_u128(uint64x2_t a, int count) {
    if (count == 8) {
        // Shift right by 8 bytes (64 bits)
        return vextq_u64(a, vdupq_n_u64(0), 1);
    }
    // Other shift amounts would require more complex handling
    // Not implemented for simplicity
    return a;
}

/// TODO untested
/// Multiply two field elements using ARM NEON instructions (PMULL)
/// Uses Karatsuba-like approach for multiplying 4-limb numbers
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_neon(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    // Load the input values as 128-bit NEON vectors
    uint64x2_t a_low = vld1q_u64(&a[0]);
    uint64x2_t a_high = vld1q_u64(&a[2]);
    uint64x2_t b_low = vld1q_u64(&b[0]);
    uint64x2_t b_high = vld1q_u64(&b[2]);
    
    // Prepare for Karatsuba-like multiplication with 9 multiplications
    // Extract individual 64-bit limbs for easier access
    uint64_t a0 = vgetq_lane_u64(a_low, 0);
    uint64_t a1 = vgetq_lane_u64(a_low, 1);
    uint64_t a2 = vgetq_lane_u64(a_high, 0);
    uint64_t a3 = vgetq_lane_u64(a_high, 1);
    
    uint64_t b0 = vgetq_lane_u64(b_low, 0);
    uint64_t b1 = vgetq_lane_u64(b_low, 1);
    uint64_t b2 = vgetq_lane_u64(b_high, 0);
    uint64_t b3 = vgetq_lane_u64(b_high, 1);
    
    // Compute the 9 multiplications needed for Karatsuba
    // 1. c0 = a0 * b0
    poly128_t c0 = clmul_neon(a0, b0);
    uint64x2_t c0v = vreinterpretq_u64_p128(c0);
    
    // 2. c6 = a3 * b3
    poly128_t c6 = clmul_neon(a3, b3);
    uint64x2_t c6v = vreinterpretq_u64_p128(c6);
    
    // 3. t = a1 * b1
    poly128_t t = clmul_neon(a1, b1);
    uint64x2_t tv = vreinterpretq_u64_p128(t);
    
    // 4. u = a2 * b2
    poly128_t u = clmul_neon(a2, b2);
    uint64x2_t uv = vreinterpretq_u64_p128(u);
    
    // 5. c1 = (a0 + a1) * (b0 + b1) - c0 - t
    poly128_t tmp = clmul_neon(a0 ^ a1, b0 ^ b1);
    uint64x2_t c1v = xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c0v), tv);
    
    // 6. c2 = (a0 + a2) * (b0 + b2) - c0 + t - u
    tmp = clmul_neon(a0 ^ a2, b0 ^ b2);
    uint64x2_t c2v = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c0v), tv), uv);
    
    // 7. c5 = (a2 + a3) * (b2 + b3) - c6 - u
    tmp = clmul_neon(a2 ^ a3, b2 ^ b3);
    uint64x2_t c5v = xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c6v), uv);
    
    // 8. c4 = (a1 + a3) * (b1 + b3) - c6 + u - t
    tmp = clmul_neon(a1 ^ a3, b1 ^ b3);
    uint64x2_t c4v = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(tmp), c6v), uv), tv);
    
    // 9. c3 = (a0 + a1 + a2 + a3) * (b0 + b1 + b2 + b3) - c0 - c1 - c2 - c4 - c5 - c6
    tmp = clmul_neon(a0 ^ a1 ^ a2 ^ a3, b0 ^ b1 ^ b2 ^ b3);
    uint64x2_t c3v = xor_u128(xor_u128(xor_u128(xor_u128(xor_u128(
                     xor_u128(vreinterpretq_u64_p128(tmp), c0v), c1v), c2v), c4v), c5v), c6v);
    
    // Merge the 6 overlapping 128-bit limbs into non-overlapping 128-bit limbs
    uint64x2_t d0 = xor_u128(c0v, slli_u128(c1v, 8));
    uint64x2_t d1 = xor_u128(xor_u128(c2v, srli_u128(c1v, 8)), slli_u128(c3v, 8));
    uint64x2_t d2 = xor_u128(xor_u128(c4v, srli_u128(c3v, 8)), slli_u128(c5v, 8));
    uint64x2_t d3 = xor_u128(c6v, srli_u128(c5v, 8));
    
    // Perform reduction with the GF(2^256) modulus
    const uint64_t modulus = MODULUS; // 0b10000100101
    
    // Reduce w.r.t. high half of d3
    poly128_t tmp1 = clmul_neon(vgetq_lane_u64(d3, 1), modulus);
    uint64x2_t tmp1v = vreinterpretq_u64_p128(tmp1);
    d2 = xor_u128(d2, srli_u128(tmp1v, 8));
    d1 = xor_u128(d1, slli_u128(tmp1v, 8));
    
    // Reduce w.r.t. low half of d3
    poly128_t tmp2 = clmul_neon(vgetq_lane_u64(d3, 0), modulus);
    uint64x2_t tmp2v = vreinterpretq_u64_p128(tmp2);
    d1 = xor_u128(d1, tmp2v);
    
    // Reduce w.r.t. high half of d2
    poly128_t tmp3 = clmul_neon(vgetq_lane_u64(d2, 1), modulus);
    uint64x2_t tmp3v = vreinterpretq_u64_p128(tmp3);
    d1 = xor_u128(d1, srli_u128(tmp3v, 8));
    d0 = xor_u128(d0, slli_u128(tmp3v, 8));
    
    // Reduce w.r.t. low half of d2
    poly128_t tmp4 = clmul_neon(vgetq_lane_u64(d2, 0), modulus);
    uint64x2_t tmp4v = vreinterpretq_u64_p128(tmp4);
    d0 = xor_u128(d0, tmp4v);
    
    // Store the final result
    vst1q_u64(&r[0], d0);
    vst1q_u64(&r[2], d1);
}

/// Wrapper function for vector multiplication of field elements
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256v_mul_neon_(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    gf2to256_mul_neon(r, a, b);
}

/// Alternative version using naive multiplication approach with NEON
/// \param r[out] Result of multiplication
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256v_mul_neon_v2(gf2to256 r, const gf2to256 a, const gf2to256 b) {
    // Load input values
    uint64x2_t a_low = vld1q_u64(&a[0]);
    uint64x2_t a_high = vld1q_u64(&a[2]);
    uint64x2_t b_low = vld1q_u64(&b[0]);
    uint64x2_t b_high = vld1q_u64(&b[2]);
    
    // Extract individual 64-bit limbs
    uint64_t a0 = vgetq_lane_u64(a_low, 0);
    uint64_t a1 = vgetq_lane_u64(a_low, 1);
    uint64_t a2 = vgetq_lane_u64(a_high, 0);
    uint64_t a3 = vgetq_lane_u64(a_high, 1);
    
    uint64_t b0 = vgetq_lane_u64(b_low, 0);
    uint64_t b1 = vgetq_lane_u64(b_low, 1);
    uint64_t b2 = vgetq_lane_u64(b_high, 0);
    uint64_t b3 = vgetq_lane_u64(b_high, 1);
    
    // Compute all 16 pairwise products with PMULL
    poly128_t m00 = clmul_neon(a0, b0);
    poly128_t m01 = clmul_neon(a0, b1);
    poly128_t m02 = clmul_neon(a0, b2);
    poly128_t m03 = clmul_neon(a0, b3);
    poly128_t m10 = clmul_neon(a1, b0);
    poly128_t m11 = clmul_neon(a1, b1);
    poly128_t m12 = clmul_neon(a1, b2);
    poly128_t m13 = clmul_neon(a1, b3);
    poly128_t m20 = clmul_neon(a2, b0);
    poly128_t m21 = clmul_neon(a2, b1);
    poly128_t m22 = clmul_neon(a2, b2);
    poly128_t m23 = clmul_neon(a2, b3);
    poly128_t m30 = clmul_neon(a3, b0);
    poly128_t m31 = clmul_neon(a3, b1);
    poly128_t m32 = clmul_neon(a3, b2);
    poly128_t m33 = clmul_neon(a3, b3);
    
    // Combine products to get overlapping 128-bit limbs
    uint64x2_t c0 = vreinterpretq_u64_p128(m00);
    uint64x2_t c1 = xor_u128(vreinterpretq_u64_p128(m01), vreinterpretq_u64_p128(m10));
    uint64x2_t c2 = xor_u128(xor_u128(vreinterpretq_u64_p128(m02), vreinterpretq_u64_p128(m11)), 
                           vreinterpretq_u64_p128(m20));
    uint64x2_t c3 = xor_u128(xor_u128(xor_u128(vreinterpretq_u64_p128(m03), vreinterpretq_u64_p128(m12)), 
                           vreinterpretq_u64_p128(m21)), vreinterpretq_u64_p128(m30));
    uint64x2_t c4 = xor_u128(xor_u128(vreinterpretq_u64_p128(m13), vreinterpretq_u64_p128(m22)), 
                           vreinterpretq_u64_p128(m31));
    uint64x2_t c5 = xor_u128(vreinterpretq_u64_p128(m23), vreinterpretq_u64_p128(m32));
    uint64x2_t c6 = vreinterpretq_u64_p128(m33);
    
    // Merge into non-overlapping 128-bit limbs
    uint64x2_t d0 = xor_u128(c0, slli_u128(c1, 8));
    uint64x2_t d1 = xor_u128(xor_u128(c2, srli_u128(c1, 8)), slli_u128(c3, 8));
    uint64x2_t d2 = xor_u128(xor_u128(c4, srli_u128(c3, 8)), slli_u128(c5, 8));
    uint64x2_t d3 = xor_u128(c6, srli_u128(c5, 8));
    
    // Reduction with the GF(2^256) modulus
    const uint64_t modulus = MODULUS;
    
    // Reduce with respect to each part of d2 and d3
    poly128_t tmp;
    uint64x2_t tmpv;
    
    // Reduce w.r.t. high half of d3
    tmp = clmul_neon(vgetq_lane_u64(d3, 1), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d2 = xor_u128(d2, srli_u128(tmpv, 8));
    d1 = xor_u128(d1, slli_u128(tmpv, 8));
    
    // Reduce w.r.t. low half of d3
    tmp = clmul_neon(vgetq_lane_u64(d3, 0), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d1 = xor_u128(d1, tmpv);
    
    // Reduce w.r.t. high half of d2
    tmp = clmul_neon(vgetq_lane_u64(d2, 1), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d1 = xor_u128(d1, srli_u128(tmpv, 8));
    d0 = xor_u128(d0, slli_u128(tmpv, 8));
    
    // Reduce w.r.t. low half of d2
    tmp = clmul_neon(vgetq_lane_u64(d2, 0), modulus);
    tmpv = vreinterpretq_u64_p128(tmp);
    d0 = xor_u128(d0, tmpv);
    
    // Store the final result
    vst1q_u64(&r[0], d0);
    vst1q_u64(&r[2], d1);
}

#endif // defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
