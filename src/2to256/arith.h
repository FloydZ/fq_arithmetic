#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../2/arith.h"

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#ifdef USE_NEON
#include <arm_neon.h>
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
        r[i] = rand() ^ ((uint64_t)rand() << 32u) ^ ((uint64_t)rand() << 62u);
    }
}

/// \param r[out]: = a
/// \param a[in]: element to copy
static inline
void gf2to256_copy(gf2to256 r,
                   const gf2to256 a) {
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}

/// \return 1 if a == b, 0 otherwise
static inline
int gf2to256_is_equal(const gf2to256 a,
                      const gf2to256 b) {
    return (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]) & (a[3] == b[3]);
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

/// Negation in GF(2^256)
/// \param r[out]: -a, which is a in characteristic 2
/// \param a[in]: element to negate
static inline
void gf2to256_neg(gf2to256 r,
                  const gf2to256 a) {
    gf2to256_copy(r, a);
}

/// Multiply two field elements - non constant-time implementation
/// This is a schoolbook multiplication with reduction
/// NOTE: This implementation is NOT constant-time (vulnerable to timing attacks)
/// \param r[out] Result of multiplication (may alias a or b)
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_v2(gf2to256 r,
                     const gf2to256 a,
                     const gf2to256 b) {
    /// Slow, but straight-forward implementation of polynomial multiplication
    uint64_t shifted[4] = {a[0], a[1], a[2], a[3]};
    const uint64_t bb[4] = {b[0], b[1], b[2], b[3]};
    uint64_t res[4] = {0};
    for (size_t i = 0; i < 4; ++i){
        for (size_t j = 0; j < 64; ++j) {
            if (bb[i] & (1ull << j)) {
                res[0] ^= shifted[0];
                res[1] ^= shifted[1];
                res[2] ^= shifted[2];
                res[3] ^= shifted[3];
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

    gf2to256_copy(r, res);
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
    }
}

/// Constant-time multiplication of two field elements
/// Uses a bit-by-bit multiplication approach that processes each bit of b
/// \param r[out] Result of multiplication (may alias a or b)
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul(gf2to256 r,
                  const gf2to256 a,
                  const gf2to256 b) {
    gf2to256 t1, aa, bb, res;
    gf2to256_copy(aa, a);
    gf2to256_copy(bb, b);
    gf2to256_mul_helper(res, aa, bb[3]>>63);
    for (int i = 254; i >= 0; --i) {
        const uint32_t limb = i / 64;
        const uint32_t pos = i % 64;

        gf2to256_mul_helper(t1, aa, (bb[limb] >> pos) & 1ull);
        t1[0] ^= (-(res[3] >> 63)) & MODULUS;
        gf2to256_mul_addition(t1, res, res);
        gf2to256_copy(res, t1);
    }

    gf2to256_copy(r, res);
}

/// Inversion in GF(2^256): a^(2^256 - 2), constant time
/// \param r[out]: a^{-1}, or 0 if a = 0 (may alias a)
/// \param a[in]: element to invert
static inline
void gf2to256_inv(gf2to256 r,
                  const gf2to256 a) {
    gf2to256 y, aa;
    gf2to256_copy(aa, a);
    gf2to256_copy(y, aa);
    // y = a^(2^255 - 1)
    for (uint32_t i = 0; i < 254; i++) {
        gf2to256_mul(y, y, y);
        gf2to256_mul(y, y, aa);
    }
    // r = a^(2^256 - 2)
    gf2to256_mul(r, y, y);
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
    const uint64_t t = -(uint64_t)b;
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = a[i] & t;
    }
}

#ifdef USE_AVX2
/// Lane layout of a `_u256` value: ONE element (limbs 0..3), i.e. the
/// memory layout of a gf2to256.

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
    const __m256i t1 = _mm256_unpacklo_epi64(a, b);
    const __m256i t2 = _mm256_unpackhi_epi64(a, b);
    const __m128i ab0 = _mm256_castsi256_si128(t1);
    const __m128i ab1 = _mm256_castsi256_si128(t2);
    const __m128i ab2 = _mm256_extracti128_si256(t1, 1);
    const __m128i ab3 = _mm256_extracti128_si256(t2, 1);
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

    return _mm256_set_m128i(d1, d0);
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

    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)), _mm_slli_si128(c3, 8));
    __m128i d2 = _mm_xor_si128(_mm_xor_si128(c4, _mm_srli_si128(c3, 8)), _mm_slli_si128(c5, 8));
    __m128i d3 = _mm_xor_si128(c6, _mm_srli_si128(c5, 8));

    __m128i tmp = _mm_clmulepi64_si128(d3, modulus, 0x01);
    d2 = _mm_xor_si128(d2, _mm_srli_si128(tmp, 8));
    d1 = _mm_xor_si128(d1, _mm_slli_si128(tmp, 8));

    tmp = _mm_clmulepi64_si128(d3, modulus, 0x00);
    d1 = _mm_xor_si128(d1, tmp);

    tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d0 = _mm_xor_si128(d0, tmp);

    return _mm256_set_m128i(d1, d0);
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

#endif // USE_AVX2

#ifdef USE_AVX512
/// Lane layout of a `_u512` value: TWO consecutive elements (limbs 0..3 and
/// 4..7), i.e. the memory layout of `gf2to256[2]`.

/// \return a + b for two elements
static inline
__m512i gf2to256v_add_u512(const __m512i a,
                           const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// \return a - b for two elements
static inline
__m512i gf2to256v_sub_u512(const __m512i a,
                           const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// Multiplication of two pairs of GF(2^256) elements. Same Karatsuba
/// algorithm as `gf2to256v_mul_u256`, but each 128-bit lane of the 256-bit
/// VPCLMULQDQ registers works on a different element.
/// \return a*b element-wise
static inline
__m512i gf2to256v_mul_u512(const __m512i a,
                           const __m512i b) {
    /* ab_k: lane e = [a_k(e), b_k(e)] */
    const __m512i idx01 = _mm512_setr_epi64(0, 8, 4, 12, 1, 9, 5, 13);
    const __m512i idx23 = _mm512_setr_epi64(2, 10, 6, 14, 3, 11, 7, 15);
    const __m512i q = _mm512_permutex2var_epi64(a, idx01, b);
    const __m512i s = _mm512_permutex2var_epi64(a, idx23, b);
    const __m256i ab0 = _mm512_castsi512_si256(q);
    const __m256i ab1 = _mm512_extracti64x4_epi64(q, 1);
    const __m256i ab2 = _mm512_castsi512_si256(s);
    const __m256i ab3 = _mm512_extracti64x4_epi64(s, 1);
    const __m256i modulus = _mm256_set_epi64x(0, MODULUS, 0, MODULUS);

    const __m256i c0 = _mm256_clmulepi64_epi128(ab0, ab0, 0x01);
    const __m256i c6 = _mm256_clmulepi64_epi128(ab3, ab3, 0x01);
    const __m256i t  = _mm256_clmulepi64_epi128(ab1, ab1, 0x01);
    const __m256i u  = _mm256_clmulepi64_epi128(ab2, ab2, 0x01);

    const __m256i xor01 = _mm256_xor_si256(ab0, ab1);
    __m256i c1 = _mm256_clmulepi64_epi128(xor01, xor01, 0x01);
    const __m256i xor_c0_t = _mm256_xor_si256(c0, t);
    c1 = _mm256_xor_si256(c1, xor_c0_t);

    const __m256i xor02 = _mm256_xor_si256(ab0, ab2);
    __m256i c2 = _mm256_clmulepi64_epi128(xor02, xor02, 0x01);
    c2 = _mm256_xor_si256(_mm256_xor_si256(c2, xor_c0_t), u);

    const __m256i xor23 = _mm256_xor_si256(ab2, ab3);
    __m256i c5 = _mm256_clmulepi64_epi128(xor23, xor23, 0x01);
    const __m256i xor_c6_u = _mm256_xor_si256(c6, u);
    c5 = _mm256_xor_si256(c5, xor_c6_u);

    const __m256i xor13 = _mm256_xor_si256(ab1, ab3);
    __m256i c4 = _mm256_clmulepi64_epi128(xor13, xor13, 0x01);
    c4 = _mm256_xor_si256(_mm256_xor_si256(c4, xor_c6_u), t);

    const __m256i xor0123 = _mm256_xor_si256(xor02, xor13);
    __m256i c3 = _mm256_clmulepi64_epi128(xor0123, xor0123, 0x01);
    c3 = c3 ^ c0 ^ c1 ^ c2 ^ c4 ^ c5 ^ c6;

    __m256i d0 = c0 ^ _mm256_bslli_epi128(c1, 8);
    __m256i d1 = c2 ^ _mm256_bsrli_epi128(c1, 8) ^ _mm256_bslli_epi128(c3, 8);
    __m256i d2 = c4 ^ _mm256_bsrli_epi128(c3, 8) ^ _mm256_bslli_epi128(c5, 8);
    const __m256i d3 = c6 ^ _mm256_bsrli_epi128(c5, 8);

    __m256i tmp = _mm256_clmulepi64_epi128(d3, modulus, 0x01);
    d2 ^= _mm256_bsrli_epi128(tmp, 8);
    d1 ^= _mm256_bslli_epi128(tmp, 8);

    tmp = _mm256_clmulepi64_epi128(d3, modulus, 0x00);
    d1 ^= tmp;

    tmp = _mm256_clmulepi64_epi128(d2, modulus, 0x01);
    d1 ^= _mm256_bsrli_epi128(tmp, 8);
    d0 ^= _mm256_bslli_epi128(tmp, 8);

    tmp = _mm256_clmulepi64_epi128(d2, modulus, 0x00);
    d0 ^= tmp;

    /* element e = [d0(e), d1(e)] */
    const __m512i d = _mm512_inserti64x4(_mm512_castsi256_si512(d0), d1, 1);
    const __m512i idx = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
    return _mm512_permutexvar_epi64(idx, d);
}
#endif // USE_AVX512

#ifdef USE_NEON
/// Lane layout of a `_u128` value: a GF(2^256) element does not fit into a
/// single NEON register, hence `uint64x2x2_t` is used: val[0] = limbs 0,1,
/// val[1] = limbs 2,3.

/// \param a[in]: element in memory
static inline
uint64x2x2_t gf2to256v_load_u128(const gf2to256 a) {
    uint64x2x2_t r;
    r.val[0] = vld1q_u64(a);
    r.val[1] = vld1q_u64(a + 2);
    return r;
}

/// \param r[out]: = a
/// \param a[in]: register pair
static inline
void gf2to256v_store_u128(gf2to256 r,
                          const uint64x2x2_t a) {
    vst1q_u64(r, a.val[0]);
    vst1q_u64(r + 2, a.val[1]);
}

/// \return a + b
static inline
uint64x2x2_t gf2to256v_add_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    uint64x2x2_t r;
    r.val[0] = veorq_u64(a.val[0], b.val[0]);
    r.val[1] = veorq_u64(a.val[1], b.val[1]);
    return r;
}

/// \return a - b
static inline
uint64x2x2_t gf2to256v_sub_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    return gf2to256v_add_u128(a, b);
}

#ifdef __ARM_FEATURE_CRYPTO
/// \return carry-less product of two 64-bit values as (lo, hi)
static inline
uint64x2_t gf2to256_clmul_neon(const uint64_t a,
                               const uint64_t b) {
    return vreinterpretq_u64_p128(vmull_p64((poly64_t)a, (poly64_t)b));
}

/// \return a << 64 (128-bit shift by 8 bytes)
static inline
uint64x2_t gf2to256_slli8_neon(const uint64x2_t a) {
    return vextq_u64(vdupq_n_u64(0), a, 1);
}

/// \return a >> 64 (128-bit shift by 8 bytes)
static inline
uint64x2_t gf2to256_srli8_neon(const uint64x2_t a) {
    return vextq_u64(a, vdupq_n_u64(0), 1);
}

/// Reduces the 512-bit product d0 + d1 B^2 + d2 B^4 + d3 B^6 (B = 2^64)
/// \param r[out]: reduced result
static inline
void gf2to256_reduce_neon(gf2to256 r,
                          uint64x2_t d0,
                          uint64x2_t d1,
                          uint64x2_t d2,
                          const uint64x2_t d3) {
    const uint64_t modulus = MODULUS;

    // Reduce w.r.t. high half of d3
    uint64x2_t tmp = gf2to256_clmul_neon(vgetq_lane_u64(d3, 1), modulus);
    d2 = veorq_u64(d2, gf2to256_srli8_neon(tmp));
    d1 = veorq_u64(d1, gf2to256_slli8_neon(tmp));

    // Reduce w.r.t. low half of d3
    tmp = gf2to256_clmul_neon(vgetq_lane_u64(d3, 0), modulus);
    d1 = veorq_u64(d1, tmp);

    // Reduce w.r.t. high half of d2
    tmp = gf2to256_clmul_neon(vgetq_lane_u64(d2, 1), modulus);
    d1 = veorq_u64(d1, gf2to256_srli8_neon(tmp));
    d0 = veorq_u64(d0, gf2to256_slli8_neon(tmp));

    // Reduce w.r.t. low half of d2
    tmp = gf2to256_clmul_neon(vgetq_lane_u64(d2, 0), modulus);
    d0 = veorq_u64(d0, tmp);

    vst1q_u64(&r[0], d0);
    vst1q_u64(&r[2], d1);
}

/// Multiply two field elements using ARM NEON instructions (PMULL)
/// Uses Karatsuba-like approach for multiplying 4-limb numbers
/// \param r[out] Result of multiplication (may alias a or b)
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256_mul_neon(gf2to256 r,
                       const gf2to256 a,
                       const gf2to256 b) {
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

    const uint64x2_t c0 = gf2to256_clmul_neon(a0, b0);
    const uint64x2_t c6 = gf2to256_clmul_neon(a3, b3);
    const uint64x2_t t  = gf2to256_clmul_neon(a1, b1);
    const uint64x2_t u  = gf2to256_clmul_neon(a2, b2);

    // c1 = (a0 + a1) * (b0 + b1) - c0 - t
    const uint64x2_t c1 = veorq_u64(veorq_u64(gf2to256_clmul_neon(a0 ^ a1, b0 ^ b1), c0), t);
    // c2 = (a0 + a2) * (b0 + b2) - c0 + t - u
    const uint64x2_t c2 = veorq_u64(veorq_u64(veorq_u64(gf2to256_clmul_neon(a0 ^ a2, b0 ^ b2), c0), t), u);
    // c5 = (a2 + a3) * (b2 + b3) - c6 - u
    const uint64x2_t c5 = veorq_u64(veorq_u64(gf2to256_clmul_neon(a2 ^ a3, b2 ^ b3), c6), u);
    // c4 = (a1 + a3) * (b1 + b3) - c6 + u - t
    const uint64x2_t c4 = veorq_u64(veorq_u64(veorq_u64(gf2to256_clmul_neon(a1 ^ a3, b1 ^ b3), c6), u), t);
    // c3 = (a0 + a1 + a2 + a3) * (b0 + b1 + b2 + b3) - c0 - c1 - c2 - c4 - c5 - c6
    uint64x2_t c3 = gf2to256_clmul_neon(a0 ^ a1 ^ a2 ^ a3, b0 ^ b1 ^ b2 ^ b3);
    c3 = veorq_u64(veorq_u64(veorq_u64(c3, c0), veorq_u64(c1, c2)),
                   veorq_u64(veorq_u64(c4, c5), c6));

    const uint64x2_t d0 = veorq_u64(c0, gf2to256_slli8_neon(c1));
    const uint64x2_t d1 = veorq_u64(veorq_u64(c2, gf2to256_srli8_neon(c1)), gf2to256_slli8_neon(c3));
    const uint64x2_t d2 = veorq_u64(veorq_u64(c4, gf2to256_srli8_neon(c3)), gf2to256_slli8_neon(c5));
    const uint64x2_t d3 = veorq_u64(c6, gf2to256_srli8_neon(c5));
    gf2to256_reduce_neon(r, d0, d1, d2, d3);
}

/// Alternative version using the naive (16 PMULL) multiplication approach
/// \param r[out] Result of multiplication (may alias a or b)
/// \param a[in] First operand
/// \param b[in] Second operand
static inline
void gf2to256v_mul_neon_v2(gf2to256 r,
                           const gf2to256 a,
                           const gf2to256 b) {
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];

    const uint64x2_t c0 = gf2to256_clmul_neon(a0, b0);
    const uint64x2_t c1 = veorq_u64(gf2to256_clmul_neon(a0, b1), gf2to256_clmul_neon(a1, b0));
    const uint64x2_t c2 = veorq_u64(veorq_u64(gf2to256_clmul_neon(a0, b2), gf2to256_clmul_neon(a1, b1)),
                                    gf2to256_clmul_neon(a2, b0));
    const uint64x2_t c3 = veorq_u64(veorq_u64(gf2to256_clmul_neon(a0, b3), gf2to256_clmul_neon(a1, b2)),
                                    veorq_u64(gf2to256_clmul_neon(a2, b1), gf2to256_clmul_neon(a3, b0)));
    const uint64x2_t c4 = veorq_u64(veorq_u64(gf2to256_clmul_neon(a1, b3), gf2to256_clmul_neon(a2, b2)),
                                    gf2to256_clmul_neon(a3, b1));
    const uint64x2_t c5 = veorq_u64(gf2to256_clmul_neon(a2, b3), gf2to256_clmul_neon(a3, b2));
    const uint64x2_t c6 = gf2to256_clmul_neon(a3, b3);

    const uint64x2_t d0 = veorq_u64(c0, gf2to256_slli8_neon(c1));
    const uint64x2_t d1 = veorq_u64(veorq_u64(c2, gf2to256_srli8_neon(c1)), gf2to256_slli8_neon(c3));
    const uint64x2_t d2 = veorq_u64(veorq_u64(c4, gf2to256_srli8_neon(c3)), gf2to256_slli8_neon(c5));
    const uint64x2_t d3 = veorq_u64(c6, gf2to256_srli8_neon(c5));
    gf2to256_reduce_neon(r, d0, d1, d2, d3);
}
#endif // __ARM_FEATURE_CRYPTO

/// \return a*b, uses PMULL if available, otherwise the constant-time
///         scalar multiplication
static inline
uint64x2x2_t gf2to256v_mul_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    gf2to256 aa, bb, rr;
    gf2to256v_store_u128(aa, a);
    gf2to256v_store_u128(bb, b);
#ifdef __ARM_FEATURE_CRYPTO
    gf2to256_mul_neon(rr, aa, bb);
#else
    gf2to256_mul(rr, aa, bb);
#endif
    return gf2to256v_load_u128(rr);
}
#endif // USE_NEON
