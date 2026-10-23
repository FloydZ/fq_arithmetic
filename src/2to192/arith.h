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
        r[i] = rand() ^ ((uint64_t)rand() << 32u) ^ ((uint64_t)rand() << 62u);
    }
}

/// \param r[out]: = a
/// \param a[in]: element to copy
static inline
void gf2to192_copy(gf2to192 r,
                   const gf2to192 a) {
    for (uint32_t i = 0; i < 3; i++) {
        r[i] = a[i];
    }
}

/// \return 1 if a == b, 0 otherwise
static inline
int gf2to192_is_equal(const gf2to192 a,
                      const gf2to192 b) {
    return (a[0] == b[0]) & (a[1] == b[1]) & (a[2] == b[2]);
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

/// Negation in GF(2^192)
/// \param r[out]: -a, which is a in characteristic 2
/// \param a[in]: element to negate
static inline
void gf2to192_neg(gf2to192 r,
                  const gf2to192 a) {
    gf2to192_copy(r, a);
}

/// Multiplication in GF(2^192) - non-constant time version
/// \param r[out]: result of multiplication (may alias a or b)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \note This implementation is NOT constant-time (vulnerable to timing attacks)
/// \note Uses the shift-and-add algorithm with conditional reduction by the modulus
static inline
void gf2to192_mul_v2(gf2to192 r,
                     const gf2to192 a,
                     const gf2to192 b) {
    uint64_t shifted[3] = {a[0], a[1], a[2]};
    const uint64_t bb[3] = {b[0], b[1], b[2]};
    uint64_t res[3] = {0};
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            if (bb[i] & (1ull << j)) {
                res[0] ^= shifted[0];
                res[1] ^= shifted[1];
                res[2] ^= shifted[2];
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

    gf2to192_copy(r, res);
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
/// \param r[out]: result of multiplication (may alias a or b)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \note This implementation is constant-time (resistant to timing attacks)
/// \note Implements multiplication using bit-masking to avoid branches
static inline
void gf2to192_mul(gf2to192 r,
                  const gf2to192 a,
                  const gf2to192 b) {
    gf2to192 t1, aa, bb, res;
    gf2to192_copy(aa, a);
    gf2to192_copy(bb, b);
    gf2to192_mul_helper(res, aa, bb[2]>>63);
    for (int i = 190; i >= 0; --i) {
        const uint32_t limb = i / 64;
        const uint32_t pos = i % 64;

        gf2to192_mul_helper(t1, aa, (bb[limb] >> pos) & 1ull);
        t1[0] ^= (-(res[2] >> 63)) & MODULUS;
        gf2to192_mul_addition(t1, res, res);
        gf2to192_copy(res, t1);
    }

    gf2to192_copy(r, res);
}

/// Inversion in GF(2^192): a^(2^192 - 2), constant time
/// \param r[out]: a^{-1}, or 0 if a = 0 (may alias a)
/// \param a[in]: element to invert
static inline
void gf2to192_inv(gf2to192 r,
                  const gf2to192 a) {
    gf2to192 y, aa;
    gf2to192_copy(aa, a);
    gf2to192_copy(y, aa);
    // y = a^(2^191 - 1)
    for (uint32_t i = 0; i < 190; i++) {
        gf2to192_mul(y, y, y);
        gf2to192_mul(y, y, aa);
    }
    // r = a^(2^192 - 2)
    gf2to192_mul(r, y, y);
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
    const uint64_t t = -(uint64_t)b;
    for (uint32_t i = 0; i < 3; i++) {
        r[i] = a[i] & t;
    }
}

#ifdef USE_AVX2
/// AVX2 vectorized implementation for GF(2^192) operations
/// Main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
///
/// Lane layout of a `_u256` value: ONE element in the limbs 0,1,2; limb 3 is
/// zero. Use `gf2to192v_load_u256`/`gf2to192v_store_u256` to move a single
/// element (24 bytes) between memory and a register without touching the
/// following 8 bytes.

/// \param a[in]: element in memory
/// \return a in limbs 0..2, limb 3 = 0
static inline
__m256i gf2to192v_load_u256(const gf2to192 a) {
    const __m256i mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    return _mm256_maskload_epi64((const long long *)a, mask);
}

/// \param r[out]: = limbs 0..2 of a
/// \param a[in]: register
static inline
void gf2to192v_store_u256(gf2to192 r,
                          const __m256i a) {
    const __m256i mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    _mm256_maskstore_epi64((long long *)r, mask, a);
}

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
    return _mm256_xor_si256(a, b);
}

/// Vectorized subtraction of GF(2^192) elements
/// \param a[in]: 256-bit vector of GF(2^192) elements
/// \param b[in]: 256-bit vector of GF(2^192) elements
/// \return Vector of element-wise differences
/// \note Identical to addition (XOR) since we're in characteristic 2
static inline
__m256i gf2to192v_sub_u256(const __m256i a,
                           const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// Vectorized multiplication of GF(2^192) elements
/// \param a[in]: one GF(2^192) element (limbs 0..2, limb 3 ignored)
/// \param b[in]: one GF(2^192) element (limbs 0..2, limb 3 ignored)
/// \return a*b in limbs 0..2, limb 3 = 0
/// \note Uses Karatsuba algorithm for optimized multiplication
/// \note Implements reduction using the field's modulus polynomial
static inline
__m256i gf2to192v_mul_u256(const __m256i a,
                           const __m256i b) {
    const __m256i t1 = _mm256_unpacklo_epi64(a, b);
    const __m256i t2 = _mm256_unpackhi_epi64(a, b);
    const __m128i ab0 = _mm256_castsi256_si128(t1);
    const __m128i ab1 = _mm256_castsi256_si128(t2);
    const __m128i ab2 = _mm256_extracti128_si256(t1, 1);
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

    /* the high half of d1 is reduced, clear it (limb 3 = 0) */
    return _mm256_set_m128i(_mm_move_epi64(d1), d0);
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

    __m128i d0 = _mm_xor_si128(c0, _mm_slli_si128(c1, 8));
    __m128i d2 = _mm_xor_si128(c4, _mm_srli_si128(c3, 8));
    __m128i d1 = _mm_xor_si128(_mm_xor_si128(c2, _mm_srli_si128(c1, 8)), _mm_slli_si128(c3, 8));

    __m128i tmp = _mm_clmulepi64_si128(d2, modulus, 0x01);
    d1 = _mm_xor_si128(d1, tmp);

    tmp = _mm_clmulepi64_si128(d2, modulus, 0x00);
    d1 = _mm_xor_si128(d1, _mm_srli_si128(tmp, 8));
    d0 = _mm_xor_si128(d0, _mm_slli_si128(tmp, 8));

    tmp = _mm_clmulepi64_si128(d1, modulus, 0x01);
    d0 = _mm_xor_si128(d0, tmp);

    _mm_storeu_si128((__m128i*) &r[0], d0);
    _mm_storel_epi64((__m128i*) &r[2], d1);
}

/// Vectorized multiplication of GF(2^192) elements by expanded GF(2) elements
/// \param a[in]: 256-bit vector containing GF(2^192) elements
/// \param b[in]: 256-bit vector containing expanded GF(2) elements (full masks)
/// \return 256-bit vector with element-wise products
static inline
__m256i gf2to192v_mul_gf2_u256(const __m256i a,
                               const __m256i b) {
    return a & b;
}

#endif // USE_AVX2

#ifdef USE_AVX512
/// Lane layout of a `_u512` value: TWO elements, element e (e = 0, 1) in the
/// limbs 4e+0, 4e+1, 4e+2; the limbs 3 and 7 are zero. Two consecutive
/// elements in memory (48 bytes) are moved with `gf2to192v_load_u512` /
/// `gf2to192v_store_u512`.

/// \param a[in]: pointer to 2 elements (or only 1 if mask = 0x07)
/// \param mask[in]: 0x77 for two elements, 0x07 for one element
static inline
__m512i gf2to192v_load_mask_u512(const gf2to192 *a,
                                 const __mmask8 mask) {
    return _mm512_maskz_expandloadu_epi64(mask, (const void *)a);
}

/// \param r[out]: pointer to 2 elements (or only 1 if mask = 0x07)
/// \param a[in]: register
/// \param mask[in]: 0x77 for two elements, 0x07 for one element
static inline
void gf2to192v_store_mask_u512(gf2to192 *r,
                               const __m512i a,
                               const __mmask8 mask) {
    _mm512_mask_compressstoreu_epi64((void *)r, mask, a);
}

/// \param a[in]: pointer to 2 consecutive elements
static inline
__m512i gf2to192v_load_u512(const gf2to192 *a) {
    return gf2to192v_load_mask_u512(a, 0x77);
}

/// \param r[out]: pointer to 2 consecutive elements
static inline
void gf2to192v_store_u512(gf2to192 *r,
                          const __m512i a) {
    gf2to192v_store_mask_u512(r, a, 0x77);
}

/// \return a + b for two elements
static inline
__m512i gf2to192v_add_u512(const __m512i a,
                           const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// \return a - b for two elements
static inline
__m512i gf2to192v_sub_u512(const __m512i a,
                           const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// Multiplication of two pairs of GF(2^192) elements. Same Karatsuba
/// algorithm as `gf2to192v_mul_u256`, but each 128-bit lane of the 256-bit
/// VPCLMULQDQ registers works on a different element.
/// \return a*b, element-wise, limbs 3 and 7 are zero
static inline
__m512i gf2to192v_mul_u512(const __m512i a,
                           const __m512i b) {
    /* ab_k: lane e = [a_k(e), b_k(e)] */
    const __m512i idx01 = _mm512_setr_epi64(0, 8, 4, 12, 1, 9, 5, 13);
    const __m512i idx2  = _mm512_setr_epi64(2, 10, 6, 14, 3, 11, 7, 15);
    const __m512i q = _mm512_permutex2var_epi64(a, idx01, b);
    const __m512i s = _mm512_permutex2var_epi64(a, idx2, b);
    const __m256i ab0 = _mm512_castsi512_si256(q);
    const __m256i ab1 = _mm512_extracti64x4_epi64(q, 1);
    const __m256i ab2 = _mm512_castsi512_si256(s);
    const __m256i modulus = _mm256_set1_epi64x(MODULUS);

    const __m256i c0 = _mm256_clmulepi64_epi128(ab0, ab0, 0x01);
    const __m256i c4 = _mm256_clmulepi64_epi128(ab2, ab2, 0x01);
    const __m256i t  = _mm256_clmulepi64_epi128(ab1, ab1, 0x01);

    const __m256i xor01 = _mm256_xor_si256(ab0, ab1);
    __m256i c1 = _mm256_clmulepi64_epi128(xor01, xor01, 0x01);
    c1 = _mm256_xor_si256(_mm256_xor_si256(c1, c0), t);

    const __m256i xor12 = _mm256_xor_si256(ab1, ab2);
    __m256i c3 = _mm256_clmulepi64_epi128(xor12, xor12, 0x01);
    c3 = _mm256_xor_si256(_mm256_xor_si256(c3, c4), t);

    const __m256i xor02 = _mm256_xor_si256(ab0, ab2);
    __m256i c2 = _mm256_clmulepi64_epi128(xor02, xor02, 0x01);
    c2 = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(c2, c0), c4), t);

    __m256i d0 = _mm256_xor_si256(c0, _mm256_bslli_epi128(c1, 8));
    __m256i d2 = _mm256_xor_si256(c4, _mm256_bsrli_epi128(c3, 8));
    __m256i d1 = _mm256_xor_si256(_mm256_xor_si256(c2, _mm256_bsrli_epi128(c1, 8)),
                                  _mm256_bslli_epi128(c3, 8));

    __m256i tmp = _mm256_clmulepi64_epi128(d2, modulus, 0x01);
    d1 = _mm256_xor_si256(d1, tmp);

    tmp = _mm256_clmulepi64_epi128(d2, modulus, 0x00);
    d1 = _mm256_xor_si256(d1, _mm256_bsrli_epi128(tmp, 8));
    d0 = _mm256_xor_si256(d0, _mm256_bslli_epi128(tmp, 8));

    tmp = _mm256_clmulepi64_epi128(d1, modulus, 0x01);
    d0 = _mm256_xor_si256(d0, tmp);

    /* element e = [d0(e).lo, d0(e).hi, d1(e).lo, 0] */
    const __m512i d = _mm512_inserti64x4(_mm512_castsi256_si512(d0), d1, 1);
    const __m512i idx = _mm512_setr_epi64(0, 1, 4, 0, 2, 3, 6, 0);
    return _mm512_maskz_permutexvar_epi64(0x77, idx, d);
}
#endif // USE_AVX512

#ifdef USE_NEON
/// Lane layout of a `_u128` value: a GF(2^192) element does not fit into a
/// single NEON register, hence `uint64x2x2_t` is used: val[0] = limbs 0,1,
/// val[1] = limb 2, 0.

/// \param a[in]: element in memory
static inline
uint64x2x2_t gf2to192v_load_u128(const gf2to192 a) {
    uint64x2x2_t r;
    r.val[0] = vld1q_u64(a);
    r.val[1] = vcombine_u64(vld1_u64(a + 2), vdup_n_u64(0));
    return r;
}

/// \param r[out]: = a
/// \param a[in]: register pair
static inline
void gf2to192v_store_u128(gf2to192 r,
                          const uint64x2x2_t a) {
    vst1q_u64(r, a.val[0]);
    vst1_u64(r + 2, vget_low_u64(a.val[1]));
}

/// \return a + b
static inline
uint64x2x2_t gf2to192v_add_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    uint64x2x2_t r;
    r.val[0] = veorq_u64(a.val[0], b.val[0]);
    r.val[1] = veorq_u64(a.val[1], b.val[1]);
    return r;
}

/// \return a - b
static inline
uint64x2x2_t gf2to192v_sub_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    return gf2to192v_add_u128(a, b);
}

#ifdef __ARM_FEATURE_CRYPTO
/// \return carry-less product of a and b as (lo, hi)
static inline
uint64x2_t gf2to192_clmul_neon(const uint64_t a,
                               const uint64_t b) {
    return vreinterpretq_u64_p128(vmull_p64((poly64_t)a, (poly64_t)b));
}

/// Multiplication in GF(2^192) using PMULL (Karatsuba, 6 multiplications)
/// \param r[out]: a*b (may alias a or b)
/// \param a[in]: first factor
/// \param b[in]: second factor
static inline
void gf2to192_mul_neon(gf2to192 r,
                       const gf2to192 a,
                       const gf2to192 b) {
    const uint64_t modulus = MODULUS;
    const uint64x2_t c0 = gf2to192_clmul_neon(a[0], b[0]);
    const uint64x2_t c4 = gf2to192_clmul_neon(a[2], b[2]);
    const uint64x2_t t  = gf2to192_clmul_neon(a[1], b[1]);
    const uint64x2_t c1 = veorq_u64(veorq_u64(gf2to192_clmul_neon(a[0] ^ a[1], b[0] ^ b[1]), c0), t);
    const uint64x2_t c3 = veorq_u64(veorq_u64(gf2to192_clmul_neon(a[1] ^ a[2], b[1] ^ b[2]), c4), t);
    const uint64x2_t c2 = veorq_u64(veorq_u64(veorq_u64(gf2to192_clmul_neon(a[0] ^ a[2], b[0] ^ b[2]), c0), c4), t);

    // product = sum_k c_k B^k, B = 2^64
    uint64_t d0_lo = vgetq_lane_u64(c0, 0);
    uint64_t d0_hi = vgetq_lane_u64(c0, 1) ^ vgetq_lane_u64(c1, 0);
    uint64_t d1_lo = vgetq_lane_u64(c1, 1) ^ vgetq_lane_u64(c2, 0);
    uint64_t d1_hi = vgetq_lane_u64(c2, 1) ^ vgetq_lane_u64(c3, 0);
    const uint64_t d2_lo = vgetq_lane_u64(c3, 1) ^ vgetq_lane_u64(c4, 0);
    const uint64_t d2_hi = vgetq_lane_u64(c4, 1);

    // reduce w.r.t. high half of d2: x^320 = x^128 * x^192
    uint64x2_t tmp = gf2to192_clmul_neon(d2_hi, modulus);
    d1_lo ^= vgetq_lane_u64(tmp, 0);
    d1_hi ^= vgetq_lane_u64(tmp, 1);

    // reduce w.r.t. low half of d2: x^256 = x^64 * x^192
    tmp = gf2to192_clmul_neon(d2_lo, modulus);
    d0_hi ^= vgetq_lane_u64(tmp, 0);
    d1_lo ^= vgetq_lane_u64(tmp, 1);

    // reduce w.r.t. high half of d1: x^192
    tmp = gf2to192_clmul_neon(d1_hi, modulus);
    d0_lo ^= vgetq_lane_u64(tmp, 0);
    d0_hi ^= vgetq_lane_u64(tmp, 1);

    r[0] = d0_lo;
    r[1] = d0_hi;
    r[2] = d1_lo;
}
#endif // __ARM_FEATURE_CRYPTO

/// \return a*b, uses PMULL if available, otherwise the constant-time
///         scalar multiplication
static inline
uint64x2x2_t gf2to192v_mul_u128(const uint64x2x2_t a,
                                const uint64x2x2_t b) {
    gf2to192 aa, bb, rr;
    gf2to192v_store_u128(aa, a);
    gf2to192v_store_u128(bb, b);
#ifdef __ARM_FEATURE_CRYPTO
    gf2to192_mul_neon(rr, aa, bb);
#else
    gf2to192_mul(rr, aa, bb);
#endif
    return gf2to192v_load_u128(rr);
}

/// Vectorized multiplication of multiple GF(2^192) elements using NEON
/// \param a[in]: Array of GF(2^192) elements
/// \param b[in]: Array of GF(2^192) elements
/// \param r[out]: Array to store results of multiplications
/// \param count[in]: Number of elements to process
static inline
void gf2to192v_mul_batch_neon(const gf2to192 *a,
                              const gf2to192 *b,
                              gf2to192 *r,
                              const size_t count) {
    for (size_t i = 0; i < count; i++) {
        gf2to192v_store_u128(r[i], gf2to192v_mul_u128(gf2to192v_load_u128(a[i]),
                                                       gf2to192v_load_u128(b[i])));
    }
}
#endif // USE_NEON
