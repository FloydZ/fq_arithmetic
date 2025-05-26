#pragma once

#include <stdint.h>
#include <stdlib.h>

#include "../2/vector.h"

#if USE_AVX2
#include <immintrin.h>
#endif

// main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
// GF(2)/(x^256 + x^10 + x^5 + x^2 + 1).
#define MODULUS 0b10000100101


typedef uint64_t gf2to256[4];

static inline
void gf2to256_set_zero(gf2to256 r) {
    for (uint32_t i = 0; i < 4; i++) {
        r[i] = 0;
    }
}

/// \param r[out]: = a + b
/// \param a[in]:
/// \param b[int]:
static inline
void gf2to256_add(gf2to256 r, 
                  const gf2to256 a,
                  const gf2to256 b) {
    r[0] = a[0] ^ b[0];
    r[1] = a[1] ^ b[1];
    r[2] = a[2] ^ b[2];
    r[3] = a[3] ^ b[3];
}

/// \param r[out]: = a - b
/// \param a[in]:
/// \param b[int]:
static inline
void gf2to256_sub(gf2to256 r,
                  const gf2to256 a,
                  const gf2to256 b) {
    gf2to256_add(r, a, b);
}

/// \param r[out]: = a * b
/// \param a[in]:
/// \param b[int]:
static inline
void gf2to256_mul(gf2to256 r,
                  const gf2to256 a,
                  const gf2to256 b) {
    /// Slow, but straight-forward
    uint64_t shifted[4] = {a[0], a[1], a[2], a[3]};
    for (size_t i = 0; i < 4; ++i){
        for (size_t j = 0; j < 64; ++j) {
            if (b[i] & (1ull << j)) {
                r[0] ^= shifted[0];
                r[1] ^= shifted[1];
                r[2] ^= shifted[2];
                r[3] ^= shifted[3];
            }

            uint32_t reduce = (shifted[3] & (1ull << 63));

            shifted[3] = (shifted[3] << 1) | (shifted[2] >> 63);
            shifted[2] = (shifted[2] << 1) | (shifted[1] >> 63);
            shifted[1] = (shifted[1] << 1) | (shifted[0] >> 63);
            shifted[0] = shifted[0] << 1;

            if (reduce) {
                shifted[0] ^= MODULUS;
            }
        }
    }
}

/// \param r[out]: = a * b
/// \param a[in]: over gf2to256
/// \param b[int]: over gf2
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


/// \param out[out]: must be 8 registers
/// \param in[in]:
/// \return nothing
static inline void gf2to256v_expand_gf2_x8_u256(__m256i *out,
                                                const uint8_t in) {
    for (uint32_t i = 0; i < 8; i++) {
        out[i] = _mm256_set1_epi64x(-((in >> i) & 1u));
    }
}

/// \param a[in]:
/// \param b[in]:
/// \return a + b
static inline
__m256i gf2to256v_add_u256(const __m256i a,
                           const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// \param a[in]:
/// \param b[in]:
/// \return a - b
static inline
__m256i gf2to256v_sub_u256(const __m256i a,
                           const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/// \param a[in]:
/// \param b[in]:
/// \return a * b
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
    const __m128i modulus = _mm_set_epi64x(MODULUS, 0);
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

/// \param a[in]:
/// \param b[in]:
/// \return a * b
static inline
__m256i gf2to256v_mul_u256_v2(const __m256i a,
                              const __m256i b) {
    /* here we compute the same c as in Karatsuba, but by just naively
       multiplying all pairs of limbs of the operands and adding together
       the results that correspond to the same shift. */
    const __m128i a_low = _mm_loadu_si128((const __m128i*) &(a[0]));
    const __m128i a_high = _mm_loadu_si128((const __m128i*) &(a[2]));
    const __m128i b_low = _mm_loadu_si128((const __m128i*) &(b[0]));
    const __m128i b_high = _mm_loadu_si128((const __m128i*) &(b[2]));
    const __m128i modulus = _mm_set_epi64x(MODULUS, MODULUS);

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

/// \param a \in gf2to256
/// \param b \in gf2 already expanded
static inline
__m256i gf2to256v_mul_gf2_u256(const __m256i a,
                               const __m256i b) {
    return a & b;
}

#endif
