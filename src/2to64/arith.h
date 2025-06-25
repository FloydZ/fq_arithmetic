#include <stdint.h>

/// main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
// GF(2)/[x^64 + x^4 + x^3 + x + 1]
#define MODULUS 0b11011ul
typedef uint64_t gf2to64;

/// \param a[in]:
/// \param b[in]:
/// \return a + b
static inline
gf2to64 gf2to64_add(const gf2to64 a,
                 const gf2to64 b) {
    return a ^ b;
}

/// \param a[in]:
/// \param b[in]:
/// \return a - b
static inline
gf2to64 gf2to64_sub(const gf2to64 a,
                 const gf2to64 b) {
    return a ^ b;
}

/// NOTE: non ct
/// \param a[in]:
/// \param b[in]:
/// \return a*b \mod 2**64
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

/// NOTE: this is ct
/// \param a
/// \param b
/// \return a*b
static inline
gf2to64 gf2to64_mul(const gf2to64 a,
                    const gf2to64 b) {
    gf2to64 r = (-(b>>63u) & a);
    for (int i = 62; i >= 0; --i) {
        r = (-(b>>i & 1u) & a) ^ (-(r>>63) & MODULUS) ^ (r+r);
    }

    return r;
}

/// \param a
/// \return -a mod 2**32
static inline
gf2to64 gf2to64_neg(const gf2to64 a) {
    return ~a;
}

#ifdef USE_AVX2
#include <immintrin.h>

/// \param a[in]: [a_0, ..., a_3]
/// \param b[in]: [b_0, ..., b_3]
/// \return a + b: [a_0 + b_0, ..., a_3 + b_3]
static inline
__m256i gf2to64v_add_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// \param a[in]: [a_0, ..., a_3]
/// \param b[in]: [b_0, ..., b_3]
/// \return a - b: [a_0 - b_0, ..., a_3 - b_3]
static inline
__m256i gf2to64v_sub_u256(const __m256i a,
                          const __m256i b) {
    return a ^ b;
}

/// \param a[in]: [a_0, ..., a_3]
/// \param b[in]: [b_0, ..., b_3]
/// \return a * b: [a_0 * b_0, ..., a_3 * b_3]
static inline
gf2to64 gf2to64v_mul_u256(const gf2to64 a,
                          const gf2to64 b) {
    const __m128i modulus = _mm_setr_epi64((__m64)MODULUS, (__m64)0ll);
    const __m128i mul128 = _mm_clmulepi64_si128(_mm_setr_epi64((__m64)a, (__m64)0ll),
                                                _mm_setr_epi64((__m64)b, (__m64)0ll), 0);

    /* reduce the 64 higher order bits of mul128. Output is 96 bits since modulus < 2^64 */
    const __m128i mul96 = _mm_clmulepi64_si128(modulus, mul128, 0x10); /* use high half of mul128 */
    __m128i rem = _mm_xor_si128(mul128, mul96);

    /* reduce the 32 higher order bits of mul96 */
    const __m128i mul64 = _mm_clmulepi64_si128(modulus, mul96, 0x10); /* use high half of mul96 */

    rem = _mm_xor_si128(rem, mul64);
    return (uint64_t)_mm_movepi64_pi64(rem);
}

static inline
__m256i gf2to64v_mul_u256_(const __m256i a,
                           const __m256i b) {
    const __m128i modulus = _mm_setr_epi64((__m64)MODULUS, (__m64)0ll);

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

#endif

