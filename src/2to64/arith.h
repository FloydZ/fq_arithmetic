#include <stdint.h>

/// main source: https://github.com/scipr-lab/libff/tree/develop/libff/algebra/fields/binary
// GF(2)/[x^64 + x^4 + x^3 + x + 1]
#define MODULUS 0b11011ul
typedef uint64_t gf2to64;

///
/// \param a
/// \param b
/// \return
static inline
gf2to64 gf2to64_add(const gf2to64 a,
                 const gf2to64 b) {
    return a ^ b;
}

///
/// \param a
/// \param b
/// \return
static inline
gf2to64 gf2to64_sub(const gf2to64 a,
                 const gf2to64 b) {
    return a ^ b;
}

///
/// \param a
/// \param b
/// \return
static inline
gf2to64 gf2to64_mul(const gf2to64 a,
                 const gf2to64 b) {
    uint32_t result = 0;
    uint32_t shifted = a;

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

///
/// \param a
/// \return
static inline
gf2to64 gf2to64_neg(const gf2to64 a) {
    return ~a;
}

#ifdef USE_AVX2
#include <immintrin.h>

///
/// \param a
/// \param b
/// \return
static inline
__m256i gf2to64v_add_u256(const __m256i a,
                       const __m256i b) {
    return a; // TODO
}

///
/// \param a
/// \param b
/// \return
static inline
__m256i gf2to64v_sub_u256(const __m256i a,
                       const __m256i b) {
    return a; // TODO
}

///
/// \param a
/// \param b
/// \return
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

// TODO
// __m256i gf2to64v_mul_u256_(const __m256i a,
//                        const __m256i b) {
//     const __m128i modulus = _mm_setr_epi64((__m64)MODULUS, (__m64)0ll);
//     const __m128i mul128 = _mm_clmulepi64_si128(_mm_setr_epi64((__m64)MODULUS, (__m64)0ll),
//                                                 _mm_setr_epi64((__m64)MODULUS, (__m64)0ll), 0);
//
//     /* reduce the 64 higher order bits of mul128. Output is 96 bits since modulus < 2^64 */
//     const __m128i mul96 = _mm_clmulepi64_si128(modulus, mul128, 0x10); /* use high half of mul128 */
//     __m128i rem = _mm_xor_si128(mul128, mul96);
//
//     /* reduce the 32 higher order bits of mul96 */
//     const __m128i mul64 = _mm_clmulepi64_si128(modulus, mul96, 0x10); /* use high half of mul96 */
//
//     rem = _mm_xor_si128(rem, mul64);
//     return (uint64_t)_mm_movepi64_pi64(rem);
// }

#endif

