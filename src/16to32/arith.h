#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../helper.h"
#include "../16/arith.h"

#ifdef USE_AVX2
#include <immintrin.h>
#endif

#ifdef USE_NEON
#include <arm_neon.h>
#endif

// GF(16^32) element: 32 GF(16) coefficients packed as nibbles. Coefficient i
// is stored in byte i/2, in the low nibble if i is even, high nibble otherwise.
// The field is defined by the irreducible polynomial
//     x^32 + B*x^28 + x^21 + F*x^2 + 9
typedef uint8_t gf16to32[16];


/// \param r[out]: = d (as the low 64 bits)
static inline
void gf16to32_set_u(gf16to32 r, const uint64_t d) {
    memset(r, 0, 16);
    memcpy(r, &d, 8);
}

/// \param r[out]: = hi*2^64 + lo
static inline
void gf16to32_set_uu(gf16to32 r, const uint64_t hi, const uint64_t lo) {
    memcpy(r, &lo, 8);
    memcpy(r + 8, &hi, 8);
}

/// \param v[out]: = 0
static inline
void gf16to32_set_zero(gf16to32 v) {
    memset(v, 0, 16);
}

/// \param v[out]: = random element
static inline
void gf16to32_set_random(gf16to32 v) {
    for (uint32_t i = 0; i < 16; i++) {
        v[i] = (uint8_t)rand();
    }
}

/// \param r[out]: = a
static inline
void gf16to32_copy(gf16to32 r, const gf16to32 a) {
    memmove(r, a, 16);
}

/// \return 1 if a == b, 0 otherwise
static inline
int gf16to32_is_equal(const gf16to32 a, const gf16to32 b) {
    return memcmp(a, b, 16) == 0;
}

/// lexicographic comparison of the byte representation (most significant byte first)
/// \return 1 if a > b, -1 if a < b, 0 if a == b
static inline
int gf16to32_cmp(const gf16to32 a, const gf16to32 b) {
    for (int64_t i = 15; i >= 0; i--) {
        if (a[i] > b[i]) {
            return 1;
        }

        if (a[i] < b[i]) {
            return -1;
        }
    }

    return 0;
}

/// \param r[out]: = a + b
static inline
void gf16to32_add(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    for (uint32_t i = 0; i < 16; i++) {
        r[i] = a[i] ^ b[i];
    }
}

/// \param r[out]: = a - b
static inline
void gf16to32_sub(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    gf16to32_add(r, a, b);
}

/// \param r[out]: = -a, which is a in characteristic 2
static inline
void gf16to32_neg(gf16to32 r, const gf16to32 a) {
    gf16to32_copy(r, a);
}

/// \param c[out]: = a*b
/// \param a[in]: element of GF(16^32)
/// \param b[in]: element of GF(16)
static inline
void gf16to32_mul_gf16(gf16to32 c, const gf16to32 a, const gf16 b) {
    const gf16 bb = b & 0xF;
    for (uint64_t i = 0; i < 16; i++) {
        c[i] = gf16_mul(a[i] & 0xF, bb) | (gf16_mul(a[i] >> 4, bb) << 4);
    }
}

/// NOTE: non ct
/// \param r[out]: = a*b (may alias a or b)
static inline
void gf16to32_mul(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    uint8_t t[64] = {0};

    static const uint8_t MOD[32] = {
        0x9, 0x0, 0xF, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0xB, 0x0, 0x0, 0x0
    };

    for (uint32_t i = 0; i < 32; i++) {
        uint8_t ai = (a[i >> 1] >> ((i & 1) * 4)) & 0xF;

        for (uint32_t j = 0; j < 32; j++) {
            uint8_t bj = (b[j >> 1] >> ((j & 1) * 4)) & 0xF;
            t[i + j] ^= gf16_mul(ai, bj);
        }
    }
    for (uint32_t k = 62; k >= 32; k--) {
        const uint8_t c = t[k];
        if (!c) { continue; }

        t[k] = 0;

        for (int i = 0; i < 32; i++) {
            if (MOD[i]) {
                t[k - 32 + i] ^= gf16_mul(c, MOD[i]);
            }
        }
    }

    memset(r, 0, 16);
    for (int i = 0; i < 32; i++) {
        r[i >> 1] |= (t[i] & 0xF) << ((i & 1) * 4);
    }
}

/// NOTE: needs the gnu (gcc) extension of __uint128_t
/// NOTE: this is a bit-level carry-less multiplication modulo a degree-128
///       binary polynomial. It does NOT compute the multiplication of this
///       field (GF(16)-coefficients, see `gf16to32_mul`). Only kept for
///       benchmarking purposes.
static inline
void gf16to32_mul_v2(gf16to32 c, const gf16to32 a, const gf16to32 b) {
    __uint128_t aa, bb;
    memcpy(&aa, a, 16);
    memcpy(&bb, b, 16);
    const __int128_t mask  = 0x1ll;
    __uint128_t r = 0;
    const __uint128_t mod =  ((__uint128_t)0x90F0000000000000ull << 64) ^ 0x000001000000B000ull;
    for (uint64_t i = 0; i < 128; i++) {
        const __uint128_t t0 = r+r;
        const __uint128_t t1 = (((__int128_t)r) >> 127) & mod;

        const __int128_t t2 = (__int128_t)bb >> (127 - i);
        const __int128_t t3 = t2 & mask;
        const __int128_t t4 = -t3;
        const __uint128_t t5 = (__uint128_t) t4 & aa;
        r = t0 ^ t1 ^ t5;
    }

    memcpy(c, &r, 16);
}

/// a^{-1} = a^(16^32 - 2)
/// \param r[out]: = b^{-1}, or 0 if b = 0
static inline
void gf16to32_inv(gf16to32 r, const gf16to32 b) {
    gf16to32 y, bb;
    memcpy(bb, b, 16);
    memcpy(y, bb, 16);
    for (uint64_t i = 0; i < 32 * 4 - 2; i++) {
        gf16to32_mul(y, y, y);
        gf16to32_mul(y, y, bb);
    }

    gf16to32_mul(r, y, y);
}

/// The SIMD multiplications below compute a*b with a Horner scheme over the
/// coefficients of b, each 128-bit lane holding one element:
///     r = 0; for j = 31..0: r = r*X mod f; r += b_j * a
/// where r*X is a 4-bit shift of the lane and the dropped coefficient c of
/// X^32 is reduced by adding c * (B*X^28 + X^21 + F*X^2 + 9). Multiplications
/// by GF(16) scalars are done bitsliced with the precomputed a*x^k, k = 0..3.

#ifdef USE_AVX2
/// Lane layout of a `_u256` value: TWO elements (bytes 0..15 and 16..31),
/// i.e. the memory layout of `gf16to32[2]`.
/// Inside USE_AVX2, the `_u128` functions are SSE helpers working on ONE element.

/// \return nibble-wise a*x in GF(16) = GF(2)[x]/(x^4 + x + 1)
static inline
__m128i gf16to32v_gf16_mulx_u128(const __m128i a) {
    const __m128i m7 = _mm_set1_epi8(0x77);
    const __m128i m1 = _mm_set1_epi8(0x11);
    const __m128i h = _mm_and_si128(_mm_srli_epi64(a, 3), m1);
    const __m128i s = _mm_slli_epi64(_mm_and_si128(a, m7), 1);
    return _mm_xor_si128(_mm_xor_si128(s, h), _mm_slli_epi64(h, 1));
}

/// \return nibble-wise a*b in GF(16)
static inline
__m128i gf16to32v_gf16_mul_u128(__m128i a,
                                const __m128i b) {
    const __m128i m1 = _mm_set1_epi8(0x11);
    __m128i r = _mm_setzero_si128();
    for (uint32_t k = 0; k < 4; k++) {
        __m128i t = _mm_and_si128(_mm_srli_epi64(b, k), m1);
        t = _mm_xor_si128(t, _mm_slli_epi64(t, 1));
        t = _mm_xor_si128(t, _mm_slli_epi64(t, 2));
        r = _mm_xor_si128(r, _mm_and_si128(a, t));
        a = gf16to32v_gf16_mulx_u128(a);
    }
    return r;
}

/// \return nibble-wise a*x in GF(16)
static inline
__m256i gf16to32v_gf16_mulx_u256(const __m256i a) {
    const __m256i m7 = _mm256_set1_epi8(0x77);
    const __m256i m1 = _mm256_set1_epi8(0x11);
    const __m256i h = _mm256_and_si256(_mm256_srli_epi64(a, 3), m1);
    const __m256i s = _mm256_slli_epi64(_mm256_and_si256(a, m7), 1);
    return _mm256_xor_si256(_mm256_xor_si256(s, h), _mm256_slli_epi64(h, 1));
}

/// \return nibble-wise a*b in GF(16)
static inline
__m256i gf16to32v_gf16_mul_u256(__m256i a,
                                const __m256i b) {
    const __m256i m1 = _mm256_set1_epi8(0x11);
    __m256i r = _mm256_setzero_si256();
    for (uint32_t k = 0; k < 4; k++) {
        __m256i t = _mm256_and_si256(_mm256_srli_epi64(b, k), m1);
        t = _mm256_xor_si256(t, _mm256_slli_epi64(t, 1));
        t = _mm256_xor_si256(t, _mm256_slli_epi64(t, 2));
        r = _mm256_xor_si256(r, _mm256_and_si256(a, t));
        a = gf16to32v_gf16_mulx_u256(a);
    }
    return r;
}

/// SSE: a + b for one element
static inline
__m128i gf16to32v_add_u128(const __m128i a128, const __m128i b128) {
    return _mm_xor_si128(a128, b128);
}

/// SSE: a - b for one element
static inline
__m128i gf16to32v_sub_u128(const __m128i a128, const __m128i b128) {
    return _mm_xor_si128(a128, b128);
}

/// \return a + b for two elements
static inline
__m256i gf16to32v_add_u256(const __m256i a256, const __m256i b256) {
    return _mm256_xor_si256(a256, b256);
}

/// \return a - b for two elements
static inline
__m256i gf16to32v_sub_u256(const __m256i a256, const __m256i b256) {
    return _mm256_xor_si256(a256, b256);
}

/// SSE: a*b for one element
static inline
__m128i gf16to32v_mul_u128(const __m128i a, const __m128i b) {
    const __m128i mod = _mm_setr_epi8(0x09, 0x0F, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0x10, 0, 0, 0, 0x0B, 0);
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i top = _mm_set1_epi8(15);
    __m128i A[4], M[4], bit[4];
    A[0] = a; M[0] = mod; bit[0] = _mm_set1_epi8(1);
    for (uint32_t k = 1; k < 4; k++) {
        A[k] = gf16to32v_gf16_mulx_u128(A[k-1]);
        M[k] = gf16to32v_gf16_mulx_u128(M[k-1]);
        bit[k] = _mm_set1_epi8((char)(1u << k));
    }

    __m128i r = _mm_setzero_si128();
    for (int j = 31; j >= 0; j--) {
        // r = r*X mod f
        const __m128i c = _mm_and_si128(_mm_srli_epi16(_mm_shuffle_epi8(r, top), 4), m0f);
        r = _mm_or_si128(_mm_slli_epi64(r, 4), _mm_srli_epi64(_mm_bslli_si128(r, 8), 60));
        for (uint32_t k = 0; k < 4; k++) {
            const __m128i m = _mm_cmpeq_epi8(_mm_and_si128(c, bit[k]), bit[k]);
            r = _mm_xor_si128(r, _mm_and_si128(M[k], m));
        }

        // r += b_j * a
        __m128i bj = _mm_shuffle_epi8(b, _mm_set1_epi8((char)(j >> 1)));
        bj = (j & 1) ? _mm_srli_epi16(bj, 4) : bj;
        bj = _mm_and_si128(bj, m0f);
        for (uint32_t k = 0; k < 4; k++) {
            const __m128i m = _mm_cmpeq_epi8(_mm_and_si128(bj, bit[k]), bit[k]);
            r = _mm_xor_si128(r, _mm_and_si128(A[k], m));
        }
    }

    return r;
}

/// \return a*b for two elements
static inline
__m256i gf16to32v_mul_u256(const __m256i a, const __m256i b) {
    const __m256i mod = _mm256_setr_epi8(0x09, 0x0F, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0x10, 0, 0, 0, 0x0B, 0,
                                         0x09, 0x0F, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0x10, 0, 0, 0, 0x0B, 0);
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    const __m256i top = _mm256_set1_epi8(15);
    __m256i A[4], M[4], bit[4];
    A[0] = a; M[0] = mod; bit[0] = _mm256_set1_epi8(1);
    for (uint32_t k = 1; k < 4; k++) {
        A[k] = gf16to32v_gf16_mulx_u256(A[k-1]);
        M[k] = gf16to32v_gf16_mulx_u256(M[k-1]);
        bit[k] = _mm256_set1_epi8((char)(1u << k));
    }

    __m256i r = _mm256_setzero_si256();
    for (int j = 31; j >= 0; j--) {
        // r = r*X mod f
        const __m256i c = _mm256_and_si256(_mm256_srli_epi16(_mm256_shuffle_epi8(r, top), 4), m0f);
        r = _mm256_or_si256(_mm256_slli_epi64(r, 4), _mm256_srli_epi64(_mm256_bslli_epi128(r, 8), 60));
        for (uint32_t k = 0; k < 4; k++) {
            const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(c, bit[k]), bit[k]);
            r = _mm256_xor_si256(r, _mm256_and_si256(M[k], m));
        }

        // r += b_j * a
        __m256i bj = _mm256_shuffle_epi8(b, _mm256_set1_epi8((char)(j >> 1)));
        bj = (j & 1) ? _mm256_srli_epi16(bj, 4) : bj;
        bj = _mm256_and_si256(bj, m0f);
        for (uint32_t k = 0; k < 4; k++) {
            const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(bj, bit[k]), bit[k]);
            r = _mm256_xor_si256(r, _mm256_and_si256(A[k], m));
        }
    }

    return r;
}

/// \param c[out]: = a*b
/// \param a[in]: element of GF(16^32)
/// \param b[in]: element of GF(16)
static inline
void gf16to32v_mul_gf16_u128(gf16to32 c, const gf16to32 a, const gf16 b) {
    const __m128i a128 = _mm_loadu_si128((const __m128i *)a);
    const __m128i b128 = _mm_set1_epi8((char)((b & 0xF) * 0x11));
    const __m128i c128 = gf16to32v_gf16_mul_u128(a128, b128);
    _mm_storeu_si128((__m128i *)c, c128);
}
#endif // USE_AVX2

#ifdef USE_AVX512
/// Lane layout of a `_u512` value: FOUR elements, i.e. the memory layout of
/// `gf16to32[4]`.

/// \return nibble-wise a*x in GF(16)
static inline
__m512i gf16to32v_gf16_mulx_u512(const __m512i a) {
    const __m512i m7 = _mm512_set1_epi8(0x77);
    const __m512i m1 = _mm512_set1_epi8(0x11);
    const __m512i h = _mm512_and_si512(_mm512_srli_epi64(a, 3), m1);
    const __m512i s = _mm512_slli_epi64(_mm512_and_si512(a, m7), 1);
    return _mm512_xor_si512(_mm512_xor_si512(s, h), _mm512_slli_epi64(h, 1));
}

/// \return a + b for four elements
static inline
__m512i gf16to32v_add_u512(const __m512i a, const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// \return a - b for four elements
static inline
__m512i gf16to32v_sub_u512(const __m512i a, const __m512i b) {
    return _mm512_xor_si512(a, b);
}

/// \return a*b for four elements
static inline
__m512i gf16to32v_mul_u512(const __m512i a, const __m512i b) {
    const __m512i mod = _mm512_broadcast_i32x4(_mm_setr_epi8(0x09, 0x0F, 0, 0, 0, 0, 0, 0,
                                                             0, 0, 0x10, 0, 0, 0, 0x0B, 0));
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    const __m512i top = _mm512_set1_epi8(15);
    __m512i A[4], M[4], bit[4];
    A[0] = a; M[0] = mod; bit[0] = _mm512_set1_epi8(1);
    for (uint32_t k = 1; k < 4; k++) {
        A[k] = gf16to32v_gf16_mulx_u512(A[k-1]);
        M[k] = gf16to32v_gf16_mulx_u512(M[k-1]);
        bit[k] = _mm512_set1_epi8((char)(1u << k));
    }

    __m512i r = _mm512_setzero_si512();
    for (int j = 31; j >= 0; j--) {
        // r = r*X mod f
        const __m512i c = _mm512_and_si512(_mm512_srli_epi16(_mm512_shuffle_epi8(r, top), 4), m0f);
        r = _mm512_or_si512(_mm512_slli_epi64(r, 4), _mm512_srli_epi64(_mm512_bslli_epi128(r, 8), 60));
        for (uint32_t k = 0; k < 4; k++) {
            const __mmask64 m = _mm512_test_epi8_mask(c, bit[k]);
            r = _mm512_xor_si512(r, _mm512_maskz_mov_epi8(m, M[k]));
        }

        // r += b_j * a
        __m512i bj = _mm512_shuffle_epi8(b, _mm512_set1_epi8((char)(j >> 1)));
        bj = (j & 1) ? _mm512_srli_epi16(bj, 4) : bj;
        bj = _mm512_and_si512(bj, m0f);
        for (uint32_t k = 0; k < 4; k++) {
            const __mmask64 m = _mm512_test_epi8_mask(bj, bit[k]);
            r = _mm512_xor_si512(r, _mm512_maskz_mov_epi8(m, A[k]));
        }
    }

    return r;
}
#endif // USE_AVX512

#ifdef USE_NEON
/// Lane layout of a `_u128` value: ONE element (uint8x16_t).

/// \return nibble-wise a*x in GF(16)
static inline
uint8x16_t gf16to32v_gf16_mulx_u128(const uint8x16_t a) {
    const uint8x16_t h = vandq_u8(vshrq_n_u8(a, 3), vdupq_n_u8(0x11));
    const uint8x16_t s = vshlq_n_u8(vandq_u8(a, vdupq_n_u8(0x77)), 1);
    return veorq_u8(veorq_u8(s, h), vshlq_n_u8(h, 1));
}

/// \return nibble-wise a*b in GF(16)
static inline
uint8x16_t gf16to32v_gf16_mul_u128(const uint8x16_t a,
                                   const uint8x16_t b) {
    const uint8x16_t m1 = vdupq_n_u8(0x11);
    uint8x16_t t, r, aa = a;

    t = vandq_u8(b, m1);
    t = vorrq_u8(t, vshlq_n_u8(t, 1)); t = vorrq_u8(t, vshlq_n_u8(t, 2));
    r = vandq_u8(aa, t);
    aa = gf16to32v_gf16_mulx_u128(aa);

    t = vandq_u8(vshrq_n_u8(b, 1), m1);
    t = vorrq_u8(t, vshlq_n_u8(t, 1)); t = vorrq_u8(t, vshlq_n_u8(t, 2));
    r = veorq_u8(r, vandq_u8(aa, t));
    aa = gf16to32v_gf16_mulx_u128(aa);

    t = vandq_u8(vshrq_n_u8(b, 2), m1);
    t = vorrq_u8(t, vshlq_n_u8(t, 1)); t = vorrq_u8(t, vshlq_n_u8(t, 2));
    r = veorq_u8(r, vandq_u8(aa, t));
    aa = gf16to32v_gf16_mulx_u128(aa);

    t = vandq_u8(vshrq_n_u8(b, 3), m1);
    t = vorrq_u8(t, vshlq_n_u8(t, 1)); t = vorrq_u8(t, vshlq_n_u8(t, 2));
    r = veorq_u8(r, vandq_u8(aa, t));
    return r;
}

/// \return a + b
static inline
uint8x16_t gf16to32v_add_u128(const uint8x16_t a, const uint8x16_t b) {
    return veorq_u8(a, b);
}

/// \return a - b
static inline
uint8x16_t gf16to32v_sub_u128(const uint8x16_t a, const uint8x16_t b) {
    return veorq_u8(a, b);
}

/// \return a*b
static inline
uint8x16_t gf16to32v_mul_u128(const uint8x16_t a, const uint8x16_t b) {
    static const uint8_t mod_tab[16] = {0x09, 0x0F, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0x10, 0, 0, 0, 0x0B, 0};
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const uint8x16_t top = vdupq_n_u8(15);
    uint8x16_t A[4], M[4], bit[4];
    A[0] = a; M[0] = vld1q_u8(mod_tab); bit[0] = vdupq_n_u8(1);
    for (uint32_t k = 1; k < 4; k++) {
        A[k] = gf16to32v_gf16_mulx_u128(A[k-1]);
        M[k] = gf16to32v_gf16_mulx_u128(M[k-1]);
        bit[k] = vdupq_n_u8((uint8_t)(1u << k));
    }

    uint8x16_t r = vdupq_n_u8(0);
    for (int j = 31; j >= 0; j--) {
        // r = r*X mod f
        const uint8x16_t c = vshrq_n_u8(vqtbl1q_u8(r, top), 4);
        const uint64x2_t r64 = vreinterpretq_u64_u8(r);
        const uint64x2_t s64 = vorrq_u64(vshlq_n_u64(r64, 4),
                                         vshrq_n_u64(vextq_u64(vdupq_n_u64(0), r64, 1), 60));
        r = vreinterpretq_u8_u64(s64);
        for (uint32_t k = 0; k < 4; k++) {
            r = veorq_u8(r, vandq_u8(M[k], vtstq_u8(c, bit[k])));
        }

        // r += b_j * a
        uint8x16_t bj = vqtbl1q_u8(b, vdupq_n_u8((uint8_t)(j >> 1)));
        bj = (j & 1) ? vshrq_n_u8(bj, 4) : vandq_u8(bj, m0f);
        for (uint32_t k = 0; k < 4; k++) {
            r = veorq_u8(r, vandq_u8(A[k], vtstq_u8(bj, bit[k])));
        }
    }

    return r;
}
#endif // USE_NEON
