#pragma once

#include <emmintrin.h>
#include <stdint.h>
#include <string.h>
#ifdef USE_AVX2
#include <immintrin.h>
#endif

#include "../helper.h"
#include "../16/arith.h"

// GF(16^32) element: 32 GF(16) symbols packed as nibbles
//     x^32 + B*x^28 + x^21 + F*x^2 + 9
typedef uint8_t gf16to32[16];


static
void gf16to32_set_u(gf16to32 r, const uint64_t d) {
    *(((uint64_t *)r) + 0) = d;
    *(((uint64_t *)r) + 1) = 0;
}

static
void gf16to32_set_uu(gf16to32 r, const uint64_t hi, const uint64_t lo) {
    *(((uint64_t *)r) + 0) = lo;
    *(((uint64_t *)r) + 1) = hi;
}

static inline
void gf16to32_set_zero(gf16to32 v) {
    memset(v, 0, 16);
}


static
int gf16to32_cmp(const gf16to32 a, const gf16to32 b) {
    for (int64_t i = 15; i >= 0; i--) {
        if (a[i] > b[i]) {
            return 1;
        }

        if (a[i] > b[i]) {
            return -1;
        }
    }

    return 0;
}

static
void gf16to32_add(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    for (uint32_t i = 0; i < 16; i++) {
        r[i] = a[i] ^ b[i];
    }
}

static
void gf16to32_sub(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    gf16to32_add(r, a, b);
}

void gf16to32_mul_gf16(gf16to32 c, const gf16to32 a, const gf16 b) {
    for (uint64_t i = 0; i < 16; i++) {
        c[i] = gf16_mul_v3(a[i] & 0xFFFF, b);
        c[i] |= gf16_mul_v3(a[i] >> 4, b) << 4;
    }
}

/// NOTE: non ct
static void gf16to32_mul(gf16to32 r, const gf16to32 a, const gf16to32 b) {
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
static
void gf16to32_mul_v2(gf16to32 c, const gf16to32 a, const gf16to32 b) {
    const __uint128_t aa = *(__uint128_t *)a;
    const __uint128_t bb = *(__uint128_t *)b;
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

    *(__uint128_t *)c = r;
}


#ifdef USE_AVX2
// static
// void gf16to32_add_u128(gf16to32 c, const gf16to32 a, const gf16to32 b) {
//     const __m128i a128 = _mm_loadu_si128((const __m128i *)a);
//     const __m128i b128 = _mm_loadu_si128((const __m128i *)b);
//     const __m128i c128 = _mm_xor_si128(a128, b128);
//     _mm_storeu_si128((__m128i *)c, c128);
// }
static
__m128i gf16to32_add_u128(const __m128i a128, const __m128i b128) {
    return _mm_xor_si128(a128, b128);
}

static
__m256i gf16to32_add_u256(const __m256i a256, const __m256i b256) {
    return _mm256_xor_si256(a256, b256);
}

/// TODO test
static
void gf16to32_mul_u128(gf16to32 c, const gf16to32 a, const gf16to32 b) {
    const __m128i zero = _mm_set1_epi8(0);
    const __m128i mod = _mm_set1_epi8(0b1001);
    __m128i r = zero;

    const __m128i aa = _mm_loadu_si128((const __m128i *)a);
    for (uint32_t i = 0; i < 16; i++) {
        for (uint32_t j = 0; j < 8; j++) {
            uint8_t v = -((b[i] >> j) & 1);
            const __m128i vv = _mm_set1_epi8(v);

            const __m128i t0 = _mm_alignr_epi8(zero, r, 15);
            const __m128i t1 = _mm_blendv_epi8(zero, mod, t0);
            const __m128i t2 = _mm_and_si128(vv, aa);
            const __m128i t3 = _mm_add_epi64(r, r);
            r = t1 ^ t2 ^ t3;
        }
    }

    _mm_storeu_si128((__m128i *)c, r);
}

void gf16to32_mul_gf16_u128(gf16to32 c, const gf16to32 a, const gf16 b) {
    const __m128i a128 = _mm_loadu_si128((const __m128i *)a);
    const __m128i b128 = _mm_set1_epi8(b);
    const __m128i c128 = gf16v_mul_u128(a128, b128);
    _mm_storeu_si128((__m128i *)c, c128);
}
#endif

// a^{-1} = a^(16^32 - 2)
static
void gf16to32_inv(gf16to32 r, const gf16to32 b) {
    gf16to32 y;
    memcpy(y, b, 16);
    for (uint64_t i = 0; i < 32 * 4 - 2; i++) {
        gf16to32_mul(y, y, y);
        gf16to32_mul(y, y, b);
    }

    gf16to32_mul(r, y, y);
}
