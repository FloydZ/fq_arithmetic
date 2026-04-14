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
//  reduction by y^32 + y^3 + 1
typedef uint8_t gf16to32[16];


void gf16to32_add(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    for (int i = 0; i < 16; i++) {
        r[i] = a[i] ^ b[i];
    }
}

void gf16to32_sub(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    gf16to32_add(r, a, b);
}

void gf16to32_mul(gf16to32 r, const gf16to32 a, const gf16to32 b) {
    uint8_t t[32] = {0};

    /* schoolbook multiply over GF(16) */
    for (uint32_t i = 0; i < 32; i++) {
        uint8_t ai = (a[i >> 1] >> ((i & 1) * 4)) & 0xF;
        for (uint32_t j = 0; j < 32; j++) {
            uint8_t bj = (b[j >> 1] >> ((j & 1) * 4)) & 0xF;
            t[i + j] ^= gf16_mul(ai, bj);
        }
    }

    /* modular reduction */
    for (uint32_t k = 63; k >= 32; k--) {
        uint8_t c = t[k];
        if (!c) continue;
        t[k] = 0;
        t[k - 32] ^= c;
        t[k - 29] ^= c;
    }

    /* repack result */
    memset(r, 0, 16);
    for (int i = 0; i < 32; i++) {
        r[i >> 1] |= (t[i] & 0xF) << ((i & 1) * 4);
    }
}

/// TODO test
void gf16to32_mul_v2(gf16to32 c, const gf16to32 a, const gf16to32 b) {
    const uint8_t mask = 0xF;
    const __m128i zero = _mm_set1_epi8(0);
    __m128i r = zero, mod = _mm_set1_epi8(0b1001);

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

// TODO test
void gf16to32_mul_v3(gf16to32 c, const gf16to32 a, const gf16to32 b) {
    const __uint128_t aa = *(__uint128_t *)a;
    const __uint128_t bb = *(__uint128_t *)b;
    __uint128_t r = 0;

    const __uint128_t mod = 0b1001;
    for (uint64_t i = 0; i < 128; i++) {
        const __uint128_t t0 = r+r;
        const __uint128_t t1 = (((__int128_t)r) >> 127) & mod;
        const __uint128_t t2 = (-((((__int128_t)bb) >>   i) & (__int128_t)1ll)) & aa;

        r = t0 ^ t1 ^ t2;
    }

    *(__uint128_t *)c = r;
}

// a^{-1} = a^(16^32 - 2)
void gf_div(gf16to32 r, const gf16to32 a, const gf16to32 b)
{
    gf16to32 x, y;
    memcpy(x, b, 16);

    memcpy(y, b, 16);
    for (int i = 0; i < 32 * 4 - 1; i++) { /* log2(16^32 - 2) */
        gf16to32_mul(y, y, y);  /* square */
        gf16to32_mul(y, y, b);  /* multiply */
    }

    gf16to32_mul(r, a, y);
}
