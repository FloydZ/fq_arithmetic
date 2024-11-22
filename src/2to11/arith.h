#pragma once

#include <stdint.h>

typedef uint16_t ff_t;

// 0b100000000101
#define MODULUS 0x805

ff_t gf2to11_add(const ff_t a,
                 const ff_t b) {
	return a ^ b;
}

ff_t gf2to11_mul(const ff_t a,
                 const ff_t b) {
    ff_t result = -(a & 1) & b;
    ff_t tmp = b;
    for(int i=1 ; i<11 ; i++) {
        tmp = ((tmp << 1) ^ (-(tmp >> 10) & MODULUS));
        result = result ^ (-(a >> i & 1) & tmp);
    }
    return result;
}

/// \return a^-1
static inline ff_t gf2to11_inv(const ff_t a) {
    ff_t result = a;
    for(int i=0 ; i<9 ; i++) {
        result = gf2to11_mul(result, result);
        result = gf2to11_mul(result, a);
    }

    result = gf2to11_mul(result, result);
    return result;
}


#ifdef USE_AVX2
#include <immintrin.h>

static inline __m256i gf2to11v_mul_u256(const __m256i a,
                                        const __m256i b) {
    const __m256i mod  = _mm256_set1_epi16((short)MODULUS);
    const __m256i one  = _mm256_set1_epi8(-1);
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i m1 = _mm256_set1_epi16(0x8000);
    const __m256i m2 = _mm256_set1_epi16(0x0080);
    __m256i mt, mm, r, t;

    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 15)) ^ (_mm256_slli_epi16(a, 7) & m2));
    r = mm & b;
    t = b;

    // i = 1;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 14)) ^ (_mm256_slli_epi16(a, 6) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 2;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 13)) ^ (_mm256_slli_epi16(a, 5) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 3;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 12)) ^ (_mm256_slli_epi16(a, 4) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 4;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 11)) ^ (_mm256_slli_epi16(a, 3)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 5;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 10)) ^ (_mm256_slli_epi16(a, 2)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 6;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 9)) ^ (_mm256_slli_epi16(a, 1)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 7;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 8)) ^ (a & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 8;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 7) & m1) ^ (_mm256_srli_epi16(a, 1)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 9;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 6) & m1) ^ (_mm256_srli_epi16(a, 2)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 10;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 5) & m1) ^ (_mm256_srli_epi16(a, 3)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    return r;
}
#elif defined(USE_NEON)
#else
#endif

#include "vector.h"
#include "matrix.h"
