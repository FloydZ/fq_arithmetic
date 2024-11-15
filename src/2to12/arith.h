#ifndef FQ_ARITHMETIC_ARITH_H
#define FQ_ARITHMETIC_ARITH_H

typedef uint16_t ff_t;

ff_t gf2to12_add(const ff_t a,
                 const ff_t b) {
    return a ^ b;
}

// sage gave me this polynomial
// x^12 + x^7 + x^6 + x^5 + x^3 + x + 1
// #define MODULUS 0b1000011101011
// sage gave me this polynomial
// x^12 + x^3 + 1
#define MODULUS 0b1000000001001

/// \return a*b
ff_t gf2to12_mul(const ff_t a,
                 const ff_t b) {
    ff_t r;
    r = (-(b>>11u     ) & a);
    r = (-(b>>10u & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>9u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>8u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>7u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>6u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>5u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>4u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>3u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>2u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    r = (-(b>>1u  & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
    return (-(b   & 1u) & a) ^ (-(r>>11) & MODULUS) ^ (r+r);
}

/// \return a*b
static inline ff_t gf2to12_mul_v2(const ff_t a,
                                  const ff_t b) {
    ff_t result = -(a & 1) & b;
    ff_t tmp = b;
    for(uint32_t i=1 ; i<12 ; i++) {
        tmp = ((tmp << 1) ^ (-(tmp >> 11) & MODULUS));
        result = result ^ (-(a >> i & 1) & tmp);
    }
    return result;
}

/// \return a^-1
static inline ff_t gf2to12_inv(const ff_t a) {
    ff_t result = a;
    for(int i=0 ; i<10 ; i++) {
        result = gf2to12_mul(result, result);
        result = gf2to12_mul(result, a);
    }
    result = gf2to12_mul(result, result);
    return result;
}

/// \return needed for bitslicing
void compute_lookup_table() {
    printf("[\n");
    for (uint32_t a = 0; a < (1u << 12u); a++) {
        for (uint32_t b = 0; b < (1u<<12u); b++) {
            uint32_t c = gf2to12_mul(a, b);
            printf("0x%x,", c);
        }
        printf("\n");
    }
    printf("]\n");
}

#ifdef USE_AVX2
#include <immintrin.h>

static inline __m128i gf2to12v_mul_u128(const __m128i a,
                                        const __m128i b) {
    const __m128i mod  = _mm_set1_epi16((short)MODULUS);
    const __m128i one  = _mm_set1_epi8(-1);
    const __m128i zero = _mm_set1_epi8(0);
    const uint32_t mask = 0b1010101010101010;
    __m128i mr, r;
    uint32_t ma;
    // TODO
    // 11

    return r;
}

// TODO also vectorize `gf2to12_mul_v2`
static inline __m256i gf2to12v_mul_u256(const __m256i a,
                                        const __m256i b) {
    const __m256i mod  = _mm256_set1_epi16((short)MODULUS);
    const __m256i one  = _mm256_set1_epi8(-1);
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i m1 = _mm256_set1_epi16(0x8000);
    const __m256i m2 = _mm256_set1_epi16(0x0080);
    __m256i ma, mr, r;


    // 11
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 4) & m1) ^ (_mm256_srli_epi16(b, 4)));
    r = ma & a;
    // 10
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 5) & m1) ^ (_mm256_srli_epi16(b, 3)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 9
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 6) & m1) ^ (_mm256_srli_epi16(b, 2)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 8
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 7) & m1) ^ (_mm256_srli_epi16(b, 1)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 7
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 8)) ^ (b & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 6
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 9)) ^ (_mm256_slli_epi16(b, 1)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 5
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 10)) ^ (_mm256_slli_epi16(b, 2)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 4
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 11)) ^ (_mm256_slli_epi16(b, 3)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 3
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 12)) ^ (_mm256_slli_epi16(b, 4) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 2
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 13)) ^ (_mm256_slli_epi16(b, 5) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 1
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 14)) ^ (_mm256_slli_epi16(b, 6) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 0
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 4) & m1) ^ (_mm256_srli_epi16(r, 4)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 15)) ^ (_mm256_slli_epi16(b, 7) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    return r;
}

#elif defined(USE_NEON)
#else
#endif

#endif //FQ_ARITHMETIC_ARITH_H
