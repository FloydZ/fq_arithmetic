#ifndef FQ_ARITHMETIC_ARITH_H
#define FQ_ARITHMETIC_ARITH_H

#include <stdint.h>
#include <stdio.h>

typedef uint8_t gf2;
typedef uint16_t gf2to12;

gf2to12 gf2to12_add(const gf2to12 a,
                 const gf2to12 b) {
    return a ^ b;
}

// sage gave me this polynomial
// x^12 + x^3 + 1
#define MODULUS 0b1000000001001

/// \return a*b
gf2to12 gf2to12_mul(const gf2to12 a,
                    const gf2to12 b) {
    gf2to12 r;
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
static inline gf2to12 gf2to12_mul_v2(const gf2to12 a,
                                     const gf2to12 b) {
    gf2to12 result = -(a & 1) & b;
    gf2to12 tmp = b;
    for(uint32_t i=1 ; i<12 ; i++) {
        tmp = ((tmp << 1) ^ (-(tmp >> 11) & MODULUS));
        result = result ^ (-(a >> i & 1) & tmp);
        result ^= 0;
    }
    return result;
}

gf2to12 gf2to12_mul_gf2(const gf2to12 a,
                        const gf2 b) {
    gf2to12 c = b;
    return a & (-c);
}

/// \return a^-1
static inline gf2to12 gf2to12_inv(const gf2to12 a) {
    gf2to12 result = a;
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


/// horizontal xor, but not withing a single limb, but over the 16 -16bit limbs
/// \param in
/// \return
static inline uint16_t gf2to12_hadd_u256(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 2));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(in, 4));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(ret, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return _mm256_extract_epi16(ret, 0);
}

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

#ifdef USE_AVX512

/// NOTE: needs BW + VL
static inline __m256i gf2to12v_mul_u256_avx512(const __m256i a,
                                               const __m256i b) {
    const __m256i mod  = _mm256_set1_epi16((short)MODULUS);
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i m1   = _mm256_set1_epi16(1<<11u);
    __m256i r, t;
    __mmask16 m;

    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 15));
    r = _mm256_mask_blend_epi16(m, zero, b);
    t = b;

    // i=1
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 14));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=2
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 13));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=3
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 12));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=4
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 11));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=5
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 10));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=6
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 9));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=7
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 8));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=8
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 7));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=9
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 6));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=10
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 5));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    // i=11
    m = _mm256_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm256_slli_epi16(t, 1) ^ _mm256_mask_blend_epi16(m, zero, mod);
    m = _mm256_movepi16_mask(_mm256_slli_epi16(a, 4));
    r ^= _mm256_mask_blend_epi16(m, zero, t);

    return r;
}
/// NOTE: needs BW + VL
static inline __m512i gf2to12v_mul_u512(const __m512i a,
                                        const __m512i b) {
    const __m512i mod  = _mm512_set1_epi16((short)MODULUS);
    const __m512i zero = _mm512_set1_epi8(0);
    const __m512i m1   = _mm512_set1_epi16(1<<11u);
    __m512i r, t;
    __mmask32 m;

    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 15));
    r = _mm512_mask_blend_epi16(m, zero, b);
    t = b;

    // i=1
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 14));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=2
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 13));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=3
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 12));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=4
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 11));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=5
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 10));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=6
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 9));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=7
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 8));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=8
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 7));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=9
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 6));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=10
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 5));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    // i=11
    m = _mm512_cmp_epu16_mask(t&m1, m1, _MM_CMPINT_EQ);
    t = _mm512_slli_epi16(t, 1) ^ _mm512_mask_blend_epi16(m, zero, mod);
    m = _mm512_movepi16_mask(_mm512_slli_epi16(a, 4));
    r ^= _mm512_mask_blend_epi16(m, zero, t);

    return r;
}
#endif
///
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


static inline __m256i gf2to12v_mul_u256_v2(const __m256i a,
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
    
    // i = 11;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 4) & m1) ^ (_mm256_srli_epi16(t, 4)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 4) & m1) ^ (_mm256_srli_epi16(a, 4)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    return r;
}

/// @param a
/// @param b in gf2, not compresses: a single bit in 
/// @return
static inline __m256i gf2to12v_mul_gf2_u256(const __m256i a,
                                            const __m256i b) {
    const __m256i m1 = _mm256_set1_epi16(-1);
    const __m256i t1 = _mm256_sign_epi16(b, m1);
    return a & t1;
}

/// @param a
/// @param b in gf2, not compresses: a single bit in 
/// @return
static inline __m128i gf2to12v_mul_gf2_u128(const __m128i a,
                                            const __m128i b) {
    const __m128i m1 = _mm_set1_epi16(-1);
    const __m128i t1 = _mm_sign_epi16(b, m1);
    return a & t1;
}

///
/// @param a
/// @param b in gf2
/// @return
static inline __m256i gf2to12v_mul_gf2_u256_v2(const __m256i a,
                                               const __m256i b) {
    return a & b;
}

///
/// @param a
/// @param b in gf2
/// @return
static inline __m128i gf2to12v_mul_gf2_u128_v2(const __m128i a,
                                               const __m128i b) {
    return a & b;
}

#elif defined(USE_NEON)
#else
#endif

#endif //FQ_ARITHMETIC_ARITH_H

#include "vector.h"
#include "matrix.h"
