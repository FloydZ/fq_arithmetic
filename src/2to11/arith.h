#pragma once

#include <stdint.h>
#include "../2/arith.h"

typedef uint16_t gf2to11;

// 0b100000000101
#define MODULUS 0x805

gf2to11 gf2to11_add(const gf2to11 a,
                    const gf2to11 b) {
	return a ^ b;
}

gf2to11 gf2to11_mul(const gf2to11 a,
                    const gf2to11 b) {
    gf2to11 result = -(a & 1) & b;
    gf2to11 tmp = b;
    for(int i=1 ; i<11 ; i++) {
        tmp = ((tmp << 1) ^ (-(tmp >> 10) & MODULUS));
        result = result ^ (-(a >> i & 1) & tmp);
    }
    return result;
}

/// \return a*b
gf2to11 gf2to11_mul_v2(const gf2to11 a,
                       const gf2to11 b) {
    gf2to11 r;
    r = (-(b>>10u     ) & a);
    r = (-(b>>9u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>8u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>7u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>6u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>5u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>4u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>3u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>2u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    r = (-(b>>1u  & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
    return (-(b   & 1u) & a) ^ (-(r>>10) & MODULUS) ^ (r+r);
}

/// \return a^-1
static inline gf2to11 gf2to11_inv(const gf2to11 a) {
    gf2to11 result = a;
    for(int i=0 ; i<9 ; i++) {
        result = gf2to11_mul(result, result);
        result = gf2to11_mul(result, a);
    }

    result = gf2to11_mul(result, result);
    return result;
}

/// scalar multiplication
/// \param a
/// \param b
/// \return a*b
gf2to11 gf2to11_mul_gf2(const gf2to11 a,
                        const gf2 b) {
    gf2to11 c = b;
    return a & (-c);
}


#ifdef USE_AVX2
#include <immintrin.h>


/// \param in[in]: 2bytes which will be extended to a avx2 register
///     = [in0, ..., in_15], where in_i is just a single bit
/// \return [in_0, ..., in_15], extened to 16bit limbs
static inline __m256i mirath_ff_mu_expand_ff_x16_u256(const uint8_t *in) {
    const uint8_t t11 = *(in + 0);
    const uint8_t t12 = *(in + 1);

    const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
    const uint64_t t22 = _pdep_u64(t12, 0x0101010101010101);

    const __m128i t1 = _mm_set_epi64x(t22, t21);
    return _mm256_cvtepu8_epi16(t1);
}

/// \param in[in]: bytes which will be extended to a sse register
///     = [in0, ..., in_7], where in_i is just a single bit
/// \return [in_0, ..., in_7], extened to 16bit limbs
static inline __m128i mirath_ff_mu_expand_ff_x8_u256(const uint8_t *in) {
    const uint32_t t11 = *in;
    const uint64_t t21 = _pdep_u64(t11, 0x0101010101010101);
    const __m128i t1 = _mm_set_epi64x(0, t21);
    return _mm_cvtepi8_epi16(t1);
}


/// horizontal xor, but not withing a single limb, but over the 16 -16bit limbs
/// \param in
/// \return
static inline uint16_t gf2to11_hadd_u256(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 2));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(in, 4));
    ret = _mm256_xor_si256(ret, _mm256_srli_si256(ret, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return _mm256_extract_epi16(ret, 0);
}

static inline __m256i gf2to11v_mul_u256(const __m256i a,
                                        const __m256i b) {
    const __m256i mod  = _mm256_set1_epi16((short)MODULUS);
    const __m256i one  = _mm256_set1_epi8(-1);
    const __m256i zero = _mm256_set1_epi8(0);
    const __m256i m1 = _mm256_set1_epi16(0x8000);
    const __m256i m2 = _mm256_set1_epi16(0x0080);
    __m256i ma, mr, r;


    // 10
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 5) & m1) ^ (_mm256_srli_epi16(b, 3)));
    r = ma & a;

    // 9
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 6) & m1) ^ (_mm256_srli_epi16(b, 2)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 8
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 7) & m1) ^ (_mm256_srli_epi16(b, 1)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 7
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 8)) ^ (b & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 6
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 9)) ^ (_mm256_slli_epi16(b, 1)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 5
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 10)) ^ (_mm256_slli_epi16(b, 2)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 4
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 11)) ^ (_mm256_slli_epi16(b, 3)));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 3
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 12)) ^ (_mm256_slli_epi16(b, 4) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 2
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 13)) ^ (_mm256_slli_epi16(b, 5) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 1
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 14)) ^ (_mm256_slli_epi16(b, 6) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    // 0
    mr = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(r, 5) & m1) ^ (_mm256_srli_epi16(r, 3)));
    ma = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(b, 15)) ^ (_mm256_slli_epi16(b, 7) & m2));
    r =  (ma & a) ^ (mr & mod) ^ _mm256_add_epi16(r, r);
    return r;
}

static inline __m256i gf2to11v_mul_u256_v2(const __m256i a,
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
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 14)) ^ (_mm256_slli_epi16(a, 6) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 2;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 13)) ^ (_mm256_slli_epi16(a, 5) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 3;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 12)) ^ (_mm256_slli_epi16(a, 4) & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 4;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 11)) ^ (_mm256_slli_epi16(a, 3)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 5;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 10)) ^ (_mm256_slli_epi16(a, 2)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 6;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 9)) ^ (_mm256_slli_epi16(a, 1)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 7;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 8)) ^ (a & m2));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 8;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 7) & m1) ^ (_mm256_srli_epi16(a, 1)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 9;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 6) & m1) ^ (_mm256_srli_epi16(a, 2)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    // i = 10;
    mm = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(t, 5) & m1) ^ (_mm256_srli_epi16(t, 3)));
    mt = _mm256_blendv_epi8(zero, one, (_mm256_slli_epi16(a, 5) & m1) ^ (_mm256_srli_epi16(a, 3)));
    t  = _mm256_slli_epi16(t, 1) ^ (mm & mod);
    r ^= (mt & t);
    
    return r;
}

/// \param a
/// \param b in gf2, not compresses: a single bit in
/// \return
static inline __m256i gf2to11v_mul_gf2_u256(const __m256i a,
                                            const __m256i b) {
    const __m256i m1 = _mm256_set1_epi16(-1);
    const __m256i t1 = _mm256_sign_epi16(b, m1);
    return a & t1;
}

/// \param a
/// \param b in gf2, not compresses: a single bit in
/// \return
static inline __m128i gf2to11v_mul_gf2_u128(const __m128i a,
                                            const __m128i b) {
    const __m128i m1 = _mm_set1_epi16(-1);
    const __m128i t1 = _mm_sign_epi16(b, m1);
    return a & t1;
}
#elif defined(USE_NEON)

#include <arm_neon.h>

/// TODO
/// @param in
/// @return
static inline
uint16x8_t gf2to11v_expand_ff_x8_u128(const uint8_t *in) {
    const uint8_t t = *in;
    const uint8x8_t m = vdup_n_u8(0x01);
    const uint8x8_t shift = {(uint8_t)0, (uint8_t)-1, (uint8_t)-2, (uint8_t)-3, (uint8_t)-4, (uint8_t)-5, (uint8_t)-7};
    const uint8x8_t t1 = vdup_n_u8(t);
    const uint8x8_t t2 = vshl_u8(t1, shift);
    const uint8x8_t t3 = t2 & m;
    uint16x8_t result = vmovl_u8(t3);
    return result;
}

///
/// @param in
/// @return
static inline
uint16x8x2_t gf2to11v_expand_ff_x16_u256(const uint8_t *in) {
    const uint16_t t = *((uint16_t *)in);
    const uint8x16_t m = vdupq_n_u8(0x01);
    const uint8x16_t shift = {(uint8_t)0, (uint8_t)-1, (uint8_t)-2, (uint8_t)-3, (uint8_t)-4, (uint8_t)-5, (uint8_t)-7, (uint8_t)0, (uint8_t)-1, (uint8_t)-2, (uint8_t)-3, (uint8_t)-4, (uint8_t)-5, (uint8_t)-7};
    const uint16x8_t t1 = vdupq_n_u16(t);
    const uint8x16_t t2 = vshlq_u8(t1, shift);
    const uint8x16_t t3 = t2 & m;

    const uint16x8_t l = vmovl_u8(vget_low_u8(t3));
    const uint16x8_t h = vmovl_u8(vget_high_u8(t3));
    const uint16x8x2_t r = { l, h };
    return r;
}

/// TODO
/// @param a
/// @param b
/// @return
static inline uint16x8_t gf2to11v_mul_u128(const uint16x8_t a,
                                           const uint16x8_t b) {
    uint16x8_t m = vdupq_n_u16(MODULUS), r;

    r =  vshrq_n_s16(vshlq_n_u16(b,  5), 15) & a;
    r = (vshrq_n_s16(vshlq_n_u16(b,  6), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b,  7), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b,  8), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b,  9), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 10), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 11), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 12), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 13), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 14), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);
    r = (vshrq_n_s16(vshlq_n_u16(b, 15), 15) & a) ^ (vshrq_n_s16(vshlq_n_u16(r, 5), 15) & m) ^ (r+r);

    return r;
}

static inline uint16x8x2_t gf2to11v_mul_u256(const uint16x8x2_t a,
                                             const uint16x8x2_t b) {
    uint16x8x2_t r;
    r.val[0] = gf2to11v_mul_u128(a.val[0], b.val[0]);
    r.val[1] = gf2to11v_mul_u128(a.val[1], b.val[1]);
    return r;
}
#else
#endif
