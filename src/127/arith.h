#pragma once

#include <stdint.h>
#include "lookup_table.h"


#define PRIME 127
#define PRIME_LOG_CEIL 7

// Select low 8-bit, skip the high 8-bit in 16 bit type
const uint8_t shuff_low_half[32] __attribute__((aligned(32))) = {
        0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};

typedef uint8_t gf127;

static inline
gf127 gf127_add(const gf127 a,
                const gf127 b) {
    return (a+b) % PRIME;
}

static inline
gf127 gf127_sub(const gf127 a,
                const gf127 b) {
    return (a+PRIME-b) % PRIME;
}

static inline
gf127 gf127_mul(const gf127 a,
                const gf127 b) {
    return (a*b)%PRIME;
}

static inline
gf127 gf127_neg(const gf127 a) {
    return (PRIME - a) % PRIME;
}

inline static uint32_t gf127v_red_u32(uint32_t Z) {
    const uint32_t mask = 0x7F7F7F7F;
    const uint32_t one = 0x01010101;
    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    uint32_t C  = ((Z+one) & ~mask) ;
    return Z + (C>>PRIME_LOG_CEIL) - C ;
}

inline static uint64_t gf127v_red_u64(uint64_t Z) {
    const uint64_t mask = 0x7F7F7F7F7F7F7F7F;
    const uint64_t one  = 0x0101010101010101;
    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    uint64_t C  = ((Z+one) & ~mask) ;
    return Z + (C>>PRIME_LOG_CEIL) - C ;
}

///
inline static uint32_t gf127v_add_u32(const uint32_t a,
                                      const uint32_t b) {
    return gf127v_red_u32(a + b);
}

///
inline static uint64_t gf127v_add_u64(const uint64_t a,
                                      const uint64_t b) {
    return gf127v_red_u64(a + b);
}

///
inline static uint32_t gf127v_sub_u32(const uint32_t a,
                                      const uint32_t b) {
    return gf127v_red_u32(a - b + 0x7F7F7F7F);
}

///
inline static uint64_t gf127v_sub_u64(const uint64_t a,
                                      const uint64_t b) {
    return gf127v_red_u64(a - b + 0x7F7F7F7F7F7F7F7F);
}

///
inline static uint32_t gf127v_scalar_u32(const uint32_t a,
                                         const uint8_t b) {
    const uint32_t mask1 = 0x00FF00FF;
    const uint32_t mask2 = 0xFF00FF00;
    const uint32_t mask3 = 0x007F007F;
    const uint32_t mask4 = 0x7F007F00;

    const uint32_t t1 = (a&mask1) * b;
    const uint64_t t2 = ((uint64_t)a&mask2) * b;

    const uint32_t v1 = ((t1 >> 7) + (t1 & mask3)) & mask1;
    const uint32_t v2 = ((t2 >> 7) + (t2 & mask4)) & mask2;

    return gf127v_red_u32(v1 ^ v2);
}

/// basically a rref core operation
/// a-a*b
inline static uint32_t gf127v_scalar_sub_u32(const uint32_t a,
                                             const uint64_t b,
                                             const uint8_t c) {
    const uint32_t mask  = 0x7F7F7F7F;
    const uint32_t one   = 0x01010101;
    const uint32_t mask1 = 0x00FF00FF;
    const uint32_t mask2 = 0xFF00FF00;
    const uint32_t mask3 = 0x007F007F;
    const uint32_t mask4 = 0x7F007F00;

    const uint32_t t1 = (b&mask1) * c;
    const uint64_t t2 = ((uint64_t)b&mask2) * c;

    const uint32_t v1 = ((t1 >> 7) + (t1 & mask3)) & mask1;
    const uint32_t v2 = ((t2 >> 7) + (t2 & mask4)) & mask2;

    uint32_t Z = v1 ^ v2;
    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    Z = mask - Z;
    Z = Z + a;

    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    uint32_t C  = ((Z+one) & ~mask) ;
    return Z + (C>>PRIME_LOG_CEIL) - C ;
}

#ifdef USE_AVX2
#include <immintrin.h>

/*
 * Fix width 16-bit Barrett modulo reduction Q = 127
 * c = a % q
 */
#define barrett_red16(c, a, t, c127, c516, c1)                 \
    t = _mm256_add_epi16(a, c1);     /* t = (a + 1) */         \
    t = _mm256_mulhi_epu16(t, c516); /* t = (a * 516) >> 16 */ \
    t = _mm256_mullo_epi16(t, c127); /* t = (t * Q) */         \
    a = _mm256_sub_epi16(a, t);      /* a = (a - t)*/

/*
 * Fix width 8-bit Barrett modulo reduction Q = 127
 * c = a % q
 */
#define barrett_red8(a, t, c127, c1) \
    t = _mm256_srli_epi16(a, 7);     \
    t = _mm256_and_si256(t, c1);     \
    a = _mm256_add_epi8(a, t);       \
    a = _mm256_and_si256(a, c127);

#define barrett_red8_half(a, t, c127, c1) \
    t = _mm_srli_epi16(a, 7);             \
    t = _mm_and_si128(t, c1);             \
    a = _mm_add_epi8(a, t);               \
    a = _mm_and_si128(a, c127);

/*
 * Fix width 16-bit Barrett multiplication Q = 127
 * c = (a * b) % q
 */
#define barrett_mul_u16(c, a, b, t) \
    a = _mm256_mullo_epi16(a, b);   \
    t = _mm256_srli_epi16(a, 7);    \
    a = _mm256_add_epi16(a, t);     \
    t = _mm256_slli_epi16(t, 7);    \
    c = _mm256_sub_epi16(a, t);


/// number of 8 bit elements in an avx register
#define LESS_WSZ 32

/// number of avx registers needed for a full row in a generator matrix
#define NW ((N + LESS_WSZ - 1) / LESS_WSZ)

/// original reduction formula
#define W_RED127(x)                                                            \
  {                                                                            \
    t = _mm256_srli_epi16(x, 7);                                               \
    t = _mm256_and_si256(t, c01);                                              \
    x = _mm256_add_epi8(x, t);                                                 \
    x = _mm256_and_si256(x, c7f);                                              \
  }

/// new reduction formula, catches the case where input is q=127
#define W_RED127_(x)                                                           \
  x = _mm256_and_si256(                                                        \
      _mm256_add_epi8(_mm256_and_si256(                                        \
                          _mm256_srli_epi16(_mm256_add_epi8(x, c01), 7), c01), \
                      x),                                                      \
      c7f);

/// NOTE: assumes that each gf31 element is in a single uint8_t
__m256i gf127v_red_u256(const __m256i a) {
    return a;
}
                                      
/// NOTE: assumes that each gf31 element is in a single uint8_t
__m256i gf127v_add_u256(const __m256i a,
                        const __m256i b) {
   const __m256i c7f = _mm256_set1_epi8(127);
   const __m256i c01 = _mm256_set1_epi8(1);
    __m256i t;
    __m256i c = _mm256_add_epi8(a, b);
    W_RED127(c);
    return c;
}

/// NOTE: assumes that each gf127 element is in a single uint8_t
/// \return [a_0 * b_0, a_1*b_1, ..., a_31 * b_31]
__m256i gf127v_mul_u256(const __m256i a,
                        const __m256i b) {
    __m256i a_lo, b_lo, a_hi, b_hi, t, r;
    __m128i tmp;

    const __m256i shuffle = _mm256_load_si256((const __m256i *)shuff_low_half),
                  c8_127 = _mm256_set1_epi8(127), 
                  c8_1 = _mm256_set1_epi8(1);

    tmp = _mm256_extracti128_si256(a, 0);
    a_lo = _mm256_cvtepu8_epi16(tmp);
    tmp = _mm256_extracti128_si256(a, 1);
    a_hi = _mm256_cvtepu8_epi16(tmp);
    tmp = _mm256_extracti128_si256(b, 0);
    b_lo = _mm256_cvtepu8_epi16(tmp);
    tmp = _mm256_extracti128_si256(b, 1);
    b_hi = _mm256_cvtepu8_epi16(tmp);

    const __m256i c7f = _mm256_set1_epi8((short)127);
    const __m256i c7f2 = _mm256_set1_epi16((short)127);
    __m256i v1, v2, w1, w2;
    a_lo = _mm256_mullo_epi16(a_lo, b_lo);
    a_hi = _mm256_mullo_epi16(a_hi, b_hi);
    v1 = _mm256_srai_epi16(a_lo, 7);
    v2 = _mm256_srai_epi16(a_hi, 7);
    w1 = _mm256_packs_epi16(a_lo&c7f2, a_hi&c7f2);
    w2 = _mm256_packs_epi16(v1, v2);

    v1 = _mm256_add_epi8(w1, w2);
    v2 = _mm256_permute4x64_epi64(v1, 0b11011000);
    w1 = _mm256_sub_epi8(v2, c7f);
    w2 = _mm256_blendv_epi8(w1, v2, w1);

    return w2;
    //barrett_mul_u16(a_lo, a_lo, b_lo, t);
    //barrett_mul_u16(a_hi, a_hi, b_hi, t);

    //a_lo = _mm256_shuffle_epi8(a_lo, shuffle);
    //a_hi = _mm256_shuffle_epi8(a_hi, shuffle);

    //a_lo = _mm256_permute4x64_epi64(a_lo, 0xd8);
    //a_hi = _mm256_permute4x64_epi64(a_hi, 0xd8);

    //t = _mm256_permute2x128_si256(a_lo, a_hi, 0x20);

    //barrett_red8(t, r, c8_127, c8_1);
    //return t;
}

/// NOTE assumes each FQ element is in a 16bit limb
static inline __m256i gf127v_mul_u256_v2(const __m256i a1, const __m256i a2,
                                         const __m256i b1, const __m256i b2) {
    __m256i v1, v2, w1, w2;
    const __m256i c7f = _mm256_set1_epi8((short)127);
    const __m256i c7f2 = _mm256_set1_epi16((short)127);

    __m256i a_lo = a1;
    __m256i a_hi = a2;
    __m256i b_lo = b1;
    __m256i b_hi = b2;

    a_lo = _mm256_mullo_epi16(a_lo, b_lo);
    a_hi = _mm256_mullo_epi16(a_hi, b_hi);
    v1 = _mm256_srai_epi16(a_lo, 7);
    v2 = _mm256_srai_epi16(a_hi, 7);
    w1 = _mm256_packs_epi16(a_lo&c7f2, a_hi&c7f2);
    w2 = _mm256_packs_epi16(v1, v2);

    v1 = _mm256_add_epi8(w1, w2);
    v2 = _mm256_permute4x64_epi64(v1, 0b11011000);
    w1 = _mm256_sub_epi8(v2, c7f);
    w2 = _mm256_blendv_epi8(w1, v2, w1);

    return w2;
}
#endif


#ifdef USE_AVX512
/// TODO optimize
static inline void gf127v_scalar_u512_compute_table(__m512i *ret,
                                                    const gf127 a) {
#if 1
    ret[0] = _mm512_load_si512((const __m512i *)(__gf127_lookuptable + 128 * a +  0));
    ret[1] = _mm512_load_si512((const __m512i *)(__gf127_lookuptable + 128 * a + 64));
#else
    for (uint32_t i = 0; i < 2; i++) {
        gf127 table[64] = {0};
        for (uint32_t j = 0; j < 64-i; j++) {
            table[j] = gf127_mul(a, i*64 + j); 
        }

        ret[i] = _mm512_loadu_si512((const __m512i *)table);
    }
#endif
}

__m512i gf127v_scalar_table_u512(const __m512i a,
                                 const __m512i table1,
                                 const __m512i table2) {
    const __m512i mask1 = _mm512_set1_epi8(64);

    const __mmask64 m1 = _mm512_cmplt_epi8_mask(a, mask1);
    const __mmask64 m2 = ~m1;
    const __m512i t1 = _mm512_maskz_permutexvar_epi8(m1, a, table1);
    const __m512i t2 = _mm512_maskz_permutexvar_epi8(m2, a, table2);
    const __m512i t = t1 ^ t2;
    return t;
}
#endif

#ifdef USE_M4

#include <arm_acle.h>

/// \return a-b*c
inline static uint8x8_t fq_scalar_sub_u32(const uint8x8_t a,
                                          const uint8x8_t b,
                                          const uint8_t c) {
    const uint8x8_t c1 = vdup_n_u8(c);
    const uint8x8_t q = vdup_n_u8(0x7f);

    const uint16x8_t t1 = vmull_u8(b, c1);
    const uint16x8_t t2 = vshrq_n_u16(t1, 7);
    const uint16x8_t t3 = vaddq_u16(t1, t2);
    const uint8x8_t t4 = vmovn_u16(t3);
    const uint8x8_t t5 = vsub_u8(t4, q);

    const uint8x8_t mask = vshr_n_s8(t5, 7);
    const uint8x8_t ret = vbsl_u8(mask, t4, t5)
    return ret;
}
#endif

#ifdef USE_NEON
#include <arm_neon.h>
uint8x16_t gf127v_mul_u128(const uint8x16_t a, 
                           const uint8x16_t b) {
}
#endif
