#pragma once

#include <stdint.h>


#define PRIME 127
// Select low 8-bit, skip the high 8-bit in 16 bit type
const uint8_t shuff_low_half[32] __attribute__((aligned(32))) = {
        0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x0, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe,
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};

typedef uint8_t ff_t;

ff_t gf127_add(ff_t a, ff_t b) {
    return (a+b) % PRIME;
}

ff_t gf127_sub(ff_t a, ff_t b) {
    return (a+PRIME-b) % PRIME;
}

ff_t gf127_mul(ff_t a, ff_t b) {
    return (a*b)%PRIME;
}

ff_t gf127_neg(ff_t a) {
    return (PRIME - a) % PRIME;
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
__m256i gf31v_add_u256(const __m256i a,
                       const __m256i b) {
   const __m256i c7f = _mm256_set1_epi8(127);
   const __m256i c01 = _mm256_set1_epi8(1);
    __m256i t;
    __m256i c = _mm256_add_epi8(a, b);
    W_RED127(c);
    return c;
}

/// NOTE: assumes that each gf31 element is in a single uint8_t
__m256i gf31v_mul_u256(const __m256i a,
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

    barrett_mul_u16(a_lo, a_lo, b_lo, t);
    barrett_mul_u16(a_hi, a_hi, b_hi, t);

    a_lo = _mm256_shuffle_epi8(a_lo, shuffle);
    a_hi = _mm256_shuffle_epi8(a_hi, shuffle);

    a_lo = _mm256_permute4x64_epi64(a_lo, 0xd8);
    a_hi = _mm256_permute4x64_epi64(a_hi, 0xd8);

    t = _mm256_permute2x128_si256(a_lo, a_hi, 0x20);

    barrett_red8(t, r, c8_127, c8_1);
    return t;
}
#endif

