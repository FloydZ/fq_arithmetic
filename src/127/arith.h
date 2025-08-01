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

/// Addition in GF(127)
/// \param a[in]: first addend 
/// \param b[in]: second addend
/// \return a+b % 127
static inline
gf127 gf127_add(const gf127 a,
                const gf127 b) {
    return (a+b) % PRIME;
}

/// Subtraction in GF(127)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b % 127
static inline
gf127 gf127_sub(const gf127 a,
                const gf127 b) {
    return (a+PRIME-b) % PRIME;
}

/// Multiplication in GF(127)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b % 127
static inline
gf127 gf127_mul(const gf127 a,
                const gf127 b) {
    return (a*b)%PRIME;
}

/// Negation in GF(127)
/// \param a[in]: value to negate
/// \return 0-a % 127
static inline
gf127 gf127_neg(const gf127 a) {
    return (PRIME - a) % PRIME;
}

/// Vectorized reduction of four GF(127) elements packed in a 32-bit value
/// \param Z[in]: 32-bit value containing four elements to reduce
inline static uint32_t gf127v_red_u32(uint32_t Z) {
    const uint32_t mask = 0x7F7F7F7F;
    const uint32_t one = 0x01010101;
    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    uint32_t C  = ((Z+one) & ~mask) ;
    return Z + (C>>PRIME_LOG_CEIL) - C ;
}

/// Vectorized reduction of eight GF(127) elements packed in a 64-bit value
/// \param Z[in]: 64-bit value containing eight elements to reduce
inline static uint64_t gf127v_red_u64(uint64_t Z) {
    const uint64_t mask = 0x7F7F7F7F7F7F7F7F;
    const uint64_t one  = 0x0101010101010101;
    Z = (Z & mask) + ((Z & ~mask) >> PRIME_LOG_CEIL);
    uint64_t C  = ((Z+one) & ~mask) ;
    return Z + (C>>PRIME_LOG_CEIL) - C ;
}

/// Vectorized addition of four GF(127) elements packed in 32-bit integers
/// \param a[in]: first vector of four GF(127) elements
/// \param b[in]: second vector of four GF(127) elements
/// \return vector of a+b in each byte position
inline static uint32_t gf127v_add_u32(const uint32_t a,
                                      const uint32_t b) {
    return gf127v_red_u32(a + b);
}

/// Vectorized addition of eight GF(127) elements packed in 64-bit integers
/// \param a[in]: first vector of eight GF(127) elements
/// \param b[in]: second vector of eight GF(127) elements
/// \return vector of a+b in each byte position
inline static uint64_t gf127v_add_u64(const uint64_t a,
                                      const uint64_t b) {
    return gf127v_red_u64(a + b);
}

/// Vectorized subtraction of four GF(127) elements packed in 32-bit integers
/// \param a[in]: vector of four GF(127) minuends
/// \param b[in]: vector of four GF(127) subtrahends
/// \return vector of a-b in each byte position
inline static uint32_t gf127v_sub_u32(const uint32_t a,
                                      const uint32_t b) {
    return gf127v_red_u32(a - b + 0x7F7F7F7F);
}

/// Vectorized subtraction of eight GF(127) elements packed in 64-bit integers
/// \param a[in]: vector of eight GF(127) minuends
/// \param b[in]: vector of eight GF(127) subtrahends
/// \return vector of a-b in each byte position
inline static uint64_t gf127v_sub_u64(const uint64_t a,
                                      const uint64_t b) {
    return gf127v_red_u64(a - b + 0x7F7F7F7F7F7F7F7F);
}

/// Vectorized scalar multiplication of four GF(127) elements by a single value
/// \param a[in]: vector of four GF(127) elements
/// \param b[in]: scalar GF(127) element to multiply by
/// \return vector of a*b in each byte position
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

/// Vectorized scalar multiply-subtract operation used in RREF (reduced row echelon form)
/// Computes a - b*c for four GF(127) elements
/// \param a[in]: vector of four GF(127) elements
/// \param b[in]: vector of four GF(127) elements to be multiplied
/// \param c[in]: scalar GF(127) element
/// \return vector of a-b*c in each byte position
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


/// AVX2 vectorized reduction of 32 GF(127) elements in a 256-bit register
/// \param a[in]: vector of 32 elements to reduce
/// \return vector of reduced elements
/// NOTE: the final `min` instruction can be replaced with `blendv`
__m256i gf127v_red_u256(const __m256i a) {
    const __m256i c7f = _mm256_set1_epi8(127);
    const __m256i t1 = _mm256_sub_epi8(a, c7f);
    const __m256i t2 = _mm256_min_epu8(t1, a);
    return t2;
}
                                      
/// AVX2 vectorized addition of 32 GF(127) elements in 256-bit registers
/// \param a[in]: first vector of 32 GF(127) elements
/// \param b[in]: second vector of 32 GF(127) elements
/// \return vector of a+b in each byte position
/// NOTE: assumes that each gf127 element is in a single uint8_t
/// vpaddb  ymm0, ymm1, ymm0; vpaddb  ymm1, ymm0, ymmword ptr [rax]; vpminub ymm0, ymm1, ymm0
/// sudo ./nanoBench.sh -asm_init "MOV RAX, R14; SUB RAX, 8; MOV [RAX], 129" -asm "vpaddb  ymm0, ymm1, ymm0; vpaddb  ymm1, ymm0, ymmword ptr [rax]; vpminub ymm0, ymm1, ymm0" -config configs/cfg_AlderLakeE_common.txt
/// sudo ./nanoBench.sh -asm_init "MOV RAX, R14; SUB RAX, 8; MOV [RAX], rax" -asm "vpaddb  ymm0, ymm1, ymm0; vpaddb  ymm1, ymm0, ymmword ptr [rax]; vpblendvb ymm0, ymm1, ymm0, ymm1;" -config configs/cfg_AlderLakeE_common.txt
__m256i gf127v_add_u256(const __m256i a,
                        const __m256i b) {
    const __m256i c7f = _mm256_set1_epi8(127);
    const __m256i t1 = _mm256_add_epi8(a, b);
    const __m256i t2 = _mm256_sub_epi8(t1, c7f);
    const __m256i t3 = _mm256_min_epu8(t2, t1);
    return t3;
}

/// AVX2 vectorized addition of 32 GF(127) elements using blendv instead of min
/// \param a[in]: first vector of 32 GF(127) elements
/// \param b[in]: second vector of 32 GF(127) elements
/// \return vector of a+b in each byte position
__m256i gf127v_add_u256_v2(const __m256i a,
                        const __m256i b) {
    const __m256i c7f = _mm256_set1_epi8(127);
    const __m256i t1 = _mm256_add_epi8(a, b);
    const __m256i t2 = _mm256_sub_epi8(t1, c7f);
    const __m256i t3 = _mm256_blendv_epi8(t2, t1, t2);
    return t3;
}

/// AVX2 vectorized multiplication of 32 GF(127) elements 
/// \param a[in]: first vector of 32 GF(127) elements
/// \param b[in]: second vector of 32 GF(127) elements
/// \return vector of a*b in each byte position [a_0*b_0, a_1*b_1, ..., a_31*b_31]
/// NOTE: assumes that each gf127 element is in a single uint8_t
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
    // NOTE: both work
    // w2 = _mm256_min_epu8(w1, v2);

    return w2;
}

/// Alternative AVX2 vectorized multiplication of GF(127) elements stored in 16-bit limbs
/// \param a1[in]: first half of first vector of GF(127) elements
/// \param a2[in]: second half of first vector of GF(127) elements
/// \param b1[in]: first half of second vector of GF(127) elements
/// \param b2[in]: second half of second vector of GF(127) elements
/// \return vector of a*b in each byte position
/// NOTE: assumes each FQ element is in a 16bit limb
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
/// Precompute lookup table for AVX512 vectorized scalar multiplication
/// \param ret[out]: output array to store precomputed table
/// \param a[in]: scalar GF(127) element to create table for
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

/// AVX512 vectorized scalar multiplication using precomputed lookup tables
/// \param a[in]: vector of 64 GF(127) elements
/// \param table1[in]: first half of precomputed lookup table
/// \param table2[in]: second half of precomputed lookup table
/// \return vector of scalar products
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

/// AVX512 vectorized addition of 64 GF(127) elements
/// \param a[in]: first vector of 64 GF(127) elements
/// \param b[in]: second vector of 64 GF(127) elements
/// \return vector of a+b in each byte position
__m512i gf127v_add_u512(const __m512i a,
                        const __m512i b) {
    const __m512i c7f = _mm512_set1_epi8(127);
    const __m512i t1 = _mm512_add_epi8(a, b);
    const __m512i t2 = _mm512_sub_epi8(t1, c7f);
    const __m512i t3 = _mm512_min_epu8(t2, t1);
    return t3;
}

/// AVX512 vectorized multiplication of 32 GF(127) elements in 16-bit limbs
/// \param aa[in]: vector of 32 GF(127) elements, each aa[i] < 127 in 16-bit limb
/// \param bb[in]: vector of 32 GF(127) elements, each bb[i] < 127 in 16-bit limb
/// \return vector of aa[i] * bb[i] for all i in range(32)
static inline __m256i gf127v_mul_u512(const __m512i aa,
                                      const __m512i bb) {

    __m512i c01, c7f, c516, tmp;
    vset16_512(c01, 0x01);
    vset16_512(c7f, 0x7F);
    vset16_512(c516, 516);

    __m512i acc = _mm512_mullo_epi16(aa, bb);
    tmp = _mm512_add_epi16(acc, c01);
    tmp = _mm512_mulhi_epu16(tmp, c516);
    tmp = _mm512_mullo_epi16(tmp, c7f);
    acc = _mm512_sub_epi16(acc, tmp);
    const __m256i t = _mm512_cvtepi16_epi8(acc);
    return t;
}
#endif

#ifdef USE_NEON

#include <arm_neon.h>
#include <arm_acle.h>

/// NEON vectorized scalar multiply-subtract for 8 GF(127) elements
/// \param a[in]: vector of 8 GF(127) elements
/// \param b[in]: vector of 8 GF(127) elements to be multiplied
/// \param c[in]: scalar GF(127) element
/// \return vector of a-b*c in each byte position
inline static uint8x8_t gf127v_scalar_sub_u64(const uint8x8_t a,
                                              const uint8x8_t b,
                                              const uint8_t c) {
    const uint8x8_t c1  = vdup_n_u8(c);
    const uint8x8_t q   = vdup_n_u8(0x7f);
    const uint16x8_t q2 = vdupq_n_u16(0x7f);

    const uint16x8_t t1 = vmull_u8(b, c1);
    const uint16x8_t t2 = vshrq_n_u16(t1, 7);
    const uint16x8_t t3 = vaddq_u16(t2, vandq_u16(t1, q2));
    const uint8x8_t t4  = vmovn_u16(t3);
    const uint8x8_t t5  = vsub_u8(a, t4);
    const uint8x8_t t6  = vadd_u8(t5, q);
    const uint8x8_t mask2 = vshr_n_s8(t5, 7);
    const uint8x8_t ret  = vbsl_u8(mask2, t6, t5);
    return ret;
}

/// NEON vectorized multiplication of 16 GF(127) elements
/// \param a[in]: first vector of 16 GF(127) elements
/// \param b[in]: second vector of 16 GF(127) elements
/// \return vector of a*b in each byte position
uint8x16_t gf127v_mul_u128(const uint8x16_t a,
                           const uint8x16_t b) {
    const uint8x16_t q  = vdupq_n_u8(0x7f);
    const uint16x8_t q2 = vdupq_n_u16(0x007f);

    const uint8x8_t la = vget_low_u8(a);
    const uint8x8_t lb = vget_low_u8(b);
    const uint8x8_t ha = vget_high_u8(a);
    const uint8x8_t hb = vget_high_u8(b);

    const uint16x8_t lc = vmull_u8(la, lb);
    const uint16x8_t hc = vmull_u8(ha, hb);

    const uint16x8_t lt1 = vshrq_n_u16(lc, 7);
    const uint16x8_t ht1 = vshrq_n_u16(hc, 7);
    const uint16x8_t lt2 = vaddq_u16(lt1, vandq_u16(lc, q2));
    const uint16x8_t ht2 = vaddq_u16(ht1, vandq_u16(hc, q2));
    const uint8x16_t c   = veorq_u16(lt2, vshlq_n_u16(ht2, 8));

    const uint8x16_t t   = vsubq_u8(c, q);
    const uint8x16_t m   = vshrq_n_s8(c, 7);
    const uint8x16_t r   = vbslq_u8(m, t, c);
    return r;
}
#endif


#ifdef USE_M4
#include <arm_acle.h>

/// ARM M4 vectorized scalar multiply-subtract for 4 GF(127) elements
/// \param a[in]: vector of 4 GF(127) elements
/// \param b[in]: vector of 4 GF(127) elements to be multiplied
/// \param c[in]: scalar GF(127) element
/// \return vector of a-b*c in each byte position
inline static uint32_t fq_scalar_sub_u32(const uint32_t a,
                                         const uint32_t b,
                                         const uint8_t c) {
    const uint32_t mask  = 0x7F7F7F7F;
    const uint32_t mask1 = 0x00FF00FF;
    const uint32_t mask3 = 0x007F007F;

    const uint32_t t1 = (b&mask1) * c;
    const uint32_t t2 = ((b >> 8)&mask1) * c;

    const uint32_t v1 = ((t1 >> 7) + (t1 & mask3)) & mask1;
    const uint32_t v2 = ((t2 >> 7) + (t2 & mask3)) & mask1;

    uint32_t Z = v1 ^ (v2<<8);
    uint8x4_t t3 = __usub8(mask, Z);
    uint8x4_t t4 = __uadd8(a, t3);
    uint8x4_t t5 = __usub8(t4, mask);
    uint8x4_t t6 = __sel(t5, t4);
    return t6;
}

#endif
