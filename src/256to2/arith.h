#ifndef FQ_ARITHMETIC_256TO2_ARITH_H
#define FQ_ARITHMETIC_256TO2_ARITH_H

#include "../256/arith.h"

// we represent the finite field as the extension field
// of degree 2 over the AES base field

// we represent each field element with two byte.
// GF256to2: uint16_t: X^2 + x + 32 = 0
typedef uint16_t gf256to2;


/// addition
static inline gf256to2 gf256to2_add(const gf256to2 a,
                      const gf256to2 b) {
    return a^b;
}

static inline gf256to2 gf256to2_sub(const gf256to2 a,
                      const gf256to2 b) {
    return a^b;
}

/// \return -a = a (characteristic 2)
static inline gf256to2 gf256to2_neg(const gf256to2 a) {
    return a;
}

/// addition
static inline gf256to2 gf256to2_add_gf256(const gf256to2 a,
                            const uint8_t b) {
    return a^b;
}
/// normal extension field multiplication
/// \param a extension field element
/// \param b extension field element
/// \return a*b mod 256**2
static inline gf256to2 gf256to2_mul(const gf256to2 a, const gf256to2 b) {
    const uint8_t h = gf256_mul(a>>8u, b>>8u);
    const uint8_t l = gf256_mul(a, b);
    const uint8_t sa = a ^ (a>>8);
    const uint8_t sb = b ^ (b>>8);
    gf256to2 ret = l ^ gf256_mul(h, 0x20);
    ret ^= (gf256_mul(sa, sb) ^ l) << 8;
    return ret;
}

/// \param a element in the extension field
/// \param b element in the base field
/// \return a*b
static inline gf256to2 gf256to2_mul_gf256(const gf256to2 a,
                            const gf256 b) {
    gf256to2 ret = gf256_mul(a, b);
    ret ^= ((gf256to2)gf256_mul(a>>8, b)) << 8;
    return ret;
}

/// inversion via the norm map: for a = a_1*X + a_0 the conjugate is
/// a_1*X + (a_0 + a_1) and N(a) = a_0*(a_0 + a_1) + 32*a_1^2 \in GF(256), so
///     a^-1 = conj(a) * N(a)^-1
/// NOTE: inv(0) = 0; not constant time, as `gf256_inv` is a table lookup
/// \return a^-1
static inline gf256to2 gf256to2_inv(const gf256to2 a) {
    const gf256 a0 = a, a1 = a >> 8u;
    const gf256 s = a0 ^ a1;
    const gf256 n = gf256_mul(a0, s) ^ gf256_mul(0x20, gf256_mul(a1, a1));
    const gf256 ninv = gf256_inv(n);
    return (gf256to2)(((gf256to2)gf256_mul(a1, ninv) << 8u) | gf256_mul(s, ninv));
}

/// addition
static inline uint64_t gf256to2v_add(const uint64_t a,
                       const uint64_t b) {
    return a^b;
}

static inline uint64_t gf256to2v_sub(const uint64_t a,
                       const uint64_t b) {
    return a^b;
}

static inline uint64_t gf256to2v_add_gf256(const uint64_t a,
                                           const uint8_t b) {
    return a^b;
}

/// \param a [a_0, ..., a_3]
/// \param b [b_0, ..., b_3]
/// \return [a_0*b_0, ..., a_3*b_3]: 4 values in parallel
static inline uint64_t gf256to2v_mul_u64(const uint64_t a,
                           const uint64_t b) {
    const uint64_t mask_low = 0x00FF00FF00FF00FFull;
    const uint64_t ab = gf256v_mul_u64(a, b);
    const uint64_t sa = a ^ (a >> 8);
    const uint64_t sb = b ^ (b >> 8);
    // lower limb: a_0*b_0 + 32*a_1*b_1
    uint64_t ret = (ab ^ gf256v_mul_u64(ab >> 8, 0x0020002000200020ull)) & mask_low;
    // upper limb: (a_0 + a_1)*(b_0 + b_1) + a_0*b_0
    ret ^= ((gf256v_mul_u64(sa, sb) ^ ab) << 8) & ~mask_low;
    return ret;
}

#ifdef USE_AVX2
/// \param a [a_0, ..., a_15]
/// \param b [b_0, ..., b_15]
/// \return [a_0+b_0, ..., a_15+b_15]
static inline __m256i gf256to2v_add_u256(const __m256i a, const __m256i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_15-b_15]
static inline __m256i gf256to2v_sub_u256(const __m256i a, const __m256i b) {
    return a ^ b;
}

/// \param a [a_0, ..., a_15]
/// \param b [b_0, ..., b_15]
/// \return [a_0*b_0, ..., a_15*b_15]
static inline __m256i gf256to2v_mul_u256(__m256i a, __m256i b) {
    __m256i c;
    const __m256i mask1 = _mm256_set1_epi16(0x00FF);
    const __m256i mask2 = _mm256_set1_epi16(0xFF00);
    const __m256i mulm  = _mm256_set1_epi16(0x0020);
    const __m256i ab = gf256v_mul_u256(a, b);
    const __m256i sa = _mm256_slli_epi16(a, 8) ^ a;
    const __m256i sb = _mm256_slli_epi16(b, 8) ^ b;

    // compute the higher
    c = (gf256v_mul_u256(sa, sb) ^ _mm256_slli_epi16(ab, 8)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256v_mul_u256(_mm256_srli_epi16(ab, 8), mulm)) & mask1;
    return c;
}

/// SSE version of `gf256to2v_mul_u256`
/// 3 gf(256) mul + 11 inst = 119 instr
/// \param a
/// \param b
/// \return
static inline __m128i gf256to2v_mul_u128(const __m128i a, const __m128i b) {
    __m128i c;
    const __m128i mask1 = _mm_set1_epi16(0x00FF);
    const __m128i mask2 = _mm_set1_epi16(0xFF00);
    const __m128i mulm  = _mm_set1_epi16(0x0020);
    const __m128i ab = gf256v_mul_u128(a, b);
    const __m128i sa = _mm_slli_epi16(a, 8) ^ a;
    const __m128i sb = _mm_slli_epi16(b, 8) ^ b;

    // compute the higher
    c = (gf256v_mul_u128(sa, sb) ^ _mm_slli_epi16(ab, 8)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256v_mul_u128 (_mm_srli_epi16(ab, 8), mulm)) & mask1;
    return c;
}

// 6 instructions
static inline __m256i gf256to2_lookup (
        __m256i tab_l, __m256i tab_h, __m256i v, __m256i mask_f){
    return _mm256_shuffle_epi8(tab_l, v & mask_f) ^
           _mm256_shuffle_epi8(tab_h, _mm256_srli_epi16(v, 4) & mask_f);
}


static inline __m256i gf256to2_gf256_multab(uint8_t b) {
    const __m256i bx = _mm256_set1_epi16(b);
    const __m256i b1 = _mm256_srli_epi16(bx, 1);

    const __m256i tab0 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*0)); // 1*i
    const __m256i tab1 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*1)); // 2*i
    const __m256i tab2 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*2)); // 4*i
    const __m256i tab3 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*3));
    const __m256i tab4 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*4));
    const __m256i tab5 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*5));
    const __m256i tab6 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*6));
    const __m256i tab7 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*7));

    const __m256i mask_1  = _mm256_set1_epi16(1);
    const __m256i mask_4  = _mm256_set1_epi16(4);
    const __m256i mask_16 = _mm256_set1_epi16(16);
    const __m256i mask_64 = _mm256_set1_epi16(64);
    const __m256i mask_0  = _mm256_setzero_si256();

    // []
    __m256i tmp;
    tmp  = (tab0 & _mm256_cmpgt_epi16(bx&mask_1 , mask_0));
    tmp ^= (tab1 & _mm256_cmpgt_epi16(b1&mask_1 , mask_0));
    tmp ^= (tab2 & _mm256_cmpgt_epi16(bx&mask_4 , mask_0));
    tmp ^= (tab3 & _mm256_cmpgt_epi16(b1&mask_4 , mask_0));
    tmp ^= (tab4 & _mm256_cmpgt_epi16(bx&mask_16, mask_0));
    tmp ^= (tab5 & _mm256_cmpgt_epi16(b1&mask_16, mask_0));
    tmp ^= (tab6 & _mm256_cmpgt_epi16(bx&mask_64, mask_0));
    tmp ^= (tab7 & _mm256_cmpgt_epi16(b1&mask_64, mask_0));
    return tmp;
}

// generate multiplication table
// outputs a register:
//          0                   1
//      [b_1*1 ^ b_2*2 ^ ...,  b_1*1*2 ^ b_2*2*2]
//  b_i= ith bit of b = [b_1, ..., b_16]
static inline __m256i gf256to2_multab(uint16_t b) {
    const __m256i bx = _mm256_set1_epi16(b);
    const __m256i b1 = _mm256_srli_epi16(bx, 1);

    const __m256i tab0 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*0)); // 1*i
    const __m256i tab1 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*1)); // 2*i
    const __m256i tab2 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*2)); // 4*i
    const __m256i tab3 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*3));
    const __m256i tab4 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*4));
    const __m256i tab5 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*5));
    const __m256i tab6 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*6));
    const __m256i tab7 = _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*7));

    const __m256i mask_1  = _mm256_set1_epi8(1);
    const __m256i mask_4  = _mm256_set1_epi8(4);
    const __m256i mask_16 = _mm256_set1_epi8(16);
    const __m256i mask_64 = _mm256_set1_epi8(64);
    const __m256i mask_0  = _mm256_setzero_si256();

    // []
    __m256i tmp;
    tmp  = (tab0 & _mm256_cmpgt_epi8(bx&mask_1 , mask_0));
    tmp ^= (tab1 & _mm256_cmpgt_epi8(b1&mask_1 , mask_0));
    tmp ^= (tab2 & _mm256_cmpgt_epi8(bx&mask_4 , mask_0));
    tmp ^= (tab3 & _mm256_cmpgt_epi8(b1&mask_4 , mask_0));
    tmp ^= (tab4 & _mm256_cmpgt_epi8(bx&mask_16, mask_0));
    tmp ^= (tab5 & _mm256_cmpgt_epi8(b1&mask_16, mask_0));
    tmp ^= (tab6 & _mm256_cmpgt_epi8(bx&mask_64, mask_0));
    tmp ^= (tab7 & _mm256_cmpgt_epi8(b1&mask_64, mask_0));
    return tmp;
}

/// generates the mult-table for *32
static inline __m256i gf256to2_multab32(void) {
    return _mm256_load_si256((__m256i const *) (__gf256_mulbase_avx2 + 32*5));
}

// computes a * 32
static inline __m256i gf256to2v_mul_32(const __m256i a) {
    const __m256i mask = _mm256_set1_epi8(0xf);
    const __m256i t = gf256to2_multab32();
    const __m256i tl = _mm256_permute2x128_si256(t, t, 0);
    const __m256i th = _mm256_permute2x128_si256(t, t, 0x11);
    return gf256to2_lookup(tl, th, a, mask);
}
#endif // USE_AVX2

#ifdef USE_NEON
/// \param a [a_0, ..., a_7]
/// \param b [b_0, ..., b_7]
/// \return [a_0+b_0, ..., a_7+b_7]
static inline uint16x8_t gf256to2v_add_u128(const uint16x8_t a, const uint16x8_t b) {
    return veorq_u16(a, b);
}

/// \return [a_0-b_0, ..., a_7-b_7]
static inline uint16x8_t gf256to2v_sub_u128(const uint16x8_t a, const uint16x8_t b) {
    return veorq_u16(a, b);
}

/// NEON version of `gf256to2v_mul_u256`
/// \param a [a_0, ..., a_7]
/// \param b [b_0, ..., b_7]
/// \return [a_0*b_0, ..., a_7*b_7]
static inline uint16x8_t gf256to2v_mul_u128(const uint16x8_t a, const uint16x8_t b) {
    const uint16x8_t mask1 = vdupq_n_u16(0x00FF);
    const uint16x8_t mask2 = vdupq_n_u16(0xFF00);
    const uint8x16_t mulm  = vreinterpretq_u8_u16(vdupq_n_u16(0x0020));
    const uint16x8_t ab = vreinterpretq_u16_u8(gf256v_mul_u128(vreinterpretq_u8_u16(a),
                                                               vreinterpretq_u8_u16(b)));
    const uint16x8_t sa = veorq_u16(vshlq_n_u16(a, 8), a);
    const uint16x8_t sb = veorq_u16(vshlq_n_u16(b, 8), b);

    // compute the higher limb
    const uint16x8_t h = vreinterpretq_u16_u8(gf256v_mul_u128(vreinterpretq_u8_u16(sa),
                                                              vreinterpretq_u8_u16(sb)));
    uint16x8_t c = vandq_u16(veorq_u16(h, vshlq_n_u16(ab, 8)), mask2);

    // compute the lower limb
    const uint16x8_t l = vreinterpretq_u16_u8(gf256v_mul_u128(vreinterpretq_u8_u16(vshrq_n_u16(ab, 8)), mulm));
    c = veorq_u16(c, vandq_u16(veorq_u16(ab, l), mask1));
    return c;
}
#endif // USE_NEON

#ifdef USE_AVX512
/// \return [a_0+b_0, ..., a_31+b_31]
static inline __m512i gf256to2v_add_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// \return [a_0-b_0, ..., a_31-b_31]
static inline __m512i gf256to2v_sub_u512(const __m512i a,
                                         const __m512i b) {
    return a ^ b;
}

/// full multiplication: every 16bit limb is multiplied
/// \param a input
/// \param b input
/// \return a*b
static inline __m512i gf256to2v_mul_u512(const __m512i a,
                                         const __m512i b) {
    __m512i c;
    const __m512i mask1 = _mm512_set1_epi16(0x00FF);
    const __m512i mask2 = _mm512_set1_epi16(0xFF00);
    const __m512i mulm  = _mm512_set1_epi16(0x0020);
    const __m512i ab = gf256v_mul_u512(a, b);
    const __m512i sa = _mm512_slli_epi16(a, 8) ^ a;
    const __m512i sb = _mm512_slli_epi16(b, 8) ^ b;

    // compute the higher
    c = (gf256v_mul_u512(sa, sb) ^ _mm512_slli_epi16(ab, 8)) & mask2;

    // compute the lower limb:
    c ^= (ab ^ gf256v_mul_u512(_mm512_srli_epi16(ab, 8), mulm)) & mask1;
    return c;
}
#endif

#endif
