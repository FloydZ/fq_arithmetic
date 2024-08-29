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
    ret ^= gf256_mul(a>>8, b);
    return ret;
}

/// \return a^-1
static inline gf256to2 gf256to2_inverse(const gf256to2 a) {
    // we are using the lagrange methode: a^-1 = a^{2^{16} - 2}
    // which can be computed as:
    //  t = 2^8
    //  x = a^{2^8 + 1} \in F_256
    //  y = x^-1
    //  return y * t
    gf256to2 t = a;
    for (uint32_t i = 0; i < 8; ++i) {
        t = gf256_mul(t, t);
    }
    const gf256to2 x = gf256_mul(t, a);
    const gf256to2 y = gf256_inverse(x);
    const gf256to2 ret = gf256_mul(y, t);
    return ret;
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
    uint64_t b_ = ((uint64_t)b) -1ull;
    return a^b;
}

/// \param a
/// \param b
/// \return a*b but 8 values in parallel
static inline uint64_t gf256to2v_mul_u64(const uint64_t a,
                           const uint64_t b) {
    const uint64_t ab = gf256v_mul_u64(a, b);
    const uint64_t sa = a ^ (a >> 8);
    const uint64_t sb = b ^ (b >> 8);
    uint64_t ret = ab ^ gf256v_mul_u64(ab, 0x20);
    ret ^= (gf256v_mul_u64(sa, sb) ^ ab) << 8;
    return ret;
}

#ifdef USE_AVX2
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


static inline __m256i gf256_multab(uint8_t b) {
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
static inline __m256i gf256to2_multab32() {
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


/// out = in1 ^ scalar*in2
static inline void gf256to2_vector_add_scalar_u256(gf256to2 *out,
                                                    const gf256to2 *in1,
                                                    const gf256to2 scalar,
                                                    const gf256to2 *in2,
                                                    const size_t bytes) {
    const __m256i mask = _mm256_set1_epi8(0xf);
    const __m256i maskff = _mm256_set1_epi16(0xff);

    // * 32 lookup table
    const __m256i t32 = gf256to2_multab32();
    const __m256i t32l = _mm256_permute2x128_si256(t32, t32, 0);
    const __m256i t32h = _mm256_permute2x128_si256(t32, t32, 0x11);

    const uint8_t bs = scalar ^ (scalar >> 8);
    const __m256i bs0 = gf256_multab(bs);
    const __m256i tbsl = _mm256_permute2x128_si256(bs0, bs0, 0);
    const __m256i tbsh = _mm256_permute2x128_si256(bs0, bs0, 0x11);
    const __m256i b = _mm256_set1_epi16(scalar);

    uint32_t i = bytes;
    while (i >= 32u) {
        __m256i a = _mm256_loadu_si256((__m256i *)in2);

        // NOTE: extremely expensive
        const __m256i ab = gf256v_mul_u256(a, b);
        const __m256i as = (a ^ (_mm256_srli_epi16(a, 8)));

        // low
        __m256i tmp = gf256to2_lookup(t32l, t32h, _mm256_srli_epi16(ab, 8), mask);
        __m256i ret = (ab & maskff) ^ tmp;

        // high
        tmp = gf256to2_lookup(tbsl, tbsh, as, mask);
        ret ^= (_mm256_slli_epi16(tmp, 8) ^ _mm256_slli_epi16(ab, 8));

        _mm256_storeu_si256((__m256i *)out, _mm256_loadu_si256((__m256i *)in1) ^ ret);
        i   -= 32u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    uint16_t tmp[16];
    for (uint32_t j = 0; j < (i + 1)/2; ++j) {
        tmp[j] = in2[j];
    }

    __m256i a = _mm256_loadu_si256((__m256i *)tmp);
    const __m256i ab = gf256v_mul_u256(a, b);
    const __m256i as = (a ^ (_mm256_srli_epi16(a, 8)));

    // low
    __m256i ttt = gf256to2_lookup(t32l, t32h, _mm256_srli_epi16(ab, 8), mask);
    __m256i ret = (ab & maskff) ^ ttt;

    // high
    ttt = gf256to2_lookup(tbsl, tbsh, as, mask);
    ret ^= (_mm256_slli_epi16(ttt, 8) ^ _mm256_slli_epi16(ab, 8));
    _mm256_storeu_si256((__m256i *)tmp, ret);

    for (uint32_t j = 0; j < (i + 1) / 2; ++j) {
        out[j] = in1[j] ^ tmp[j];
    }
}
#endif // USE_AVX2

#ifdef __AVX512F__
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

/// out = in1 ^ in2
static inline void gf256to2_vector_add_u512(gf256 *out,
                                            const gf256 *in1,
                                            const gf256 *in2,
                                            const size_t bytes) {
    size_t i = bytes;

    // avx512 code
    while (i >= 64u) {
        _mm512_storeu_si512((__m512i *)out,
                            _mm512_loadu_si512((__m512i *)in1) ^
                            _mm512_loadu_si512((__m512i *)in2));
        i   -= 64u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)out,
                            _mm256_loadu_si256((__m256i *)in1) ^
                            _mm256_loadu_si256((__m256i *)in2));
        i   -= 32u;
        in1 += 16u;
        in2 += 16u;
        out += 16u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)out,
                         _mm_loadu_si128((__m128i *)in1) ^
                         _mm_loadu_si128((__m128i *)in2));
        i   -= 16u;
        in1 += 8u;
        in2 += 8u;
        out += 8u;
    }

    for(uint32_t j = 0; j<((i+1)/2); j++) {
        out[j] = in1[j] ^ in2[j];
    }
}


/// out = in1 ^ scalar*in2
static inline void gf256to2_vector_add_scalar_u512(gf256to2 *out,
                                            const gf256to2 *in1,
                                            const gf256to2 scalar,
                                            const gf256to2 *in2,
                                            const size_t bytes) {
    const __m512i b = _mm512_set1_epi16(scalar);
    size_t i = bytes;
    while (i >= 64u) {
        __m512i tmp = _mm512_loadu_si512((__m512i *)in2);
        tmp = gf256to2v_mul_u512(tmp, b);
        _mm512_storeu_si512((__m512i *)out, _mm512_loadu_si512((__m512i *)in1) ^ tmp);
        i   -= 64u;
        in1 += 32u;
        in2 += 32u;
        out += 32u;
    }

    uint16_t tmp[32] __attribute__ ((aligned(64)));
    for (uint32_t j = 0; j < (i + 1)/2; ++j) {
        tmp[j] = in2[j];
    }

    __m512i t = _mm512_load_si512((__m512i *)tmp);
    t = gf256to2v_mul_u512(t, b);
    _mm512_store_si512((__m512i *)tmp, t);

    for (uint32_t j = 0; j < (i + 1) / 2; ++j) {
        out[j] = in1[j] ^ tmp[j];
    }
}
#endif

#endif
