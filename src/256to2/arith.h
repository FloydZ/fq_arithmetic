#ifndef FQ_ARITHMETIC_256TO2_ARITH_H
#define FQ_ARITHMETIC_256TO2_ARITH_H

#include "../256/arith.h"

// we represent the finite field as the extension field
// of degree 2 over the AES base field

/// normal extension field multiplication
/// \param a extension field element
/// \param b extension field element
/// \return a*b mod 256**2
uint16_t gf256to2_mul(const uint16_t a, const uint16_t b) {
    const uint8_t leading = gf256_mul(a>>8, b>>8);
    const uint8_t cnst  = gf256_mul(a, b);
    const uint8_t sum_a = a ^ (a>>8);
    const uint8_t sum_b = b ^ (b>>8);
    uint16_t ret = gf256_add(cnst, gf256_mul(leading, 0x20));
    //uint16_t ret = gf256_add(cnst, leading << 5u);
    ret ^= gf256_add(gf256_mul(sum_a,sum_b), cnst) << 8;
    return ret;
}

/// TODO not finished
/// \param a
/// \param b
/// \return
uint64_t gf256to2_mul_u64(const uint64_t a, const uint64_t b) {
    const uint64_t ab = gf256v_mul_u64_v3(a, b);
    const uint64_t sa = a ^ (a >> 8);
    const uint64_t sb = b ^ (b >> 8);
    uint64_t ret = ab ^ gf256v_mul_u64_v3(ab, 0x20); // TODO not coorect
    ret ^= (gf256v_mul_u64_v3(sa, sb) ^ ab) << 8;
    return ret;
}

/// \param a element in the extension field
/// \param b element in the base field
/// \return a*b
uint16_t gf256to2_mul_gf256(const uint16_t a, const uint8_t b) {
    uint16_t ret = gf256_mul(a, b);
    ret ^= gf256_mul(a>>8, b);
    return ret;
}

/// addition
uint16_t gf256to2_add(const uint16_t a, const uint16_t b) {
    return a^b;
}

/// addition
uint16_t gf256to2_add_gf256(const uint16_t a, const uint8_t b) {
    return a^b;
}

#ifdef USE_AVX2
__m256i gf256to2v_mul_u256(__m256i a, __m256i b) {
    __m256i c;
    const __m256i mask1 = _mm256_set1_epi16(0x00FF);
    const __m256i mask2 = _mm256_set1_epi16(0xFF00);
    const __m256i mulm  = _mm256_set1_epi16(0x0020);
    const __m256i ab = gf256v_mul_u256_v3(a, b);
    const __m256i sa = _mm256_slli_epi16(a, 8) ^ a;
    const __m256i sb = _mm256_slli_epi16(b, 8) ^ b;

    // compute the higher limgf256to2_mul(i, j)b
    c = (gf256v_mul_u256_v3(sa, sb) ^ _mm256_slli_epi16(ab, 8)) & mask2;

    // compute the lower limb:
    // c ^= (ab ^ _mm256_slli_epi16(_mm256_srli_epi16(ab, 8), 5)) & mask1;
    // TODO do not understand why Y + x + 32 = 0
    c ^= (ab ^ gf256v_mul_u256_v3(_mm256_srli_epi16(ab, 8), mulm)) & mask1;
    return c;
}
#endif

#endif //FQ_ARITHMETIC_ARITH_H
