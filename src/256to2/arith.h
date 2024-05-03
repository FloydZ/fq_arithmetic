#ifndef FQ_ARITHMETIC_256TO2_ARITH_H
#define FQ_ARITHMETIC_256TO2_ARITH_H

// we represent the finite field as the extension field
// of degree 2 over the AES base field

/// normal extension field multiplication
/// \param a extension field element
/// \param b extension field element
/// \return a*b mod 256**2
uint16_t gf256to2_mul(const uint16_t a, const uint16_t b) {
    const uint8_t leading = gf256_mul(a>>8, b>>8);
    const uint8_t cnst  = gf256_mul(a, b);
    const uint8_t sum_a = gf256_add(a, a>>8);
    const uint8_t sum_b = gf256_add(b, b>>8);
    uint16_t ret = gf256_add(cnst, gf256_mul(leading, 0x20));
    ret ^= gf256_add(gf256_mul(sum_a,sum_b), cnst) << 8;
    return ret;
}

/// 4 parallel multiplications
/// \param a
/// \param b
/// \return
__m256i gf256to2_mul_u256(const __m256i a, const __m256i b) {
    const __m256i ab = _mm256_gf2p8mul_epi8(a, b);

}

/// TODO not finished
/// \param a
/// \param b
/// \return
uint64_t gf256to2_mul_u64(const uint64_t a, const uint64_t b) {
    const uint64_t ab = gf256v_mul_u64_v3(a, b);
    const uint64_t sa = a ^ (a >> 8);
    const uint64_t sb = b ^ (b >> 8);
    uint64_t ret = ab ^ gf256v_mul_u64_v3(ab, 0x20); // TODO
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


#endif //FQ_ARITHMETIC_ARITH_H
