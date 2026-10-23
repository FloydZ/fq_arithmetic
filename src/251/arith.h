#define PRIME 251

typedef uint8_t ff_t;

///
/// @param a
/// @param b
/// @return
static inline
ff_t gf251_add(const ff_t a, const ff_t b) {
    return (a+b) % PRIME;
}

///
/// @param a
/// @param b
/// @return
static inline
ff_t gf251_sub(const ff_t a, const ff_t b) {
    return (a+PRIME-b) % PRIME;
}

///
/// @param a
/// @param b
/// @return
static inline
ff_t gf251_mul(const ff_t a, const ff_t b) {
    return (a*b)%PRIME;
}

///
/// @param a
/// @return
static inline
ff_t gf251_neg(const ff_t a) {
    return (PRIME - a) % PRIME;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// reduces every byte from [0, 256) to [0, 251)
/// NOTE: assumes that each gf251 element is in a single uint8_t
static inline
__m256i gf251v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    const __m256i t = _mm256_sub_epi8(a, Q);
    return _mm256_min_epu8(a, t);
}

/// NOTE: assumes that each gf251 element is in a single uint8_t, and reduced
static inline
__m256i gf251v_add_u256(const __m256i a,
                        const __m256i b) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    // a + b >= 251 <=> a >= 251 - b
    const __m256i d = _mm256_sub_epi8(Q, b);
    const __m256i m = _mm256_cmpeq_epi8(_mm256_max_epu8(a, d), a);
    return _mm256_blendv_epi8(_mm256_add_epi8(a, b), _mm256_sub_epi8(a, d), m);
}

/// \param c[in]: [c_0, ..., c_15], c_i < 251^2
/// \return [c_0 mod 251, ..., c_15 mod 251]
static inline
__m256i gf251v_red_u16_u256(const __m256i c) {
    const __m256i Q = _mm256_set1_epi16(251);
    // ceil(2^23 / 251), exact for c < 133152
    const __m256i M = _mm256_set1_epi16((short)33421);
    __m256i t = _mm256_mulhi_epu16(c, M);
    t = _mm256_srli_epi16(t, 7);
    t = _mm256_mullo_epi16(t, Q);
    return _mm256_sub_epi16(c, t);
}

/// NOTE: assumes that each gf251 element is in a single uint8_t, and reduced
static inline
__m256i gf251v_mul_u256(const __m256i a,
                        const __m256i b) {
    const __m256i m = _mm256_set1_epi16(0xFF);
    const __m256i al = a & m;
    const __m256i ah = _mm256_srli_epi16(a, 8);
    const __m256i bl = b & m;
    const __m256i bh = _mm256_srli_epi16(b, 8);

    // actual multiplication
    const __m256i cl = _mm256_mullo_epi16(al, bl);
    const __m256i ch = _mm256_mullo_epi16(ah, bh);

    // reduction
    const __m256i rl = gf251v_red_u16_u256(cl);
    const __m256i rh = gf251v_red_u16_u256(ch);
    return rl ^ _mm256_slli_epi16(rh, 8);
}
#endif //FQ_ARITHMETIC_ARITH_H
