#define PRIME 251

typedef uint8_t ff_t;

ff_t gf251_add(const ff_t a, const ff_t b) {
    return (a+b) % PRIME;
}

ff_t gf251_sub(const ff_t a, const ff_t b) {
    return (a+PRIME-b) % PRIME;
}

ff_t gf251_mul(const ff_t a, const ff_t b) {
    return (a*b)%PRIME;
}

ff_t gf251_neg(const ff_t a) {
    return (PRIME - a) % PRIME;
}


#ifdef USE_AVX2
#include <immintrin.h>

/// NOTE: assumes that each gf251 element is in a single uint8_t
/// TODO not finished
static inline
__m256i gf251v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8((int8_t)251);
    const __m256i mask = _mm256_set1_epi8(7u);
    __m256i t = _mm256_srli_epi16(a, 5);
    t &= mask;
    t &= Q;
    __m256i ret = _mm256_add_epi8(a, t);
    ret &= Q;
    return ret;
}

/// NOTE: assumes that each gf251 element is in a single uint8_t
/// TODO not finished
static inline
__m256i gf251v_add_u256(const __m256i a,
                       const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf251v_red_u256(c);
}

/// NOTE: assumes reduced inputs
/// TODO not finished
static inline
__m256i gf251v_mul_u256(const __m256i a,
                        const __m256i b) {
    __m256i tl, th, sl, sh, r;
    const __m256i one = _mm256_set1_epi16(1);

    const __m256i m = _mm256_set1_epi16(0xFF);
    const __m256i al = a & m;
    const __m256i ah = _mm256_srli_epi16(a, 8);
    const __m256i bl = b & m;
    const __m256i bh = _mm256_srli_epi16(b, 8);

    // actual multiplication
    const __m256i cl = _mm256_mullo_epi16(al, bl);
    const __m256i ch = _mm256_mullo_epi16(ah, bh);

    // reduction
    tl = _mm256_srli_epi16(cl, 5);
    th = _mm256_srli_epi16(ch, 5);
    sl = _mm256_add_epi16(tl, cl);
    sh = _mm256_add_epi16(th, ch);
    tl = _mm256_slli_epi16(tl, 5);
    th = _mm256_slli_epi16(th, 5);

    r  = _mm256_sub_epi16(sl, tl);
    sh = _mm256_sub_epi16(sh, th);
    sh = _mm256_slli_epi16(sh, 8);
    r = r ^ sh;
    return r;
}
#endif //FQ_ARITHMETIC_ARITH_H
