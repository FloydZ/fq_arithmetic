#include <stdint.h>

#define PRIME 31

// Work as soon as x <= 63487 = 2^15.95
#define _gf31_reduce16(x) ((x) - 31 * (((uint32_t)(x) * 67651) >> 21))

// Work as soon as x <= 4160749567 = 2^31.95
#define _gf31_reduce32(x) ((x) - 31 * (((uint64_t)(x) * 4433514629) >> 37))


typedef uint8_t ff_t;

static inline
ff_t gf31_add(ff_t a, ff_t b) {
    return (a+b) % PRIME;
}

static inline
ff_t gf31_sub(ff_t a, ff_t b) {
    return (a+PRIME-b) % PRIME;
}

static inline
ff_t gf31_mul(ff_t a, ff_t b) {
    return (a*b)%PRIME;
}

static inline
ff_t gf31_neg(ff_t a) {
    return (PRIME - a) % PRIME;
}


#ifdef USE_AVX2 
#include <immintrin.h>

/// NOTE: assumes that each gf31 element is in a single uint8_t
/// NOTE: not really a full reduction, as 31 is not going to be reduced
static inline
__m256i gf31v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8(31);
    const __m256i mask = _mm256_set1_epi8(7u);
    __m256i t = _mm256_srli_epi16(a, 5);
    t &= mask;
    t &= Q;
    __m256i ret = _mm256_add_epi8(a, t);
    ret &= Q;
    return ret;
}

/// NOTE: assumes that each gf31 element is in a single uint8_t
static inline
__m256i gf31v_full_red_u256(const __m256i a) {
    const __m256i mod = _mm256_set1_epi8(31u);
    const __m256i one = _mm256_set1_epi8(1u);
    __m256i t;

    t = _mm256_add_epi8(a, one);
    t = _mm256_srli_epi16(t, 5);
    t = _mm256_and_si256(t, one);
    t = _mm256_add_epi8(a, t);
    t = _mm256_and_si256(t, mod);
    return t;
}

/// NOTE: assumes that each gf31 element is in a single uint8_t
/// NOTE: not really a full reduction, as 31 is not going to be reduced
static inline
__m256i gf31v_add_u256(const __m256i a,
                       const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_red_u256(c);
}

static inline
__m256i gf31v_full_add_u256(const __m256i a,
                            const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_full_red_u256(c);
}

/// NOTE: assumes reduced inputs
/// NOTE: not full recution
static inline
__m256i gf31v_mul_u256(const __m256i a,
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

/// full
static inline
__m256i gf31v_full_mul_u256(const __m256i a,
                            const __m256i b) {
    return gf31v_red_u256(gf31v_mul_u256(a, b));
}
#endif
