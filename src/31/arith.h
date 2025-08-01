#include <stdint.h>

/// The prime field modulus
#define PRIME 31

/// Fast reduction modulo 31 for 16-bit values
/// This constant-time formula works for inputs x <= 63487 = 2^15.95
/// Uses Barrett reduction technique with precomputed constants
#define _gf31_reduce16(x) ((x) - 31 * (((uint32_t)(x) * 67651) >> 21))

/// Fast reduction modulo 31 for 32-bit values
/// This constant-time formula works for inputs x <= 4160749567 = 2^31.95
/// Uses Barrett reduction technique with precomputed constants
#define _gf31_reduce32(x) ((x) - 31 * (((uint64_t)(x) * 4433514629) >> 37))


/// Type definition for GF(31) field elements
typedef uint8_t ff_t;

/// Addition in GF(31)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b mod 31
static inline
ff_t gf31_add(ff_t a, ff_t b) {
    return (a+b) % PRIME;
}

/// Subtraction in GF(31)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b mod 31
static inline
ff_t gf31_sub(ff_t a, ff_t b) {
    return (a+PRIME-b) % PRIME;
}

/// Multiplication in GF(31)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b mod 31
static inline
ff_t gf31_mul(ff_t a, ff_t b) {
    return (a*b)%PRIME;
}

/// Negation in GF(31)
/// \param a[in]: value to negate
/// \return -a mod 31
static inline
ff_t gf31_neg(ff_t a) {
    return (PRIME - a) % PRIME;
}


#ifdef USE_AVX2 
#include <immintrin.h>

/// AVX2 vectorized partial reduction of 32 GF(31) elements
/// Reduces values to range [0,31] but doesn't reduce 31 to 0
/// \param a[in]: vector of 32 GF(31) elements to reduce
/// \return vector of reduced elements
/// \note Assumes that each GF(31) element is stored in a single uint8_t
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

/// AVX2 vectorized complete reduction of 32 GF(31) elements
/// Fully reduces values to range [0,30] (proper GF(31) elements)
/// \param a[in]: vector of 32 GF(31) elements to reduce
/// \return vector of completely reduced elements
/// \note Assumes that each GF(31) element is stored in a single uint8_t
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

/// AVX2 vectorized addition of 32 GF(31) elements with partial reduction
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of element-wise sums with partial reduction
/// \note Assumes that each GF(31) element is stored in a single uint8_t
/// \note Not a complete reduction, as 31 is not reduced to 0
static inline
__m256i gf31v_add_u256(const __m256i a,
                       const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_red_u256(c);
}

/// AVX2 vectorized addition of 32 GF(31) elements with complete reduction
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of completely reduced element-wise sums
/// \note Performs complete reduction to ensure all elements are in range [0,30]
static inline
__m256i gf31v_full_add_u256(const __m256i a,
                            const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_full_red_u256(c);
}

/// AVX2 vectorized multiplication of 32 GF(31) elements with partial reduction
/// Implements optimized 16-bit multiplication and reduction for GF(31) elements
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of element-wise products with partial reduction
/// \note Assumes inputs have already been reduced
/// \note Does not perform complete reduction (31 not reduced to 0)
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

/// AVX2 vectorized multiplication of 32 GF(31) elements with additional reduction
/// Multiplies two vectors and applies partial reduction to the result
/// \param a[in]: first vector of 32 GF(31) elements
/// \param b[in]: second vector of 32 GF(31) elements
/// \return vector of element-wise products with additional reduction
/// \note Still only applies partial reduction (not full reduction)
static inline
__m256i gf31v_full_mul_u256(const __m256i a,
                            const __m256i b) {
    return gf31v_red_u256(gf31v_mul_u256(a, b));
}
#endif
