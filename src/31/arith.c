#include <stdint.h>

#define PRIME 31

// Work as soon as x <= 63487 = 2^15.95
#define _gf31_reduce16(x) ((x) - 31 * (((uint32_t)(x) * 67651) >> 21))

// Work as soon as x <= 4160749567 = 2^31.95
#define _gf31_reduce32(x) ((x) - 31 * (((uint64_t)(x) * 4433514629) >> 37))


typedef uint8_t ff_t;

ff_t gf31_add(ff_t a, ff_t b) {
    return (a+b) % PRIME;
}

ff_t gf31_sub(ff_t a, ff_t b) {
    return (a+PRIME-b) % PRIME;
}

ff_t gf31_mul(ff_t a, ff_t b) {
    return (a*b)%PRIME;
}

ff_t gf31_neg(ff_t a) {
    return (PRIME - a) % PRIME;
}


#ifdef USE_AVX2 
#include <immintrin.h>

/// NOTE: assumes that each gf31 element is in a single uint8_t
__m256i gf31v_red_u256(const __m256i a) {
    const __m256i Q = _mm256_set1_epi8(31);
    __m256i t = _mm256_slli_epi16(a, 5); 
    t &= Q;
    __m256i ret = _mm256_add_epi8(a, t);
    ret &= Q;
    return ret;
}
                                      
/// NOTE: assumes that each gf31 element is in a single uint8_t
__m256i gf31v_add_u256(const __m256i a,
                                      const __m256i b) {
    const __m256i c = _mm256_add_epi8(a, b);
    return gf31v_red_u256(c);
}


#endif


int main() {
    return 0;
}
