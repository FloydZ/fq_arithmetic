#include <stdint.h>

// GF(2)/[x^32 + x^22 + x^2 + x^1 + 1]
#define MODULUS 0b10000000000000000000111


typedef uint32_t gf2to32;

gf2to32 gf32_add(const gf2to32 a,
                 const gf2to32 b) {
    return a ^ b;
}

gf2to32 gf32_sub(const gf2to32 a,
                 const gf2to32 b) {
    return a ^ b;
}

gf2to32 gf32_mul(const gf2to32 a,
                 const gf2to32 b) {
    uint32_t result = 0;
    uint32_t shifted = a;

    for (uint32_t i = 0; i < 32; ++i) {
        if ((b & (1u << i)) != 0u) {
            result ^= shifted;
        }

        if ((shifted & (1u << 31u)) != 0u) {
            shifted <<= 1u;
            shifted ^= MODULUS;
        } else {
            shifted <<= 1u;
        }
    }

    return result;
}

gf2to32 gf32_neg(const gf2to32 a) {
    return ~a;
}

#ifdef USE_AVX2
#include <immintrin.h>

__m256i gf32v_add_u256(const __m256i a,
                       const __m256i b) {
}

#endif
