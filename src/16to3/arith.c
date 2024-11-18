#include <stdint.h>
#include <stdbool.h>

#include "arith.h"

bool test_add() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t = gf16to3_add(i, i);
        if (t != 0) {
            return 1;
        }
    }

    return 0;
}

bool test_mul() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t1 = gf16to3_mul(i, 1);
        if (t1 != i) { return 1; }

        const gf16to3 t2 = gf16to3_mul(1, i);
        if (t2 != i) { return 1; }
    }

    return 0;
}

#ifdef USE_AVX2

uint32_t test_vector_mul() {
    uint16_t tmp[32];
    for (int i = 0; i < 1u << 12; ++i) {
        for (int j = 0; j < 1u << 12; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf16to3v_mul_u256(a256, b256);
            const uint16_t c = gf16to3_mul(i, j);

            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf16to3 avx mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}
#endif



int main() {
    if (test_add()) { return 1; }
    if (test_mul()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif
    return 0;
}
