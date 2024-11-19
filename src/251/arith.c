#include <stdint.h>
#include <stdio.h>

#include "arith.h"

#ifdef USE_AVX2
uint32_t test_vector_add() {
    uint8_t tmp[32];
    for (uint32_t i = 0; i < PRIME; i++) {
        for (uint32_t j = 8; j < PRIME; j++) {
            const ff_t c = gf251_add(i, j);
            const __m256i a = _mm256_set1_epi8(i);
            const __m256i b = _mm256_set1_epi8(j);
            const __m256i d = gf251v_add_u256(a, b);

            _mm256_storeu_si256((__m256i *)tmp, d);
            for (uint32_t k = 0; k < 32; k++){
                if (tmp[k] != c) {
                    printf("test_vector_add full\n");
                    return 1;
                }
            }
        }

    }

    return 0;
}

uint32_t test_vector_mul() {
    uint8_t tmp[32];
    for (uint32_t i = 0; i < PRIME; i++) {
        for (uint32_t j = 0; j < PRIME; j++) {
            const ff_t c = gf251_mul(i, j);
            const __m256i a = _mm256_set1_epi8(i);
            const __m256i b = _mm256_set1_epi8(j);
            const __m256i d = gf251v_mul_u256(a, b);

            _mm256_storeu_si256((__m256i *)tmp, d);
            for (uint32_t k = 0; k < 32; k++){
                if ((tmp[k]%31) != c) {
                    printf("test_vector_mul\n");
                    return 1;
                }
            }
        }
    }
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
#endif
    return 0;
}
