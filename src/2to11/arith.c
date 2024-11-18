#include <stdint.h>
#include <stdio.h>

#include "arith.h"


#ifdef USE_AVX2
#include <immintrin.h>

uint32_t test_vector_mul() {
    uint16_t tmp[16];
    for (int i = 2; i < 1u << 11; ++i) {
        for (int j = 1; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to11v_mul_u256(a256, b256);
            const ff_t c = gf2to11_mul(i, j);
            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx mul error\n");
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
    if (test_vector_mul()) { return 1; }
#endif

	return 0;
}
