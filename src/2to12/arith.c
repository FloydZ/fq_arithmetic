#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "arith.h"


#ifdef USE_AVX2
uint32_t test_vector_mul() {
    uint16_t tmp[32];
    for (int i = 0; i < 1u << 11; ++i) {
        for (int j = 0; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to12v_mul_u256(a256, b256);
            const __m256i d256 = gf2to12v_mul_u256_v2(a256, b256);
            const ff_t c = gf2to12_mul(i, j);
            const ff_t d = gf2to12_mul_v2(i, j);
            if (c != d) {
                printf("gf2to12 mul error\n");
                return 1;
            }

            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx mul error\n");
                    return 1;
                }
            }
            
            _mm256_storeu_si256((__m256i *)tmp, d256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx mul_v2 error\n");
                    return 1;
                }
            }

#ifdef __AVX512VLINTRIN_H
            const __m256i e256 = gf2to12v_mul_u256_avx512(a256, b256);
            _mm256_storeu_si256((__m256i *)tmp, e256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx512 mul256 error\n");
                    return 1;
                }
            }


            const __m512i a512 = _mm512_set1_epi16(i);
            const __m512i b512 = _mm512_set1_epi16(j);
            const __m512i e512 = gf2to12v_mul_u512(a512, b512);
            _mm512_storeu_si512((__m512i *)tmp, e512);
            for (int k = 0; k < 32; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx512 mul512 error\n");
                    return 1;
                }
            }
#endif
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
