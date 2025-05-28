#include <stdint.h>
#include <stdio.h>

#include "arith.h"

const uint64_t N = 1u<<10u;


uint32_t test_add() {
    for (uint64_t a = 0; a < N; ++a) {
        for (uint64_t b = 0; b < N; ++b) {
            uint32_t i = rand();
            uint32_t j = rand();

            const uint32_t c1 = gf2to32_mul(i, j);
            const uint32_t c2 = gf2to32_mul_v2(i, j);

            if (c1 != c2){
                printf("error: test add: %d %d\n", c1, c2);
                return 1;
            }
        }
    }

    return 0;
}

#ifdef USE_AVX2
uint32_t test_vector_add() {
    return 0;
}

uint32_t test_vector_mul() {
    uint32_t tmp1[8] = {0};
    for (uint64_t a = 0; a < N; ++a) {
        for (uint64_t b = 0; b < N; ++b) {
            uint32_t i = rand();
            uint32_t j = rand();
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = i; }
            const __m256i aa = _mm256_loadu_si256((const __m256i *)tmp1);
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = j; }
            const __m256i bb = _mm256_loadu_si256((const __m256i *)tmp1);

            const uint32_t c1 = gf2to32_mul(i, j);
            const __m256i c2 = gf2to32v_mul_u256(aa, bb);
            _mm256_storeu_si256((__m256i *)tmp1, c2);

            for (uint32_t k = 0; k < 8; ++k) {
                if (c1 != tmp1[k]){
                    printf("error: test vector mul: %d %d\n", c1, tmp1[k]);
                    return 1;
                }
            }

            const __m256i c3 = gf2to32v_mul_u256_v2(aa, bb);
            _mm256_storeu_si256((__m256i *)tmp1, c3);

            for (uint32_t k = 0; k < 8; ++k) {
                if (c1 != tmp1[k]){
                    printf("error: test vector mul (2): %d %d\n", c1, tmp1[k]);
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

#ifdef USE_AVX2
    // if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
