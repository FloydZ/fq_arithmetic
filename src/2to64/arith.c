#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"

const uint64_t N = 1u << 10;

uint32_t test_mul() {
    for (int a = 0; a < N; ++a) {
        for (int b = 0; b < N; ++b) {
            uint64_t i =  rand() ^ (((uint64_t)rand())<<32u);
            uint64_t j =  rand() ^ (((uint64_t)rand())<<32u);

            const uint64_t c1 = gf2to64_mul(i, j);
            const uint64_t c2 = gf2to64_mul_v2(i, j);

            if (c1 != c2) {
                printf("error test_add() %lu %lu\n", c1, c2);
                return 1;
            }
        }
    }
    return 0;
}

#ifdef USE_AVX2
uint32_t test_mul_u256() {
    uint64_t tmp1[8] = {0};
    for (uint64_t a = 0; a < N; ++a) {
        for (uint64_t b = 0; b < N; ++b) {
            uint64_t i = rand() ^ (((uint64_t)rand()) << 32u);
            uint64_t j = rand() ^ (((uint64_t)rand()) << 32u);
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = i; }
            const __m256i aa = _mm256_loadu_si256((const __m256i *)tmp1);
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = j; }
            const __m256i bb = _mm256_loadu_si256((const __m256i *)tmp1);

            const uint64_t c1 = gf2to64_mul(i, j);
            const uint64_t c2 = gf2to64v_mul_u256(i, j);
            const __m256i c3 = gf2to64v_mul_u256_(aa, bb);
            _mm256_storeu_si256((__m256i *)tmp1, c3);

            if (c1 != c2){
                printf("error: test vector mul: %lu %lu\n", c1, c2);
                return 1;
            }

            for (uint32_t k = 0; k < 4; ++k) {
                if (c1 != tmp1[k]){
                    printf("error: test vector mul (2): %lu %lu\n", c1, tmp1[k]);
                    return 1;
                }
            }
        }
    }
    return 0;
}

#endif

int main() {
    if (test_mul()) { return 1; }

#ifdef USE_AVX2
    if (test_mul_u256()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
