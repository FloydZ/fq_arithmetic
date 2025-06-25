#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"


const uint64_t N = 1u << 10u;

#ifdef USE_AVX2
uint32_t test_vector_add() {
    return 0;
}

int test_vector_mul() {
    gf2to256 c1, c2, c3, a, b;
    for (uint64_t i = 0; i < N; i++) {
        gf2to256_set_random(a);
        gf2to256_set_random(b);

        gf2to256_mul(c1, a, b);
        gf2to256v_mul_u256_(c2, a, b);
        gf2to256_mul_v2(c3, a, b);

        for (uint32_t t = 0; t < 4; t++) {
            if (c1[t] != c2[t]) {
                printf("test_vector_mul: %llx %llx\n",
                       (unsigned long long)c1, (unsigned long long)c2);
                return 1;
            }
        }

        for (uint32_t t = 0; t < 4; t++) {
            if (c1[t] != c3[t]) {
                printf("test_vector_mul(3): %llx %llx\n",
                       (unsigned long long)c1, (unsigned long long)c3);
                return 1;
            }
        }
    }
    return 0;
}

int test_mul_u256() {
    gf2to256 a, b, c, t1;
    for (uint64_t i = 0; i < N; i++) {
        gf2to256_set_random(a);
        gf2to256_set_random(b);
        gf2to256_mul(c, a, b);

        const __m256i aa = _mm256_loadu_si256((const __m256i *)a);
        const __m256i bb = _mm256_loadu_si256((const __m256i *)b);

        const __m256i c1 = gf2to256v_mul_u256(aa, bb);
        const __m256i c2 = gf2to256v_mul_u256_v2(aa, bb);
        _mm256_storeu_si256((__m256i *)t1, c1);
        for (uint32_t t = 0; t < 4; t++) {
            if (c[t] != t1[t]) {
                printf("test_mul_256: %llx %llx\n",
                       (unsigned long long)c[t], (unsigned long long)t1[t]);
                return 1;
            }
        }

        _mm256_storeu_si256((__m256i *)t1, c2);
        for (uint32_t t = 0; t < 4; t++) {
            if (c[t] != t1[t]) {
                printf("test_mul_u256 (2): %llx %llx\n",
                       (unsigned long long)c[t], (unsigned long long)t1[t]);
                return 1;
            }
        }
    }

    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if (test_mul_u256()) { return 1; }

    // if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }

#endif

    printf("all good\n");
    return 0;
}
