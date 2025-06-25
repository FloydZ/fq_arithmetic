#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
// #include "matrix.h"

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

int test_vector_add_gf2_u256() {
    const uint32_t n = 123;

    gf2to32 *a1 = gf2to32_vector_alloc(n);
    gf2to32 *a2 = gf2to32_vector_alloc(n);
    gf2 *b = gf2_vector_alloc(n);

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a1, n);
        gf2_vector_random(b, n);
        gf2to32_vector_copy(a2, a1, n);
        gf2to32_vector_add_gf2(a1, b, n);
        gf2to32_vector_add_gf2_u256(a2, b, n);

        for (uint32_t k = 0; k < n; k++) {
            if (a1[k] != a2[k]) {
                printf("error test_vector_add_gf2_u256\n");
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
        free(a1); free(a2); free(b);
    return ret;
}

int test_vector_scalar_add_u256() {
    const uint32_t n = 123;

    gf2to32 *a1 = gf2to32_vector_alloc(n);
    gf2to32 *c1 = gf2to32_vector_alloc(n);
    gf2to32 *c2 = gf2to32_vector_alloc(n);

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a1, n);
        gf2to32_vector_random(c1, n);
        gf2to32_vector_copy(c2, c1, n);
        gf2to32 a = rand();

        gf2to32_vector_scalar_add(c1, a, a1, n);
        gf2to32_vector_scalar_add_u256(c2, a, a1, n);

        for (uint32_t k = 0; k < n; k++) {
            if (c1[k] != c2[k]) {
                printf("error test_vector_scalar_add_u256\n");
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
        free(a1); free(c1); free(c2);
    return ret;
}

#endif

int main() {
    if (test_add()) { return 1; }

#ifdef USE_AVX2
    // if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_vector_add_gf2_u256()) { return 1; }
    if (test_vector_scalar_add_u256()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
