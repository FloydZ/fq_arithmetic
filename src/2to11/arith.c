#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

#ifdef USE_AVX2
#include <immintrin.h>

uint32_t test_arith_vector_mul() {
    uint16_t tmp[16];
    for (int i = 0; i < 1u << 11; ++i) {
        for (int j = 0; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to11v_mul_u256(a256, b256);
            const __m256i cc256 = gf2to11v_mul_u256_v2(a256, b256);
            const gf2to11 c1 = gf2to11_mul(i, j);
            const gf2to11 c2 = gf2to11_mul_v2(i, j);
            _mm256_storeu_si256((__m256i *)tmp, c256);
            if (c1 != c2) {
                printf("gf2to12 mul error\n");
                return 1;
            }
            for (uint32_t k = 0; k < 16; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 avx mul error\n");
                    return 1;
                }
            }
            _mm256_storeu_si256((__m256i *)tmp, cc256);
            for (uint32_t k = 0; k < 16; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 avx_v2 mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}

uint32_t test_vector_add() {
    const uint32_t N = 256;

    gf2to11 *v1 = gf2to11_vector_alloc(N);
    gf2to11 *v2 = gf2to11_vector_alloc(N);
    gf2to11 *v3 = gf2to11_vector_alloc(N);

    gf2to11_vector_random(v1, N);
    gf2to11_vector_zero(v2, N);
    gf2to11_vector_zero(v3, N);

    gf2to11_vector_add(v2, v1, N);
    gf2to11_vector_add_u256(v3, v1, N);

    // gf2to11_vector_print(v2, N);
    // gf2to11_vector_print(v3, N);
    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_add");
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_add_gf2() {
    const uint32_t N = 256;

    gf2 *v1 = gf2_vector_alloc(N);
    gf2to11 *v2 = gf2to11_vector_alloc(N);
    gf2to11 *v3 = gf2to11_vector_alloc(N);

    gf2_vector_random(v1, N);
    gf2to11_vector_zero(v2, N);
    gf2to11_vector_zero(v3, N);

    gf2to11_vector_add_gf2(v2, v1, N);
    gf2to11_vector_add_gf2_u256(v3, v1, N);

    // gf2to11_vector_print(v2, N);
    // gf2to11_vector_print(v3, N);
    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_add_gf2");
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}


uint32_t test_vector_scalar_add_gf2_v3() {
    const uint32_t N = 256;

    gf2 *v1     = gf2_vector_alloc(N);
    gf2to11 *v2 = gf2to11_vector_alloc(N);
    gf2to11 *v3 = gf2to11_vector_alloc(N);

    gf2_vector_random(v1, N);
    gf2to11_vector_zero(v2, N);
    gf2to11_vector_zero(v3, N);

    gf2to11 t = 1;
    gf2to11_vector_scalar_add_gf2_v3(v2, t, v1, N);
    gf2to11_vector_scalar_add_gf2_u256_v3(v3, t, v1, N);

    // gf2to11_vector_print(v2, N);
    // gf2to11_vector_print(v3, N);
    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_scalar_add_gf2_v3");
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_mul_acc() {
    const uint32_t N = 256;

    gf2to11 *v1 = gf2to11_vector_alloc(N);
    gf2to11 *v2 = gf2to11_vector_alloc(N);

    gf2to11_vector_zero(v1, N);
    gf2to11_vector_zero(v2, N);

    const gf2to11 t1 = gf2to11_vector_mul_acc(v2, v1, N);
    const gf2to11 t2 = gf2to11_vector_mul_acc_u256(v2, v1, N);

    if (t1 != t2) {
        printf("error test_vector_mul_acc");
        return 1;
    }

    free(v1); free(v2);
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if (test_arith_vector_mul()) { return 1; }

    if (test_vector_add()) { return 1; }
    if (test_vector_add_gf2()) { return 1; }
    if (test_vector_scalar_add_gf2_v3()) { return 1; }
    if (test_vector_mul_acc()) { return 1; }
#endif

    printf("all good\n");
	return 0;
}
