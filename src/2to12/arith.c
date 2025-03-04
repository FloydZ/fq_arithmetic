#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

#include "../2/matrix.h"

#if defined(USE_NEON)
uint32_t test_arith_vector_mul() {
    uint16_t tmp[16];
    for (uint16_t i = 1; i < 1u << 11; ++i) {
        for (uint16_t j = 1; j < 1u << 11; ++j) {
            const uint16x8_t a128 = vdupq_n_u16(i);
            const uint16x8_t b128 = vdupq_n_u16(j);
            const uint16x8_t c128 = gf2to12v_mul_u128(a128, b128);
            vst1q_u16(tmp, c128);

            const gf2to12 c1 = gf2to12_mul(i, j);
            for (uint32_t k = 0; k < 8; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 neon 128 mul error\n");
                    return 1;
                }
            }

            uint16x8x2_t a256, b256, c256;
            a256.val[0] = vdupq_n_u16(i); a256.val[1] = vdupq_n_u16(i);
            b256.val[0] = vdupq_n_u16(j); b256.val[1] = vdupq_n_u16(j);
            c256 = gf2to12v_mul_u256(a256, b256);
            vst1q_u16_x2(tmp, c256);

            for (uint32_t k = 0; k < 16; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 neon 256 mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}
#endif

#ifdef USE_AVX2

uint32_t test_arith_vector_mul() {
    uint16_t tmp[32];
    for (int i = 0; i < 1u << 11; ++i) {
        for (int j = 0; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to12v_mul_u256(a256, b256);
            const __m256i d256 = gf2to12v_mul_u256_v2(a256, b256);
            const gf2to12 c = gf2to12_mul(i, j);
            const gf2to12 d = gf2to12_mul_v2(i, j);
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

#ifdef USE_AVX512
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

uint32_t test_vector_add() {
    const uint32_t N = 256;

    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2to12_vector_random(v1, N);
    gf2to12_vector_zero(v2, N);
    gf2to12_vector_zero(v3, N);

    gf2to12_vector_add(v2, v1, N);
    gf2to12_vector_add_u256(v3, v1, N);

    // gf2to12_vector_print(v2, N);
    // gf2to12_vector_print(v3, N);
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
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2_vector_random(v1, N);
    gf2to12_vector_zero(v2, N);
    gf2to12_vector_zero(v3, N);

    gf2to12_vector_add_gf2(v2, v1, N);
    gf2to12_vector_add_gf2_u256(v3, v1, N);

    // gf2to12_vector_print(v2, N);
    // gf2to12_vector_print(v3, N);
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
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2_vector_random(v1, N);
    gf2to12_vector_zero(v2, N);
    gf2to12_vector_zero(v3, N);

    gf2to12 t = 1;
    gf2to12_vector_scalar_add_gf2_v3(v2, t, v1, N);
    gf2to12_vector_scalar_add_gf2_u256_v3(v3, t, v1, N);

    // gf2to12_vector_print(v2, N);
    // gf2to12_vector_print(v3, N);
    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_scalar_add_gf2_v3");
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_set_to_gf2() {
    const uint32_t N = 321;

    gf2 *v1     = gf2_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2_vector_random(v1, N);

    gf2to12 t = 1;
    gf2to12_vector_set_to_gf2(v2, v1, N);
    gf2to12_vector_set_to_gf2_u256(v3, v1, N);

    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_scalar_add_gf2_v3\n");
            gf2to12_vector_print(v2, N);
            gf2to12_vector_print(v3, N);
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_mul_acc() {
    const uint32_t N = 256;

    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);

    gf2to12_vector_zero(v1, N);
    gf2to12_vector_zero(v2, N);

    const gf2to12 t1 = gf2to12_vector_mul_acc(v2, v1, N);
    const gf2to12 t2 = gf2to12_vector_mul_acc_u256(v2, v1, N);

    if (t1 != t2) {
        printf("error test_vector_mul_acc");
        return 1;
    }

    free(v1); free(v2);
    return 0;
}



uint32_t test_matrix_add_gf2() {
    const uint32_t nrows = 17;
    const uint32_t ncols = 17;

    gf2 *v1     = gf2_matrix_alloc(nrows, ncols);
    gf2to12 *v2 = gf2to12_matrix_alloc(nrows, ncols);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows, ncols);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows, ncols);

    gf2_matrix_random(v1, nrows, ncols);
    gf2to12_matrix_random(v2, nrows, ncols);

    gf2to12_matrix_add_gf2(C1, v1, v2, nrows, ncols);
    gf2to12_matrix_add_gf2_u256(C2, v1, v2, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                printf("error test_matrix_add_gf2");
                gf2to12_matrix_print(C1, nrows, ncols);
                gf2to12_matrix_print(C2, nrows, ncols);
                return 1;
            }
        }
    }

    free(v1); free(v2); free(C1); free(C2);
    return 0;
}

uint32_t test_matrix_add_scalar_gf2() {
    const uint32_t nrows = 5;
    const uint32_t ncols = 45; // 45

    gf2 *v1     = gf2_matrix_alloc(nrows, ncols);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows, ncols);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows, ncols);

    gf2_matrix_random(v1, nrows, ncols);

    const gf2to12 v2 = 2;
    gf2to12_matrix_add_scalar_gf2(C1, v2, v1, nrows, ncols);
    gf2to12_matrix_add_scalar_gf2_u256(C2, v2, v1, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                printf("error test_matrix_add_scalar_gf2\n");
                gf2to12_matrix_print(C1, nrows, ncols);
                gf2to12_matrix_print(C2, nrows, ncols);
                return 1;
            }
        }
    }

    free(v1); free(C1); free(C2);
    return 0;
}

uint32_t test_matrix_mul_gf2() {
    const uint32_t nrows1 = 33;
    const uint32_t ncols1 = 32;
    const uint32_t ncols2 = 1;

    gf2 *v1     =     gf2_matrix_alloc(nrows1, ncols1);
    gf2to12 *v2 = gf2to12_matrix_alloc(ncols1, ncols2);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows1, ncols2);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows1, ncols2);

    gf2_matrix_random(v1, nrows1, ncols1);
    gf2to12_matrix_random(v2, ncols1, ncols2);

    gf2to12_matrix_mul_gf2(C1, v1, v2, nrows1, ncols1, ncols2);
    gf2to12_matrix_mul_gf2_u256(C2, v1, v2, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows1, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows1, i, j);
            if (t1 != t2) {
                printf("error test_matrix_mul_gf2\n");
                gf2to12_matrix_print(C1, nrows1, ncols2);
                gf2to12_matrix_print(C2, nrows1, ncols2);
                return 1;
            }
        }
    }

    free(v1); free(v2); free(C1); free(C2);
    return 0;
}

uint32_t test_matrix_mul() {
    const uint32_t nrows1 = 22;
    const uint32_t ncols1 = 637;
    const uint32_t ncols2 = 1;

    gf2to12 *v1 = gf2to12_matrix_alloc(nrows1, ncols1);
    gf2to12 *v2 = gf2to12_matrix_alloc(ncols1, ncols2);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows1, ncols2);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows1, ncols2);

    gf2to12_matrix_random(v1, nrows1, ncols1);
    gf2to12_matrix_random(v2, ncols1, ncols2);

    gf2to12_matrix_mul(C1, v1, v2, nrows1, ncols1, ncols2);
    gf2to12_matrix_mul_u256(C2, v1, v2, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows1, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows1, i, j);
            if (t1 != t2) {
                printf("error test_matrix_mul\n");
                gf2to12_matrix_print(C1, nrows1, ncols2);
                gf2to12_matrix_print(C2, nrows1, ncols2);
                return 1;
            }
        }
    }

    free(v1); free(v2); free(C1); free(C2);
    return 0;
}


#endif


int main() {
#if defined(USE_AVX2) || defined(USE_NEON)
    if (test_arith_vector_mul()) { return 1; }
#if defined(USE_AVX2)
    //if (test_vector_add()) { return 1; }
    //if (test_vector_add_gf2()) { return 1; }
    //if (test_vector_scalar_add_gf2_v3()) { return 1; }
    //if (test_vector_mul_acc()) { return 1; }
    //if (test_vector_set_to_gf2()) { return 1; }

    //if (test_matrix_add_gf2()) { return 1; }
    if (test_matrix_add_scalar_gf2()) { return 1; }
    //if (test_matrix_mul_gf2()) { return 1; }
    //if (test_matrix_mul()) { return 1; }
#endif
#endif

    printf("all good\n");
	return 0;
}
