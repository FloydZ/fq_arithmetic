#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"
#include "bitslice.h"

#include "../16/vector.h"

#ifdef USE_AVX2

uint32_t test_matrix_map_gf16_to_gf256_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 16;
    gf16 *B   = gf16_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_map_gf16_to_gf256(C1, B, nrows, ncols);
    gf256_matrix_map_gf16_to_gf256_u256(C2, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_map_gf16_to_gf256_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 100;
    const uint32_t ncols = 100;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf256 *B  = gf256_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    gf256_matrix_random(B, nrows, ncols);
    gf256_matrix_add(C1, A, B, nrows, ncols);
    gf256_matrix_add_u256(C2, A, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            const gf256 t3 = gf256_matrix_get_entry(B, nrows, i, j);
            if (t1 != t2) {
                printf("test_matrix_add_u256 error: %d %d\n", i, j);
                ret = 1;
                break;
            }
            if (t1 != t3) {
                printf("test_matrix_add_u256 error: %d %d\n", i, j);
                ret = 1;
                break;
            }
        }
    }

    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_gf16_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf16 *B   = gf16_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_add_gf16(C1, A, B, nrows, ncols);
    gf256_matrix_add_gf16_u256(C2, A, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_gf16_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_multiple_gf16_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 16;
    const uint32_t ncols = 16;
    gf16 *B   = gf16_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_add_multiple_gf16(C1, scalar, B, nrows, ncols);
    gf256_matrix_add_multiple_gf16_u256(C2, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_multiple_gf16_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_multiple_2_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf256 *B  = gf256_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_add_multiple_2(C1, scalar, B, nrows, ncols);
    gf256_matrix_add_multiple_2_u256(C2, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_multiple_2_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_multiple_3_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf256 *B  = gf256_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_add_multiple(C1, A, scalar, B, nrows, ncols);
    gf256_matrix_add_multiple_u256(C2, A, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_multiple_3_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_product_gf16_1_u256() {
    uint32_t ret = 0;
    const uint32_t nrows1 = 15;
    const uint32_t ncols1 = 15;
    const uint32_t ncols2 = 15;

    gf16 *A   = gf16_matrix_alloc (ncols1, ncols2);
    gf256 *B  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *C1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *C2 = gf256_matrix_alloc(nrows1, ncols2);

    gf16_matrix_random(A, ncols1, ncols2);
    gf256_matrix_random(B, ncols1, ncols2);
    gf256_matrix_product_gf16_1(C1, A, B, nrows1, ncols1, ncols2);
    gf256_matrix_product_gf16_1_u256(C2, A, B, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get_entry(C1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get_entry(C2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows1, ncols2);
                gf256_matrix_print(C2, nrows1, ncols2);
                printf("test_matrix_product_gf16_1_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_product_gf16_2_u256() {
    uint32_t ret = 0;
    const uint32_t nrows1 = 15;
    const uint32_t ncols1 = 15;
    const uint32_t ncols2 = 15;

    gf16 *a   = gf16_matrix_alloc (ncols1, ncols2);
    gf256 *b  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf16_matrix_random(a, ncols1, ncols2);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_product_gf16_2(c1, b, a, nrows1, ncols1, ncols2);
    gf256_matrix_product_gf16_2_u256(c2, b, a, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get_entry(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get_entry(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_product_gf16_2_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_product_8x8xC_u256() {
    uint32_t ret = 0;
    // NOTE hardcoded
    const uint32_t nrows1 = 8;
    const uint32_t ncols1 = 8;
    // free
    const uint32_t ncols2 = 15;

    gf256 *b  = gf256_matrix_alloc(ncols1, ncols2);
    gf256 *a  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf256_matrix_random(a, nrows1, ncols1);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_product(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_product_8x8xC_u256(c2, a, b, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get_entry(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get_entry(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_product_8x8xX_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_product_16x16xC_u256() {
    uint32_t ret = 0;
    // NOTE hardcoded
    const uint32_t nrows1 = 16;
    const uint32_t ncols1 = 16;
    // free
    const uint32_t ncols2 = 15;

    gf256 *b  = gf256_matrix_alloc(ncols1, ncols2);
    gf256 *a  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf256_matrix_random(a, nrows1, ncols1);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_product(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_product_16x16xC_u256(c2, a, b, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get_entry(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get_entry(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_product_8x8xX_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_product_le32xBxle16_u256() {
    uint32_t ret = 0;
    // NOTE hardcoded
    const uint32_t nrows1 = 16;
    const uint32_t ncols1 = 13;
    // free
    const uint32_t ncols2 = 15;

    gf256 *b  = gf256_matrix_alloc(ncols1, ncols2);
    gf256 *a  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf256_matrix_random(a, nrows1, ncols1);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_product(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_product_le32xBxle16_u256(c2, a, b, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get_entry(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get_entry(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_product_le32xBxle16_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}


uint32_t test_vector_set_to_gf16_u256() {
    uint32_t ret = 0;
    const uint32_t N = 31;
    gf16 *A = gf16_vector_alloc(N); 
    gf256 *B1 = gf256_vector_alloc(N); 
    gf256 *B2 = gf256_vector_alloc(N);
    gf256 *B3 = gf256_vector_alloc(N);

    gf16_vector_rand(A, N);
    gf256_vector_set_to_gf16(B1, A, N);
    gf256_vector_set_to_gf16_u256(B2, A, N);
    gf256_vector_set_to_gf16_u256_v2(B3, A, N);

    for (uint32_t i = 0; i < N; i++) {
        if (B1[i] != B2[i]) {
            printf("test_vector_set_to_gf16_u256 error %d\n", i);
            ret = 1;
            break;
        }
        if (B1[i] != B3[i]) {
            printf("test_vector_set_to_gf16_u256_v2 error %d\n", i);
            ret = 1;
            break;
        }
    }

    free(A); free(B1); free(B2); free(B3);
    return ret;
}


uint32_t test_vector_set_to_gf2_u256() {
    uint32_t ret = 0;
    const uint32_t N = 31;
    gf2 *A = gf2_vector_alloc(N);
    gf256 *B1 = gf256_vector_alloc(N);
    gf256 *B2 = gf256_vector_alloc(N);

    gf2_vector_random(A, N);
    gf256_vector_set_to_gf2(B1, A, N);
    gf256_vector_set_to_gf2_u256(B2, A, N);

    for (uint32_t i = 0; i < N; i++) {
        if (B1[i] != B2[i]) {
            printf("test_vector_set_to_gf2_u256 error %d\n", i);
            ret = 1;
            break;
        }
    }

    free(A); free(B1); free(B2);
    return ret;
}

uint32_t test_vector_add_gf2_2_u256() {
    uint32_t ret = 0;
    const uint32_t N = 31;
    gf2 *A = gf2_vector_alloc(N);
    gf256 *B = gf256_vector_alloc(N);
    gf256 *C1 = gf256_vector_alloc(N);
    gf256 *C2 = gf256_vector_alloc(N);

    gf2_vector_random(A, N);
    gf256_vector_add_2_gf2(C1, B, A, N);
    gf256_vector_add_2_gf2_u256(C2, B, A, N);

    for (uint32_t i = 0; i < N; i++) {
        if (C1[i] != C2[i]) {
            printf("test_vector_add_2_gf2_u256 error %d\n", i);
            ret = 1;
            break;
        }
    }

    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_vector_scalar_gf2_u256() {
    uint32_t ret = 0;
    const uint32_t N = 31;
    gf2 *A = gf2_vector_alloc(N);
    gf256 *C1 = gf256_vector_alloc(N);
    gf256 *C2 = gf256_vector_alloc(N);

    gf256 B = 1;
    gf2_vector_random(A, N);
    gf256_vector_scalar_gf2(C1, B, A, N);
    gf256_vector_scalar_gf2_u256(C2, B, A, N);

    for (uint32_t i = 0; i < N; i++) {
        if (C1[i] != C2[i]) {
            gf256_vector_print(C1, N);
            gf256_vector_print(C2, N);
            printf("test_vector_scalar_2_gf2_u256 error %d\n", i);
            ret = 1;
            break;
        }
    }

    free(A); free(C1); free(C2);
    return ret;
}


int check_gf256v_mul_u256() {
    v256 v1 = {0}, v2 = {0}, v3;
    for (uint32_t i = 0; i < 8; ++i) {
        v1.v32[i] = rand();
        v2.v32[i] = rand();
    }

    for (uint32_t k = 0; k < 32; ++k) {
        for (uint32_t i = 0; i < 256; ++i) {
            for (uint32_t j = 0; j < 256; ++j) {
                v1.v8[k] = i;
                v2.v8[k] = j;

                v3.v256 = gf256v_mul_u32_u256(v1.v256, v2.v256);
                if (v3.v8[k] != gf256_mul(i, j)) {
                    return 1;
                }
                assert(v3.v8[k] == gf256_mul(i, j));
            }
        }
    }
    return 0;
}

int check_gf256v_sqr_u256() {
    v256 sv1 = {0}, sv3;
    sv1.v32[1] = (210<<16) | (22 << 8);
    sv3.v256 = gf256v_squ_u256(sv1.v256);
    printf("%u %lu\n", sv3.v32[1]>>16, gf256v_squ_u64(210));

    return true;
}

int check_gf256v_mul_u256_v2() {
    v256 v1 = {0}, v2 = {0}, v3, v4;
    v1.v32[0] = 121;
    v2.v32[0] = 12;

    v3.v256 = gf256v_mul_u256_v2(v1.v256, v2.v256);
    v4.v256 = gf256v_mul_u32_u256(v1.v256, v2.v256);
    printf("%u %u\n", v3.v32[0], v4.v32[0]);

    if (check_gf256v_mul_u256()) return 1;
    return 0;
}

int check_gf256v_mul_scalar_u256() {
    const __m256i a = _mm256_set1_epi8(1);

    v256 c1 = {0}, c2 = {0};
    for (uint32_t b = 2; b < (1u<<4u); b++) {
        const __m256i b1 = _mm256_set1_epi8(b);
        c1.v256 = gf256v_mul_u256(a, b1);
        c2.v256 = gf256v_mul_scalar_u256(a, b);
        for (uint32_t k = 0; k < 32; k++) {
            if (c1.v8[k] != c2.v8[k]) {
                printf("error\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if(test_matrix_map_gf16_to_gf256_u256()) { return 1; }
    if(test_matrix_add_u256()) { return 1; }
    if(test_matrix_add_gf16_u256()) { return 1; }
    if(test_matrix_add_multiple_gf16_u256()) { return 1; }
    if(test_matrix_add_multiple_2_u256()) { return 1; }
    if(test_matrix_add_multiple_3_u256()) { return 1; }
    if(test_matrix_product_gf16_1_u256()) { return 1; }
    if(test_matrix_product_gf16_2_u256()) { return 1; }
    if(test_matrix_product_8x8xC_u256()) { return 1; }
    if(test_matrix_product_16x16xC_u256()) { return 1; }
    if(test_matrix_product_le32xBxle16_u256()) { return 1; }

    if(test_vector_set_to_gf16_u256()) { return 1; }
    if(test_vector_set_to_gf2_u256()) { return 1; }
    if(test_vector_add_gf2_2_u256()) { return 1; }
    if(test_vector_scalar_gf2_u256()) {}

    if(check_gf256v_mul_scalar_u256()) { return 1; }
    // if(check_gf256v_mul_u256()) { return 1; }
#endif

    printf("All good\n");
	return 0;
}
