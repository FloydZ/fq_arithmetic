#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"

#ifdef USE_AVX2 

uint32_t test_vector_add() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);
    gf2 *B  = gf2_vector_alloc(N);
    gf2 *C1 = gf2_vector_alloc(N);
    gf2 *C2 = gf2_vector_alloc(N);

    gf2_vector_rng(A, N);
    //gf2_vector_rng(B, N);
    gf2_vector_zero(B, N);

    gf2_vector_add(C1, A, B, N);
    gf2_vector_add_u256(C2, A, B, N);

    const uint32_t p = gf2_vector_size(N);
    for (uint32_t i = 0; i < p; i++) {
        if (C1[i] != C2[i]) {
            printf("error test_vector_add\n");
            ret = 1;
            goto finish;
        }
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_vector_scalar() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);
    gf2 *C1 = gf2_vector_alloc(N);
    gf2 *C2 = gf2_vector_alloc(N);

    gf2_vector_rng(A, N);

    gf2 t = 1; 
    gf2_vector_scalar(C1, A, t, N);
    gf2_vector_scalar_u256(C2, A, t, N);

    const uint32_t p = gf2_vector_size(N);
    for (uint32_t i = 0; i < p; i++) {
        if (C1[i] != C2[i]) {
            printf("error test_vector_scalar\n");
            ret = 1;
            goto finish;
        }
    }

finish:
    free(A); free(C1); free(C2);
    return ret;
}

uint32_t test_vector_scalar_add() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);
    gf2 *C1 = gf2_vector_alloc(N);
    gf2 *C2 = gf2_vector_alloc(N);

    gf2_vector_rng(A, N);
    gf2_vector_zero(C1, N);
    gf2_vector_zero(C2, N);

    const gf2 t = 1;
    gf2_vector_scalar_add(C1, A, t, N);
    gf2_vector_scalar_add_u256(C2, A, t, N);


    const uint32_t p = gf2_vector_size(N);
    for (uint32_t i = 0; i < p; i++) {
        if (C1[i] != C2[i]) {
            printf("error test_vector_scalar_add\n");
            ret += 1;
            goto finish;
        }
    }

finish:
    free(A); free(C1); free(C2);
    return ret;
}

uint32_t test_vector_scalar_add_v2() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);
    gf2 *B  = gf2_vector_alloc(N);
    gf2 *C1 = gf2_vector_alloc(N);
    gf2 *C2 = gf2_vector_alloc(N);

    gf2_vector_rng(A, N);
    gf2_vector_rng(B, N);

    const gf2 t = 1;
    gf2_vector_scalar_add_v2(C1, A, t, B, N);
    gf2_vector_scalar_add_u256_v2(C2, A, t, B, N);


    const uint32_t p = gf2_vector_size(N);
    for (uint32_t i = 0; i < p; i++) {
        if (C1[i] != C2[i]) {
            printf("error test_vector_scalar_add_v2\n");
            ret = 1;
            goto finish;
        }
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_vector_eval() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);

    gf2_vector_rng(A, N);

    const gf2 t = 1;
    const gf2 t1 = gf2_vector_eval(A, t, N);
    const gf2 t2 = gf2_vector_eval(A, t, N);

    if (t1 != t2) {
        printf("error test_vector_eval\n");
        ret += 1;
    }

    free(A);
    return ret;
}


uint32_t test_matrix_add() {
    const uint32_t nrows = 16;
    const uint32_t ncols = 16;
    uint32_t ret = 0;

    gf2 *A = gf2_matrix_alloc(nrows, ncols);
    gf2 *B = gf2_matrix_alloc(nrows, ncols);
    gf2 *C1 = gf2_matrix_alloc(nrows, ncols);
    gf2 *C2 = gf2_matrix_alloc(nrows, ncols);

    gf2_matrix_rng(A, nrows, ncols);
    gf2_matrix_rng(B, nrows, ncols);

    gf2_matrix_add(C1, A, B, nrows, ncols);
    gf2_matrix_add_u256(C2, A, B, nrows, ncols);

    const uint32_t p = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t i = 0; i < ncols; i++) {
        for (uint32_t j = 0; j < p; j++) {
            if (C1[i*p + j] != C2[i*p + j]) {
                printf("error test_matrix_add\n");
                ret = 1;
                goto finish;
            }
        } 
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_scalar_add() {
    const uint32_t nrows = 16;
    const uint32_t ncols = 16;
    uint32_t ret = 0;

    gf2 *B = gf2_matrix_alloc(nrows, ncols);
    gf2 *C1 = gf2_matrix_alloc(nrows, ncols);
    gf2 *C2 = gf2_matrix_alloc(nrows, ncols);

    gf2_matrix_rng(B, nrows, ncols);
    gf2_matrix_zero(C1, nrows, ncols);
    gf2_matrix_zero(C2, nrows, ncols);

    const uint8_t scalar = 1;
    gf2_matrix_scalar_add(C1, scalar, B, nrows, ncols);
    gf2_matrix_scalar_add_u256(C2, scalar, B, nrows, ncols);

    const uint32_t p = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t i = 0; i < ncols; i++) {
        for (uint32_t j = 0; j < p; j++) {
            if (C1[i*p + j] != C2[i*p + j]) {
                printf("error test_matrix_scalar_add %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        } 
    }

finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_mul() {
    const uint32_t nrows1 = 16;
    const uint32_t ncols1 = 16;
    const uint32_t ncols2 = 16;
    uint32_t ret= 0;

    gf2 *A  = gf2_matrix_alloc(nrows1, ncols1);
    gf2 *B  = gf2_matrix_alloc(ncols1, ncols2);
    gf2 *C1 = gf2_matrix_alloc(nrows1, ncols2);
    gf2 *C2 = gf2_matrix_alloc(nrows1, ncols2);

    gf2_matrix_rng(A, nrows1, ncols1);
    gf2_matrix_rng(B, nrows1, ncols1);

    gf2_matrix_mul(C1, A, B, nrows1, ncols1, ncols2);
    gf2_matrix_mul_u256(C2, A, B, nrows1, ncols1, ncols2);

    const uint32_t p = gf2_matrix_bytes_per_column(nrows1);
    for (uint32_t i = 0; i < ncols2; i++) {
        for (uint32_t j = 0; j < p; j++) {
            if (C1[i*p + j] != C2[i*p + j]) {
                printf("error test_matrix_mul\n");
                ret = 1;
                goto finish;
            }
        } 
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}



#endif

bool test1() {
	gf2 a = 1, b = 1;
	if (gf2_add(a, b)) { return false; }

	return true;
}

int main() {
	if (!test1()) { return -1; }

#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_scalar()) { return 1; }
    if (test_vector_scalar_add()) { return 1; }
    if (test_vector_scalar_add_v2()) { return 1; }
    if (test_vector_eval()) { return 1; }

    if (test_matrix_add()) { return 1; }
    if (test_matrix_scalar_add()) { return 1; }
    if (test_matrix_mul()) { return 1; }
#endif

	return 0;
}
