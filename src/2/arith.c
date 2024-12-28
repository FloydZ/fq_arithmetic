#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "matrix.h"
#include "vector.h"

uint32_t test_matrix_m4ri() {
    uint32_t ret = 0;
    const uint32_t nrows = 10, 
                   ncols = 10;
    const uint32_t k = 2;
    gf2 *A = gf2_matrix_alloc(nrows, ncols);
    gf2_matrix_rng_full_rank(A, nrows, ncols, 1);
    gf2_matrix_echelonize(A, nrows, ncols, k);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < nrows; j++) {
            const gf2 t = gf2_matrix_get(A, ncols, i, j);
            if (t != (i == j)) {
                printf("error: test_m4ri_gaus\n");
                gf2_matrix_print_row_major(A, nrows, ncols);
                printf("\n");
                gf2_matrix_print(A, nrows, ncols);

                ret = 1;
                goto finish;
            }
        }
    }

finish: 
    free(A);
    return ret;
}

uint32_t test_matrix_transpose_le8xle8() {
    uint32_t ret = 0;
    for (uint32_t nrows = 8; nrows <= 8; nrows ++) {
        // const uint32_t ncols = nrows;
        for (uint32_t ncols = 1; ncols <= 8; ncols ++) {
            gf2 *A = gf2_matrix_alloc(nrows, ncols);
            gf2 *B = gf2_matrix_alloc(nrows, ncols);
            gf2_matrix_random(A, nrows, ncols);

            const uint32_t m = MAX(nrows, ncols);
            gf2_matrix_transpose_le8xle8(B, A, 1, 1, nrows, ncols, m);

            for (uint32_t i = 0; i < nrows; i++) {
                for (uint32_t j = 0; j < ncols; j++) {
                    const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                    const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                    if (ta != tb) {
                        printf("error: test_matrix_transpose_le8xle8\n");
                        gf2_matrix_print(A, nrows, ncols);
                        printf("\n");
                        gf2_matrix_print(B, ncols, nrows);
                        ret = 1;
                        goto finish;
                    }
                }
            }

        finish: 
            free(A);
            free(B);
        }
    }
    return ret;
}

uint32_t test_matrix_transpose_le16xle16() {
    uint32_t ret = 0;
    for (uint32_t nrows = 9; nrows <= 16; nrows ++) {
        // const uint32_t ncols = nrows;
        for (uint32_t ncols = 9; ncols <= 16; ncols ++) {
            gf2 *A = gf2_matrix_alloc(nrows, ncols);
            gf2 *B = gf2_matrix_alloc(nrows, ncols);
            gf2_matrix_random(A, nrows, ncols);

            gf2_matrix_transpose_le16xle16(B, A, 2, 2, nrows, ncols, nrows);

            for (uint32_t i = 0; i < nrows; i++) {
                for (uint32_t j = 0; j < ncols; j++) {
                    const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                    const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                    if (ta != tb) {
                        printf("error: test_matrix_transpose_le16xle16\n");
                        gf2_matrix_print(A, nrows, ncols);
                        printf("\n");
                        gf2_matrix_print(B, nrows, ncols);
                        ret = 1;
                        goto finish;
                    }
                }
            }

        finish: 
            free(A);
            free(B);
        }
    }
    return ret;
}

uint32_t test_matrix_transpose_le32xle32() {
    uint32_t ret = 0;
    for (uint32_t nrows = 17; nrows <= 32; nrows ++) {
        // const uint32_t ncols = nrows;
        for (uint32_t ncols = 17; ncols <= 32; ncols ++) {
            gf2 *A = gf2_matrix_alloc(nrows, ncols);
            gf2 *B = gf2_matrix_alloc(nrows, ncols);
            gf2_matrix_random(A, nrows, ncols);

            gf2_matrix_transpose_le32xle32(B, A, 4, 4, nrows, ncols);

            for (uint32_t i = 0; i < nrows; i++) {
                for (uint32_t j = 0; j < ncols; j++) {
                    const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                    const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                    if (ta != tb) {
                        printf("error: test_matrix_transpose_le32xle32\n");
                        gf2_matrix_print(A, nrows, ncols);
                        printf("\n");
                        gf2_matrix_print(B, nrows, ncols);
                        ret = 1;
                        goto finish;
                    }
                }
            }

        finish: 
            free(A);
            free(B);
        }
    }
    return ret;
}

uint32_t test_matrix_transpose_64x64() {
    uint32_t ret = 0, nrows=64, ncols=64;
    gf2 *A = gf2_matrix_alloc(nrows, ncols);
    gf2 *B = gf2_matrix_alloc(nrows, ncols);
    gf2_matrix_random(A, nrows, ncols);

    gf2_matrix_transpose_64x64(B, A, 8, 8);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf2 ta = gf2_matrix_get(A, ncols, i, j);
            const gf2 tb = gf2_matrix_get(B, nrows, j, i);
            if (ta != tb) {
                printf("error: test_matrix_transpose_64x64: %d %d\n", i, j);
                gf2_matrix_print(A, nrows, ncols);
                printf("\n");
                gf2_matrix_print(B, nrows, ncols);
                printf("\n");

                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(A);
    free(B);
    return ret;
}

uint32_t test_matrix_transpose_le64xle64() {
    uint32_t ret = 0;
    for (uint32_t nrows = 33; nrows <= 64; nrows ++) {
        //const uint32_t ncols = nrows;
        for (uint32_t ncols = 33; ncols <= 64; ncols ++) {
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(nrows, ncols);
        gf2_matrix_random(A, nrows, ncols);
        gf2_matrix_transpose_le64xle64(B, A, 8, 8, nrows, ncols);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose_le64xle64\n");
                    gf2_matrix_print(A, nrows, ncols);
                    printf("\n");
                    gf2_matrix_print(B, nrows, ncols);

                    ret = 1;
                    goto finish;
                }
            }
        }

        finish:
        free(A);
        free(B);
        }
    }
    return ret;
}

uint32_t test_matrix_transpose_64xle64() {
    uint32_t ret = 0;
    const uint32_t nrows = 64;
    for (uint32_t ncols = 1; ncols <= 64; ncols ++) {
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(ncols, nrows);
        gf2_matrix_id(A, nrows, ncols);
        const uint32_t cstride = (ncols + 7) / 8;
        gf2_matrix_transpose_64xle64(B, A, cstride, 8, ncols);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                const gf2 ta = gf2_matrix_get(A, nrows, i, j);
                const gf2 tb = gf2_matrix_get(B, ncols, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose_64xle64 %d: %d %d\n", ncols, i, j);
                    gf2_matrix_print(A, nrows, ncols);
                    printf("\n");
                    gf2_matrix_print(B, ncols, nrows);
                    printf("\n");

                    ret = 1;
                    goto finish;
                }
            }
        }

        finish:
        free(A);
        free(B);
    }
    return ret;
}

uint32_t test_matrix_transpose_le64x64() {
    uint32_t ret = 0;
    const uint32_t ncols = 64;
    for (uint32_t nrows = 33; nrows <= 64; nrows ++) {
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(ncols, nrows);
        gf2_matrix_id(A, nrows, ncols);
        const uint32_t cstride = (nrows + 7) / 8;
        gf2_matrix_transpose_le64x64(B, A, 8, cstride, nrows);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                const gf2 ta = gf2_matrix_get(A, nrows, i, j);
                const gf2 tb = gf2_matrix_get(B, ncols, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose_le64x64 %d: %d %d\n", nrows, i, j);
                    gf2_matrix_print(A, nrows, ncols);
                    printf("\n");
                    gf2_matrix_print(B, ncols, nrows);
                    printf("\n");

                    ret = 1;
                    goto finish;
                }
            }
        }

        finish:
        free(A);
        free(B);
    }
    return ret;
}

uint32_t test_matrix_transpose_small() {
    uint32_t ret = 0;
    for (uint32_t nrows = 8; nrows <= 64; nrows ++) {
        // for (uint32_t ncols = 0; ncols <= 8; ncols ++) {
        const uint32_t ncols = nrows;
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(nrows, ncols);
        gf2_matrix_random(A, nrows, ncols);

        const uint32_t maxsize = MAX(nrows, ncols);
        const uint32_t stride = (nrows + 7u) / 8u;
        gf2_matrix_transpose_small(B, A, stride, stride, nrows, ncols, maxsize);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < nrows; j++) {
                const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose_small nrows: %d, ncols: %d\n", nrows, ncols);
                    gf2_matrix_print(A, nrows, ncols);
                    printf("\n");
                    gf2_matrix_print(B, nrows, ncols);
                    ret = 1;
                    goto finish;
                }
            }
        }
        finish:
        free(A);
        free(B);
        // }
    }
    return ret;
}

uint32_t test_matrix_transpose_middle() {
    uint32_t ret = 0;
    for (uint32_t nrows = 64; nrows <= 512; nrows ++) {
        // for (uint32_t ncols = 0; ncols <= 8; ncols ++) {
        const uint32_t ncols = nrows;
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(nrows, ncols);
        gf2_matrix_random(A, nrows, ncols);

        const uint32_t maxsize = MAX(nrows, ncols);
        const uint32_t stride = (nrows + 7u) / 8u;
        gf2_matrix_transpose_small(B, A, stride, stride, nrows, ncols, maxsize);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < nrows; j++) {
                const gf2 ta = gf2_matrix_get(A, ncols, i, j);
                const gf2 tb = gf2_matrix_get(B, nrows, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose_middle nrows: %d, ncols: %d\n", nrows, ncols);
                    gf2_matrix_print(A, nrows, ncols);
                    printf("\n");
                    gf2_matrix_print(B, nrows, ncols);
                    ret = 1;
                    goto finish;
                }
            }
        }
    finish:
        free(A);
        free(B);
        // }
    }
    return ret;
}

#ifdef USE_AVX2

uint32_t test_vector_add() {
    const uint32_t N = 100;
    uint32_t ret = 0;

    gf2 *A  = gf2_vector_alloc(N);
    gf2 *B  = gf2_vector_alloc(N);
    gf2 *C1 = gf2_vector_alloc(N);
    gf2 *C2 = gf2_vector_alloc(N);

    gf2_vector_random(A, N);
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

    gf2_vector_random(A, N);

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

    gf2_vector_random(A, N);
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

    gf2_vector_random(A, N);
    gf2_vector_random(B, N);

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
    gf2_vector_random(A, N);

    const gf2 t = 1;
    const gf2 t1 = gf2_vector_eval(A, t, N);
    const gf2 t2 = gf2_vector_eval_u256(A, t, N);

    if (t1 != t2) {
        printf("error test_vector_eval: %d %d\n", t1, t2);
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

    gf2_matrix_random(A, nrows, ncols);
    gf2_matrix_random(B, nrows, ncols);

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

    gf2_matrix_random(B, nrows, ncols);
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

    gf2_matrix_random(A, nrows1, ncols1);
    gf2_matrix_random(B, nrows1, ncols1);

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

int main() {
    if (test_matrix_m4ri()) { return 1; }
    // if (test_matrix_transpose_le8xle8()) { return 1; }
    if (test_matrix_transpose_le16xle16()) { return 1; }
    if (test_matrix_transpose_le32xle32()) { return 1; }
    if (test_matrix_transpose_64x64()) { return 1; }
    if (test_matrix_transpose_le64xle64()) { return 1; }
    if (test_matrix_transpose_64xle64()) { return 1; }
    if (test_matrix_transpose_le64x64()) { return 1; }
    // TODO if (test_matrix_transpose_small()) { return 1; }
    // TODO if (test_matrix_transpose_middle()) { return 1; }

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

    printf("all done!\n");
	return 0;
}
