#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"
#include "matrix.h"
#include "vector.h"

/// vector lengths (in elements) used by the vector tests
static const uint32_t test_lengths[] = {
    1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
    255, 256, 257, 511, 512, 513, 1000, 1025, 2049
};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// matrix shapes {n_rows1, n_cols1, n_cols2} used by the matrix tests
static const uint32_t test_shapes[][3] = {
    {1, 1, 1}, {3, 5, 2}, {8, 7, 1}, {17, 9, 33}, {33, 40, 7}, {50, 8, 30},
    {64, 64, 64}, {100, 3, 70}, {300, 20, 1}, {257, 65, 9}, {272, 100, 5}
};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

/// fills `n` bytes with random data (including the unused bits)
static void fill_raw(uint8_t *a, const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) { a[i] = rand(); }
}

uint32_t test_arith() {
    for (gf2 a = 0; a < 2; a++) {
        if (gf2_add(a, gf2_neg(a)) != 0) { printf("error test_arith: neg\n"); return 1; }
        if (a && (gf2_mul(a, gf2_inv(a)) != 1)) { printf("error test_arith: inv\n"); return 1; }
        for (gf2 b = 0; b < 2; b++) {
            if (gf2_sub(gf2_add(a, b), b) != a) { printf("error test_arith: sub\n"); return 1; }
            if (gf2_mul(a, b) != (a & b)) { printf("error test_arith: mul\n"); return 1; }
        }
    }
    return 0;
}

/// checks the reference vector functions against their element wise definition
uint32_t test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        const uint32_t nb = gf2_vector_size(n);
        gf2 *a = (gf2 *)malloc(nb), *b = (gf2 *)malloc(nb),
            *r = (gf2 *)malloc(nb), *o = (gf2 *)malloc(nb);
        fill_raw(a, nb); fill_raw(b, nb); fill_raw(r, nb);
        memcpy(o, r, nb);

        for (gf2 c = 0; c < 2; c++) {
            memcpy(r, o, nb);
            gf2_vector_scalar_add(r, c, a, n);
            for (uint32_t i = 0; i < 8*nb; i++) {
                const gf2 e = (i < n) ? (gf2_vector_get(o, i) ^ (c & gf2_vector_get(a, i)))
                                      : gf2_vector_get(o, i);
                if (gf2_vector_get(r, i) != e) {
                    printf("error test_vector_ref: scalar_add %u %u\n", n, i);
                    return 1;
                }
            }

            memcpy(r, o, nb);
            gf2_vector_scalar_add_v2(r, a, c, b, n);
            for (uint32_t i = 0; i < 8*nb; i++) {
                const gf2 e = (i < n) ? (gf2_vector_get(a, i) ^ (c & gf2_vector_get(b, i)))
                                      : gf2_vector_get(o, i);
                if (gf2_vector_get(r, i) != e) {
                    printf("error test_vector_ref: scalar_add_v2 %u %u\n", n, i);
                    return 1;
                }
            }
        }

        memcpy(r, o, nb);
        gf2_vector_add(r, a, n);
        gf2 acc = 0;
        for (uint32_t i = 0; i < 8*nb; i++) {
            const gf2 e = (i < n) ? (gf2_vector_get(o, i) ^ gf2_vector_get(a, i))
                                  : gf2_vector_get(o, i);
            if (gf2_vector_get(r, i) != e) {
                printf("error test_vector_ref: add %u %u\n", n, i);
                return 1;
            }
            if (i < n) { acc ^= gf2_vector_get(a, i) & gf2_vector_get(b, i); }
        }

        if (gf2_vector_mul_acc(a, b, n) != acc) {
            printf("error test_vector_ref: mul_acc %u\n", n);
            return 1;
        }

        memcpy(r, o, nb);
        gf2_vector_copy(r, a, n);
        for (uint32_t i = 0; i < 8*nb; i++) {
            const gf2 e = (i < n) ? gf2_vector_get(a, i) : gf2_vector_get(o, i);
            if (gf2_vector_get(r, i) != e) {
                printf("error test_vector_ref: copy %u %u\n", n, i);
                return 1;
            }
        }

        free(a); free(b); free(r); free(o);
    }
    return 0;
}

typedef void (*vec_add_t)(gf2 *, const gf2 *, uint32_t);
typedef void (*vec_add_v2_t)(gf2 *, const gf2 *, const gf2 *, uint32_t);
typedef void (*vec_scalar_add_t)(gf2 *, gf2, const gf2 *, uint32_t);
typedef void (*vec_scalar_add_v2_t)(gf2 *, const gf2 *, gf2, const gf2 *, uint32_t);
typedef gf2  (*vec_mul_acc_t)(const gf2 *, const gf2 *, uint32_t);

/// compares the simd vector functions against the reference functions
uint32_t test_vector_simd(const char *name,
                          vec_add_t add,
                          vec_add_v2_t add_v2,
                          vec_scalar_add_t scalar_add,
                          vec_scalar_add_v2_t scalar_add_v2,
                          vec_mul_acc_t mul_acc) {
    uint32_t ret = 0;
    for (uint32_t l = 0; (l < NR_TEST_LENGTHS) && (ret == 0); l++) {
        const uint32_t n = test_lengths[l];
        const uint32_t nb = gf2_vector_size(n);
        // exact sizes, s.t. the address sanitizer catches out of bound accesses
        gf2 *a  = (gf2 *)malloc(nb), *b  = (gf2 *)malloc(nb),
            *r1 = (gf2 *)malloc(nb), *r2 = (gf2 *)malloc(nb);
        fill_raw(a, nb); fill_raw(b, nb); fill_raw(r1, nb);
        memcpy(r2, r1, nb);

        gf2_vector_add(r1, a, n);
        add(r2, a, n);
        if (memcmp(r1, r2, nb)) { printf("error %s: add %u\n", name, n); ret = 1; }

        gf2_vector_add_v2(r1, a, b, n);
        add_v2(r2, a, b, n);
        if (memcmp(r1, r2, nb)) { printf("error %s: add_v2 %u\n", name, n); ret = 1; }

        for (gf2 c = 0; c < 2; c++) {
            gf2_vector_scalar_add(r1, c, a, n);
            scalar_add(r2, c, a, n);
            if (memcmp(r1, r2, nb)) { printf("error %s: scalar_add %u\n", name, n); ret = 1; }

            gf2_vector_scalar_add_v2(r1, b, c, a, n);
            scalar_add_v2(r2, b, c, a, n);
            if (memcmp(r1, r2, nb)) { printf("error %s: scalar_add_v2 %u\n", name, n); ret = 1; }
        }

        if (gf2_vector_mul_acc(a, b, n) != mul_acc(a, b, n)) {
            printf("error %s: mul_acc %u\n", name, n);
            ret = 1;
        }

        free(a); free(b); free(r1); free(r2);
    }
    return ret;
}

/// checks the reference matrix functions: id*A == A, (A^T)^T == A,
/// transpose against get/set, add_mul == mul + add, add_scalar, copy
uint32_t test_matrix_ref() {
    uint32_t ret = 0;
    for (uint32_t s = 0; (s < NR_TEST_SHAPES) && (ret == 0); s++) {
        const uint32_t nrows = test_shapes[s][0],
                       ncols = test_shapes[s][1],
                       ncols2 = test_shapes[s][2];
        gf2 *I  = gf2_matrix_alloc(nrows, nrows);
        gf2 *A  = gf2_matrix_alloc(nrows, ncols);
        gf2 *B  = gf2_matrix_alloc(ncols, ncols2);
        gf2 *C1 = gf2_matrix_alloc(nrows, ncols2);
        gf2 *C2 = gf2_matrix_alloc(nrows, ncols2);
        gf2 *C3 = gf2_matrix_alloc(nrows, ncols2);
        gf2 *T  = gf2_matrix_alloc(ncols, nrows);
        gf2 *A2 = gf2_matrix_alloc(nrows, ncols);

        gf2_matrix_random(A, nrows, ncols);
        gf2_matrix_random(B, ncols, ncols2);
        gf2_matrix_random(C1, nrows, ncols2);

        // id * A == A
        gf2_matrix_id(I, nrows, nrows);
        gf2_matrix_mul(A2, I, A, nrows, nrows, ncols);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                if (gf2_matrix_get(A2, nrows, i, j) != gf2_matrix_get(A, nrows, i, j)) {
                    printf("error test_matrix_ref: id %u %u\n", nrows, ncols);
                    ret = 1; goto finish;
                }
            }
        }

        // transpose
        gf2_matrix_transpose(T, A, nrows, ncols);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                if (gf2_matrix_get(T, ncols, j, i) != gf2_matrix_get(A, nrows, i, j)) {
                    printf("error test_matrix_ref: transpose %u %u\n", nrows, ncols);
                    ret = 1; goto finish;
                }
            }
        }
        gf2_matrix_zero(A2, nrows, ncols);
        gf2_matrix_transpose(A2, T, ncols, nrows);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                if (gf2_matrix_get(A2, nrows, i, j) != gf2_matrix_get(A, nrows, i, j)) {
                    printf("error test_matrix_ref: transpose^2 %u %u\n", nrows, ncols);
                    ret = 1; goto finish;
                }
            }
        }

        // C2 = C1 + A*B, C3 = C1; C3 += A*B
        gf2_matrix_mul(C2, A, B, nrows, ncols, ncols2);
        gf2_matrix_add(C2, C2, C1, nrows, ncols2);
        gf2_matrix_copy(C3, C1, nrows, ncols2);
        gf2_matrix_add_mul(C3, A, B, nrows, ncols, ncols2);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols2; j++) {
                if (gf2_matrix_get(C2, nrows, i, j) != gf2_matrix_get(C3, nrows, i, j)) {
                    printf("error test_matrix_ref: add_mul %u %u %u\n", nrows, ncols, ncols2);
                    ret = 1; goto finish;
                }
            }
        }

        // C3 = C1 + 1*C2 == C1 + C1 + A*B == A*B
        gf2_matrix_copy(C3, C1, nrows, ncols2);
        gf2_matrix_add_scalar(C3, 1, C2, nrows, ncols2);
        gf2_matrix_mul(C2, A, B, nrows, ncols, ncols2);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols2; j++) {
                if (gf2_matrix_get(C2, nrows, i, j) != gf2_matrix_get(C3, nrows, i, j)) {
                    printf("error test_matrix_ref: add_scalar %u %u %u\n", nrows, ncols, ncols2);
                    ret = 1; goto finish;
                }
            }
        }

    finish:
        free(I); free(A); free(B); free(C1); free(C2); free(C3); free(T); free(A2);
    }
    return ret;
}

typedef void (*mat_add_t)(gf2 *, const gf2 *, const gf2 *, uint32_t, uint32_t);
typedef void (*mat_add_scalar_t)(gf2 *, gf2, const gf2 *, uint32_t, uint32_t);
typedef void (*mat_mul_t)(gf2 *, const gf2 *, const gf2 *, uint32_t, uint32_t, uint32_t);

/// \return 0 if m1 == m2
static uint32_t matrix_cmp(const gf2 *m1, const gf2 *m2, const uint32_t nrows, const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            if (gf2_matrix_get(m1, nrows, i, j) != gf2_matrix_get(m2, nrows, i, j)) { return 1; }
        }
    }
    return 0;
}

/// compares the simd matrix functions against the reference functions
uint32_t test_matrix_simd(const char *name,
                          mat_add_t add,
                          mat_add_scalar_t add_scalar,
                          mat_mul_t mul,
                          mat_mul_t add_mul) {
    uint32_t ret = 0;
    for (uint32_t s = 0; (s < NR_TEST_SHAPES) && (ret == 0); s++) {
        const uint32_t nrows = test_shapes[s][0],
                       ncols = test_shapes[s][1],
                       ncols2 = test_shapes[s][2];
        gf2 *A  = gf2_matrix_alloc(nrows, ncols);
        gf2 *A2 = gf2_matrix_alloc(nrows, ncols);
        gf2 *B  = gf2_matrix_alloc(ncols, ncols2);
        gf2 *C1 = gf2_matrix_alloc(nrows, ncols2);
        gf2 *C2 = gf2_matrix_alloc(nrows, ncols2);
        gf2 *D1 = gf2_matrix_alloc(nrows, ncols);
        gf2 *D2 = gf2_matrix_alloc(nrows, ncols);

        gf2_matrix_random(A, nrows, ncols);
        gf2_matrix_random(A2, nrows, ncols);
        gf2_matrix_random(B, ncols, ncols2);

        gf2_matrix_add(D1, A, A2, nrows, ncols);
        add(D2, A, A2, nrows, ncols);
        if (matrix_cmp(D1, D2, nrows, ncols)) { printf("error %s: add %u %u\n", name, nrows, ncols); ret = 1; }

        for (gf2 c = 0; c < 2; c++) {
            gf2_matrix_add_scalar(D1, c, A, nrows, ncols);
            add_scalar(D2, c, A, nrows, ncols);
            if (matrix_cmp(D1, D2, nrows, ncols)) { printf("error %s: add_scalar %u %u\n", name, nrows, ncols); ret = 1; }
        }

        gf2_matrix_mul(C1, A, B, nrows, ncols, ncols2);
        gf2_matrix_random(C2, nrows, ncols2);
        mul(C2, A, B, nrows, ncols, ncols2);
        if (matrix_cmp(C1, C2, nrows, ncols2)) { printf("error %s: mul %u %u %u\n", name, nrows, ncols, ncols2); ret = 1; }

        gf2_matrix_random(C1, nrows, ncols2);
        gf2_matrix_copy(C2, C1, nrows, ncols2);
        gf2_matrix_add_mul(C1, A, B, nrows, ncols, ncols2);
        add_mul(C2, A, B, nrows, ncols, ncols2);
        if (matrix_cmp(C1, C2, nrows, ncols2)) { printf("error %s: add_mul %u %u %u\n", name, nrows, ncols, ncols2); ret = 1; }

        free(A); free(A2); free(B); free(C1); free(C2); free(D1); free(D2);
    }
    return ret;
}

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

/// The m4ri style transposes below work on row major bit matrices: row r
/// starts at `M + r*stride`, column c is bit (c%8) of byte c/8 of the row.
/// The buffers have enough slack for the word sized accesses of the kernels.
#define RM_BUF_SIZE (64*8 + 64)

/// \return M[r, c]
static gf2 rm_get(const uint8_t *M, const uint32_t stride, const uint32_t r, const uint32_t c) {
    return (M[r*stride + c/8] >> (c%8)) & 1u;
}

/// \return a zeroed buffer
static uint8_t *rm_alloc(void) {
    return (uint8_t *)calloc(RM_BUF_SIZE, 1);
}

/// sets the first `ncols` bits of each row randomly, all other bits to zero
static void rm_random(uint8_t *M, const uint32_t stride, const uint32_t nrows, const uint32_t ncols) {
    memset(M, 0, RM_BUF_SIZE);
    for (uint32_t r = 0; r < nrows; r++) {
        for (uint32_t c = 0; c < ncols; c++) {
            M[r*stride + c/8] |= (rand() & 1u) << (c%8);
        }
    }
}

/// \return 0 if B (m x n, stride sb) == A^T (A: n x m, stride sa)
static uint32_t rm_check_transpose(const char *name,
                                   const uint8_t *A, const uint32_t sa,
                                   const uint8_t *B, const uint32_t sb,
                                   const uint32_t n, const uint32_t m) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            if (rm_get(A, sa, i, j) != rm_get(B, sb, j, i)) {
                printf("error: %s n=%u m=%u at (%u,%u)\n", name, n, m, i, j);
                return 1;
            }
        }
    }
    return 0;
}

uint32_t test_matrix_transpose_le8xle8() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    for (uint32_t nrows = 1; (nrows <= 8) && (ret == 0); nrows++) {
        for (uint32_t ncols = 1; (ncols <= 8) && (ret == 0); ncols++) {
            rm_random(A, 1, nrows, ncols);
            memset(B, 0, RM_BUF_SIZE);
            gf2_matrix_transpose_le8xle8(B, A, 1, 1, nrows, ncols, MAX(nrows, ncols));
            ret = rm_check_transpose("test_matrix_transpose_le8xle8", A, 1, B, 1, nrows, ncols);
        }
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_le16xle16() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    for (uint32_t nrows = 9; (nrows <= 16) && (ret == 0); nrows++) {
        for (uint32_t ncols = 9; (ncols <= 16) && (ret == 0); ncols++) {
            rm_random(A, 2, nrows, ncols);
            memset(B, 0, RM_BUF_SIZE);
            gf2_matrix_transpose_le16xle16(B, A, 2, 2, nrows, ncols, MAX(nrows, ncols));
            ret = rm_check_transpose("test_matrix_transpose_le16xle16", A, 2, B, 2, nrows, ncols);
        }
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_le32xle32() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    for (uint32_t nrows = 17; (nrows <= 32) && (ret == 0); nrows++) {
        for (uint32_t ncols = 17; (ncols <= 32) && (ret == 0); ncols++) {
            rm_random(A, 4, nrows, ncols);
            memset(B, 0, RM_BUF_SIZE);
            gf2_matrix_transpose_le32xle32(B, A, 4, 4, nrows, ncols);
            ret = rm_check_transpose("test_matrix_transpose_le32xle32", A, 4, B, 4, nrows, ncols);
        }
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_64x64() {
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    rm_random(A, 8, 64, 64);
    gf2_matrix_transpose_64x64(B, A, 8, 8);
    const uint32_t ret = rm_check_transpose("test_matrix_transpose_64x64", A, 8, B, 8, 64, 64);
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_le64xle64() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    for (uint32_t nrows = 1; (nrows <= 64) && (ret == 0); nrows++) {
        for (uint32_t ncols = 1; (ncols <= 64) && (ret == 0); ncols++) {
            rm_random(A, 8, nrows, ncols);
            memset(B, 0, RM_BUF_SIZE);
            gf2_matrix_transpose_le64xle64(B, A, 8, 8, nrows, ncols);
            ret = rm_check_transpose("test_matrix_transpose_le64xle64", A, 8, B, 8, nrows, ncols);
        }
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_64xle64() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    const uint32_t nrows = 64;
    for (uint32_t ncols = 1; (ncols <= 64) && (ret == 0); ncols++) {
        rm_random(A, 8, nrows, ncols);
        memset(B, 0, RM_BUF_SIZE);
        gf2_matrix_transpose_64xle64(B, A, 8, 8, ncols);
        ret = rm_check_transpose("test_matrix_transpose_64xle64", A, 8, B, 8, nrows, ncols);
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_le64x64() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    const uint32_t ncols = 64;
    for (uint32_t nrows = 1; (nrows <= 64) && (ret == 0); nrows++) {
        rm_random(A, 8, nrows, ncols);
        memset(B, 0, RM_BUF_SIZE);
        gf2_matrix_transpose_le64x64(B, A, 8, 8, nrows);
        ret = rm_check_transpose("test_matrix_transpose_le64x64", A, 8, B, 8, nrows, ncols);
    }
    free(A); free(B);
    return ret;
}

uint32_t test_matrix_transpose_small() {
    uint32_t ret = 0;
    uint8_t *A = rm_alloc(), *B = rm_alloc();
    for (uint32_t nrows = 1; (nrows <= 64) && (ret == 0); nrows++) {
        const uint32_t ncols = nrows;
        const uint32_t stride = (nrows + 7u) / 8u;
        rm_random(A, stride, nrows, ncols);
        memset(B, 0, RM_BUF_SIZE);
        gf2_matrix_transpose_small(B, A, stride, stride, nrows, ncols, MAX(nrows, ncols));
        ret = rm_check_transpose("test_matrix_transpose_small", A, stride, B, stride, nrows, ncols);
    }
    free(A); free(B);
    return ret;
}

/// tests the canonical (column major) transpose for medium sized matrices
uint32_t test_matrix_transpose_middle() {
    uint32_t ret = 0;
    for (uint32_t nrows = 64; (nrows <= 512) && (ret == 0); nrows += 13) {
        const uint32_t ncols = (nrows * 3) / 2 + 1;
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(ncols, nrows);
        gf2_matrix_random(A, nrows, ncols);
        gf2_matrix_transpose(B, A, nrows, ncols);

        for (uint32_t i = 0; (i < nrows) && (ret == 0); i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                if (gf2_matrix_get(A, nrows, i, j) != gf2_matrix_get(B, ncols, j, i)) {
                    printf("error test_matrix_transpose_middle nrows: %d, ncols: %d\n", nrows, ncols);
                    ret = 1;
                    break;
                }
            }
        }
        free(A); free(B);
    }
    return ret;
}

/// tests the strided (block based) transpose
uint32_t test_matrix_transpose() {
    const uint32_t shapes[][2] = {{144, 448}, {200, 100}, {64, 64}, {65, 130}, {13, 77}, {128, 9}};
    uint32_t ret = 0;
    for (uint32_t s = 0; (s < sizeof(shapes)/sizeof(shapes[0])) && (ret == 0); s++) {
        const uint32_t nrows = shapes[s][0], ncols = shapes[s][1];
        gf2 *A = gf2_matrix_alloc(nrows, ncols);
        gf2 *B = gf2_matrix_alloc(ncols, nrows);
        gf2_matrix_random(A, nrows, ncols);

        gf2_matrix_transpose_strided(B, A, gf2_matrix_bytes_per_column(ncols),
                                     gf2_matrix_bytes_per_column(nrows), nrows, ncols);

        for (uint32_t i = 0; (i < nrows) && (ret == 0); i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                const gf2 ta = gf2_matrix_get(A, nrows, i, j);
                const gf2 tb = gf2_matrix_get(B, ncols, j, i);
                if (ta != tb) {
                    printf("error: test_matrix_transpose: %d %d (%d x %d)\n", i, j, nrows, ncols);
                    ret = 1;
                    break;
                }
            }
        }

        free(A); free(B);
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
    gf2_vector_random(B, N);

    gf2_vector_add_v2(C1, A, B, N);
    gf2_vector_add_u256_v2(C2, A, B, N);

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
    gf2_vector_scalar_add(C1, t, A, N);
    gf2_vector_scalar_add_u256(C2, t, A, N);


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
    uint32_t ret = 0;
    const uint32_t lengths[] = {1, 7, 100, 256, 257, 1000};

    for (uint32_t l = 0; l < sizeof(lengths)/sizeof(lengths[0]); l++) {
        const uint32_t N = lengths[l];
        gf2 *A  = gf2_vector_alloc(N);
        gf2_vector_random(A, N);

        for (gf2 t = 0; t < 2; t++) {
            const gf2 t1 = gf2_vector_eval(A, t, N);
            const gf2 t2 = gf2_vector_eval_u256(A, t, N);

            if (t1 != t2) {
                printf("error test_vector_eval: %d %d\n", t1, t2);
                ret += 1;
            }
        }

        free(A);
    }
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

uint32_t test_matrix_add_scalar() {
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
    gf2_matrix_add_scalar(C1, scalar, B, nrows, ncols);
    gf2_matrix_add_scalar_u256(C2, scalar, B, nrows, ncols);

    const uint32_t p = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t i = 0; i < ncols; i++) {
        for (uint32_t j = 0; j < p; j++) {
            if (C1[i*p + j] != C2[i*p + j]) {
                printf("error test_matrix_add_scalar %d %d\n", i, j);
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
    const uint32_t nrows1 = 272; // 144; // 321
    const uint32_t ncols1 = 892; // 448; //1443
    const uint32_t ncols2 = 256; // 128;
    uint32_t ret= 0;

    gf2 *A  = gf2_matrix_alloc(nrows1, ncols1);
    gf2 *B  = gf2_matrix_alloc(ncols1, ncols2);
    gf2 *C1 = gf2_matrix_alloc(nrows1, ncols2);
    gf2 *C2 = gf2_matrix_alloc(nrows1, ncols2);
    gf2 *C3 = gf2_matrix_alloc(nrows1, ncols2);

    gf2_matrix_random(A, nrows1, ncols1);
    gf2_matrix_random(B, ncols1, ncols2);

    gf2_matrix_mul(C1, A, B, nrows1, ncols1, ncols2);
    gf2_matrix_mul_transpose(C2, A, B, nrows1, ncols1, ncols2);
    gf2_matrix_mul_u256(C3, A, B, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const uint8_t t1 = gf2_matrix_get(C1, nrows1, i, j);
            const uint8_t t2 = gf2_matrix_get(C2, nrows1, i, j);
            const uint8_t t3 = gf2_matrix_get(C3, nrows1, i, j);
            if ((t1 != t2) || (t1 != t3)) {
                printf("error test_matrix_mul: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(A); free(B); free(C1); free(C2); free(C3);
    return ret;
}

#endif

int main() {
    srand(0x2);
    if (test_arith()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }

    if (test_matrix_m4ri()) { return 1; }
    if (test_matrix_transpose_le8xle8()) { return 1; }
    if (test_matrix_transpose_le16xle16()) { return 1; }
    if (test_matrix_transpose_le32xle32()) { return 1; }
    if (test_matrix_transpose_64x64()) { return 1; }
    if (test_matrix_transpose_le64xle64()) { return 1; }
    if (test_matrix_transpose_64xle64()) { return 1; }
    if (test_matrix_transpose_le64x64()) { return 1; }
    if (test_matrix_transpose()) { return 1; }
    if (test_matrix_transpose_small()) { return 1; }
    if (test_matrix_transpose_middle()) { return 1; }

#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_scalar()) { return 1; }
    if (test_vector_scalar_add()) { return 1; }
    if (test_vector_scalar_add_v2()) { return 1; }
    if (test_vector_eval()) { return 1; }

    if (test_matrix_add()) { return 1; }
    if (test_matrix_add_scalar()) { return 1; }
    if (test_matrix_mul()) { return 1; }

    if (test_vector_simd("vector_u256", gf2_vector_add_u256, gf2_vector_add_u256_v2,
                         gf2_vector_scalar_add_u256, gf2_vector_scalar_add_u256_v2,
                         gf2_vector_mul_acc_u256)) { return 1; }
    if (test_matrix_simd("matrix_u256", gf2_matrix_add_u256, gf2_matrix_add_scalar_u256,
                         gf2_matrix_mul_u256, gf2_matrix_add_mul_u256)) { return 1; }
#endif

#ifdef USE_NEON
    if (test_vector_simd("vector_u128", gf2_vector_add_u128, gf2_vector_add_u128_v2,
                         gf2_vector_scalar_add_u128, gf2_vector_scalar_add_u128_v2,
                         gf2_vector_mul_acc_u128)) { return 1; }
    if (test_matrix_simd("matrix_u128", gf2_matrix_add_u128, gf2_matrix_add_scalar_u128,
                         gf2_matrix_mul_u128, gf2_matrix_add_mul_u128)) { return 1; }
#endif

#ifdef USE_AVX512
    if (test_vector_simd("vector_u512", gf2_vector_add_u512, gf2_vector_add_u512_v2,
                         gf2_vector_scalar_add_u512, gf2_vector_scalar_add_u512_v2,
                         gf2_vector_mul_acc_u512)) { return 1; }
    if (test_matrix_simd("matrix_u512", gf2_matrix_add_u512, gf2_matrix_add_scalar_u512,
                         gf2_matrix_mul_u512, gf2_matrix_add_mul_u512)) { return 1; }
#endif

    printf("all done!\n");
	return 0;
}
