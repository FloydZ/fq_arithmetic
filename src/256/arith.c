#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"
#include "bitslice.h"

#include "../16/vector.h"
#include "../2/matrix.h"

/// vector lengths used to test the vectorized functions
static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// matrix shapes (nrows1, ncols1, ncols2) used to test the matrix functions
static const uint32_t test_shapes[][3] = {
    {1,1,1}, {3,5,2}, {8,8,1}, {17,9,4}, {33,40,7}, {50,5,45}, {64,3,9}, {65,16,3}, {16,13,15},
};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

typedef void  (*vec_add_fn)(gf256 *, const gf256 *, const uint32_t);
typedef void  (*vec_add_v2_fn)(gf256 *, const gf256 *, const gf256 *, const uint32_t);
typedef void  (*vec_scalar_add_fn)(gf256 *, const gf256, const gf256 *, const uint32_t);
typedef void  (*vec_scalar_add_v2_fn)(gf256 *, const gf256 *, const gf256, const gf256 *, const uint32_t);
typedef gf256 (*vec_mul_acc_fn)(const gf256 *, const gf256 *, const uint32_t);
typedef void  (*mat_add_fn)(gf256 *, const gf256 *, const gf256 *, const uint32_t, const uint32_t);
typedef void  (*mat_add_scalar_fn)(gf256 *, const gf256, const gf256 *, const uint32_t, const uint32_t);
typedef void  (*mat_mul_fn)(gf256 *, const gf256 *, const gf256 *, const uint32_t, const uint32_t, const uint32_t);

/// \return 0 if equal
static int cmp_vectors(const gf256 *a, const gf256 *b, const uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("%s error: n=%u, pos=%u: %u != %u\n", name, n, i, a[i], b[i]);
            return 1;
        }
    }
    return 0;
}

/// checks add, sub, mul, neg, inv exhaustively
uint32_t test_arith() {
    if (gf256_inv(0) != 0) {
        printf("test_arith error: inv(0) != 0\n");
        return 1;
    }

    for (uint32_t a = 0; a < 256; a++) {
        if (gf256_add(a, gf256_neg(a)) != 0) {
            printf("test_arith error: neg %u\n", a);
            return 1;
        }

        if (a) {
            const gf256 ai = gf256_inv(a);
            if (gf256_mul(a, ai) != 1) {
                printf("test_arith error: inv %u\n", a);
                return 1;
            }
            if (ai != gf256_inverse_tab[a]) {
                printf("test_arith error: inv table %u\n", a);
                return 1;
            }
        }

        if (gf256_squ(a) != gf256_mul(a, a)) {
            printf("test_arith error: squ %u\n", a);
            return 1;
        }

        for (uint32_t b = 0; b < 256; b++) {
            if (gf256_mul(a, b) != gf256_mul_v2(a, b)) {
                printf("test_arith error: mul %u %u\n", a, b);
                return 1;
            }
            if (gf256_mul(a, b) != gf256_mul(b, a)) {
                printf("test_arith error: mul commutativity %u %u\n", a, b);
                return 1;
            }
            if (gf256_sub(gf256_add(a, b), b) != a) {
                printf("test_arith error: sub %u %u\n", a, b);
                return 1;
            }
        }
    }

    // a*(b+c) = a*b + a*c
    for (uint32_t i = 0; i < 10000; i++) {
        const gf256 a = rand(), b = rand(), c = rand();
        if (gf256_mul(a, gf256_add(b, c)) != gf256_add(gf256_mul(a, b), gf256_mul(a, c))) {
            printf("test_arith error: distributivity\n");
            return 1;
        }
    }

    // 8 elements in parallel
    for (uint32_t i = 0; i < 1000; i++) {
        const uint64_t a = ((uint64_t)rand() << 32) ^ rand();
        const uint64_t b = ((uint64_t)rand() << 32) ^ rand();
        const uint64_t c = gf256v_mul_u64(a, b);
        const uint64_t d = gf256v_squ_u64(a);
        for (uint32_t k = 0; k < 8; k++) {
            const gf256 ak = a >> (8*k), bk = b >> (8*k);
            if ((gf256)(c >> (8*k)) != gf256_mul(ak, bk)) {
                printf("test_arith error: mul_u64\n");
                return 1;
            }
            if ((gf256)(d >> (8*k)) != gf256_mul(ak, ak)) {
                printf("test_arith error: squ_u64\n");
                return 1;
            }
        }
        if ((gf256v_add_u64(a, b) != (a ^ b)) || (gf256v_sub_u64(a, b) != (a ^ b))) {
            printf("test_arith error: add/sub u64\n");
            return 1;
        }
    }

    return 0;
}

/// checks the reference vector functions
uint32_t test_vector() {
    uint32_t ret = 0;
    const uint32_t n = 77;
    gf256 *a = gf256_vector_alloc(n);
    gf256 *b = gf256_vector_alloc(n);
    gf256 *c1 = gf256_vector_alloc(n);
    gf256 *c2 = gf256_vector_alloc(n);

    gf256_vector_random(a, n);
    gf256_vector_random(b, n);

    // get/set
    for (uint32_t i = 0; i < n; i++) {
        gf256_vector_set(c1, i, gf256_vector_get(a, i));
    }
    if (cmp_vectors(a, c1, n, "test_vector get/set")) { ret = 1; goto finish; }

    // copy, zero
    gf256_vector_copy(c1, b, n);
    if (cmp_vectors(b, c1, n, "test_vector copy")) { ret = 1; goto finish; }
    gf256_vector_zero(c1, n);
    for (uint32_t i = 0; i < n; i++) {
        if (c1[i]) { printf("test_vector zero error\n"); ret = 1; goto finish; }
    }

    // add
    gf256_vector_copy(c1, a, n);
    gf256_vector_add(c1, b, n);
    for (uint32_t i = 0; i < n; i++) { c2[i] = gf256_add(a[i], b[i]); }
    if (cmp_vectors(c1, c2, n, "test_vector add")) { ret = 1; goto finish; }

    // scalar_add and scalar_add_v2
    const gf256 s = rand();
    gf256_vector_copy(c1, a, n);
    gf256_vector_scalar_add(c1, s, b, n);
    gf256_vector_scalar_add_v2(c2, a, s, b, n);
    if (cmp_vectors(c1, c2, n, "test_vector scalar_add_v2")) { ret = 1; goto finish; }
    for (uint32_t i = 0; i < n; i++) { c2[i] = gf256_add(a[i], gf256_mul(s, b[i])); }
    if (cmp_vectors(c1, c2, n, "test_vector scalar_add")) { ret = 1; goto finish; }

    // mul_acc
    gf256 acc = 0;
    for (uint32_t i = 0; i < n; i++) { acc ^= gf256_mul(a[i], b[i]); }
    if (acc != gf256_vector_mul_acc(a, b, n)) {
        printf("test_vector mul_acc error\n");
        ret = 1;
    }

finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

/// checks the reference matrix functions
uint32_t test_matrix() {
    uint32_t ret = 0;
    const uint32_t r = 13, k = 7, c = 9;
    gf256 *A  = gf256_matrix_alloc(r, k);
    gf256 *At = gf256_matrix_alloc(k, r);
    gf256 *A2 = gf256_matrix_alloc(r, k);
    gf256 *I  = gf256_matrix_alloc(k, k);
    gf256 *B  = gf256_matrix_alloc(k, c);
    gf256 *C  = gf256_matrix_alloc(r, c);
    gf256 *R1 = gf256_matrix_alloc(r, c);
    gf256 *R2 = gf256_matrix_alloc(r, c);

    gf256_matrix_random(A, r, k);
    gf256_matrix_random(B, k, c);
    gf256_matrix_random(C, r, c);

    // get/set
    for (uint32_t i = 0; i < r; i++) {
        for (uint32_t j = 0; j < k; j++) {
            gf256_matrix_set(A2, r, i, j, gf256_matrix_get(A, r, i, j));
        }
    }
    if (cmp_vectors(A, A2, r*k, "test_matrix get/set")) { ret = 1; goto finish; }

    // A*I == A
    gf256_matrix_id(I, k, k);
    gf256_matrix_mul(A2, A, I, r, k, k);
    if (cmp_vectors(A, A2, r*k, "test_matrix id")) { ret = 1; goto finish; }

    // (A^T)^T == A
    gf256_matrix_transpose(At, A, r, k);
    for (uint32_t i = 0; i < r; i++) {
        for (uint32_t j = 0; j < k; j++) {
            if (gf256_matrix_get(At, k, j, i) != gf256_matrix_get(A, r, i, j)) {
                printf("test_matrix transpose error\n"); ret = 1; goto finish;
            }
        }
    }
    gf256_matrix_zero(A2, r, k);
    gf256_matrix_transpose(A2, At, k, r);
    if (cmp_vectors(A, A2, r*k, "test_matrix transpose")) { ret = 1; goto finish; }

    // add_mul == mul + add
    gf256_matrix_copy(R1, C, r, c);
    gf256_matrix_add_mul(R1, A, B, r, k, c);
    gf256_matrix_mul(R2, A, B, r, k, c);
    gf256_matrix_add(R2, R2, C, r, c);
    if (cmp_vectors(R1, R2, r*c, "test_matrix add_mul")) { ret = 1; goto finish; }

    // add_scalar
    const gf256 s = rand();
    gf256_matrix_copy(R1, C, r, c);
    gf256_matrix_add_scalar(R1, s, R2, r, c);
    for (uint32_t i = 0; i < r*c; i++) { C[i] ^= gf256_mul(s, R2[i]); }
    if (cmp_vectors(R1, C, r*c, "test_matrix add_scalar")) { ret = 1; goto finish; }

finish:
    free(A); free(At); free(A2); free(I); free(B); free(C); free(R1); free(R2);
    return ret;
}

/// checks the vectorized vector functions against the reference ones
uint32_t test_vector_simd(const char *name,
                          vec_add_fn add,
                          vec_add_v2_fn add_v2,
                          vec_scalar_add_fn scalar_add,
                          vec_scalar_add_v2_fn scalar_add_v2,
                          vec_mul_acc_fn mul_acc) {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t n = test_lengths[t];
        gf256 *a  = gf256_vector_alloc(n);
        gf256 *b  = gf256_vector_alloc(n);
        gf256 *c  = gf256_vector_alloc(n);
        gf256 *r1 = gf256_vector_alloc(n);
        gf256 *r2 = gf256_vector_alloc(n);
        gf256_vector_random(a, n);
        gf256_vector_random(b, n);
        gf256_vector_random(c, n);
        const gf256 s = rand();

        gf256_vector_copy(r1, c, n);
        gf256_vector_add(r1, a, n);
        gf256_vector_copy(r2, c, n);
        add(r2, a, n);
        ret |= cmp_vectors(r1, r2, n, name);

        gf256_vector_copy(r1, a, n);
        gf256_vector_add(r1, b, n);
        gf256_vector_copy(r2, c, n);
        add_v2(r2, a, b, n);
        ret |= cmp_vectors(r1, r2, n, name);

        gf256_vector_copy(r1, c, n);
        gf256_vector_scalar_add(r1, s, a, n);
        gf256_vector_copy(r2, c, n);
        scalar_add(r2, s, a, n);
        ret |= cmp_vectors(r1, r2, n, name);

        gf256_vector_scalar_add_v2(r1, a, s, b, n);
        gf256_vector_copy(r2, c, n);
        scalar_add_v2(r2, a, s, b, n);
        ret |= cmp_vectors(r1, r2, n, name);

        if (gf256_vector_mul_acc(a, b, n) != mul_acc(a, b, n)) {
            printf("%s error: mul_acc n=%u\n", name, n);
            ret = 1;
        }

        free(a); free(b); free(c); free(r1); free(r2);
        if (ret) { return ret; }
    }
    return ret;
}

/// checks the vectorized matrix functions against the reference ones
uint32_t test_matrix_simd(const char *name,
                          mat_add_fn add,
                          mat_add_scalar_fn add_scalar,
                          mat_mul_fn mul,
                          mat_mul_fn add_mul) {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {
        const uint32_t r = test_shapes[t][0], k = test_shapes[t][1], c = test_shapes[t][2];
        gf256 *A  = gf256_matrix_alloc(r, k);
        gf256 *X  = gf256_matrix_alloc(r, k);
        gf256 *B  = gf256_matrix_alloc(k, c);
        gf256 *C  = gf256_matrix_alloc(r, c);
        gf256 *R1 = gf256_matrix_alloc(r, c);
        gf256 *R2 = gf256_matrix_alloc(r, c);
        gf256 *S1 = gf256_matrix_alloc(r, k);
        gf256 *S2 = gf256_matrix_alloc(r, k);
        gf256_matrix_random(A, r, k);
        gf256_matrix_random(X, r, k);
        gf256_matrix_random(B, k, c);
        gf256_matrix_random(C, r, c);
        const gf256 s = rand();

        gf256_matrix_mul(R1, A, B, r, k, c);
        gf256_matrix_random(R2, r, c);
        mul(R2, A, B, r, k, c);
        ret |= cmp_vectors(R1, R2, r*c, name);

        gf256_matrix_copy(R1, C, r, c);
        gf256_matrix_add_mul(R1, A, B, r, k, c);
        gf256_matrix_copy(R2, C, r, c);
        add_mul(R2, A, B, r, k, c);
        ret |= cmp_vectors(R1, R2, r*c, name);

        gf256_matrix_add(S1, A, X, r, k);
        gf256_matrix_random(S2, r, k);
        add(S2, A, X, r, k);
        ret |= cmp_vectors(S1, S2, r*k, name);

        gf256_matrix_copy(S1, A, r, k);
        gf256_matrix_add_scalar(S1, s, X, r, k);
        gf256_matrix_copy(S2, A, r, k);
        add_scalar(S2, s, X, r, k);
        ret |= cmp_vectors(S1, S2, r*k, name);

        free(A); free(X); free(B); free(C); free(R1); free(R2); free(S1); free(S2);
        if (ret) {
            printf("%s error: shape %u %u %u\n", name, r, k, c);
            return ret;
        }
    }
    return ret;
}

#ifdef USE_AVX2
/// checks gf256v_{add,sub,mul}_u256 lane-wise against the scalar code
uint32_t test_arith_u256() {
    v256 a, b, c, d, e, f, g;
    for (uint32_t i = 0; i < 256; i++) {
        for (uint32_t j = 0; j < 256; j++) {
            for (uint32_t k = 0; k < 32; k++) {
                a.v8[k] = (uint8_t)(i + k);
                b.v8[k] = (uint8_t)(j*3 + k);
            }
            c.v256 = gf256v_add_u256(a.v256, b.v256);
            d.v256 = gf256v_sub_u256(a.v256, b.v256);
            e.v256 = gf256v_mul_u256(a.v256, b.v256);
            f.v256 = gf256v_mul_u256_v2(a.v256, b.v256);
            g.v256 = gf256v_mul_u256_v3(a.v256, b.v256);
            for (uint32_t k = 0; k < 32; k++) {
                const gf256 m = gf256_mul(a.v8[k], b.v8[k]);
                if ((c.v8[k] != gf256_add(a.v8[k], b.v8[k])) ||
                    (d.v8[k] != gf256_sub(a.v8[k], b.v8[k])) ||
                    (e.v8[k] != m) || (f.v8[k] != m) || (g.v8[k] != m)) {
                    printf("test_arith_u256 error: %u %u %u\n", i, j, k);
                    return 1;
                }
            }
        }
    }
    return 0;
}

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
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
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
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
            const gf256 t3 = gf256_matrix_get(B, nrows, i, j);
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
    const uint32_t nrows = 19;
    const uint32_t ncols = 14;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf16 *B   = gf16_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_random(A, nrows, ncols);
    gf256_matrix_add_gf16(C1, A, B, nrows, ncols);
    gf256_matrix_add_gf16_u256(C2, A, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
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

uint32_t test_matrix_add_gf2_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 50;
    const uint32_t ncols = 5;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf2   *B  = gf2_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    gf2_matrix_random(B, nrows, ncols);
    gf256_matrix_random(A, nrows, ncols);
    gf256_matrix_add_gf2(C1, A, B, nrows, ncols);
    gf256_matrix_add_gf2_u256(C2, A, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_gf2_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_scalar_gf2_u256() {
    uint32_t ret = 0;
    // NOTE: nrows = 4 has a special code path, ncols % 4 != 0 tests its tail
    const uint32_t shapes[][2] = {{4, 42}, {4, 41}, {4, 43}, {5, 14}, {16, 3}};
    for (uint32_t s = 0; s < 5; s++) {
        const uint32_t nrows = shapes[s][0];
        const uint32_t ncols = shapes[s][1];
        gf2    *B = gf2_matrix_alloc(nrows, ncols);
        gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
        gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

        const gf256 scalar = 2;
        gf2_matrix_random(B, nrows, ncols);
        gf256_matrix_add_scalar_gf2(C1, scalar, B, nrows, ncols);
        gf256_matrix_add_scalar_gf2_u256(C2, scalar, B, nrows, ncols);

        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
                const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
                if (t1 != t2) {
                    gf256_matrix_print(C1, nrows, ncols);
                    gf256_matrix_print(C2, nrows, ncols);
                    printf("test_matrix_add_scalar_gf2_u256 error: %d %d\n", i, j);
                    ret = 1;
                }
            }
        }
        free(B); free(C1); free(C2);
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_matrix_add_scalar_gf16_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 5;
    const uint32_t ncols = 14;
    gf16 *B   = gf16_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf16_matrix_random(B, nrows, ncols);
    gf256_matrix_add_scalar_gf16(C1, scalar, B, nrows, ncols);
    gf256_matrix_add_scalar_gf16_u256(C2, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_scalar_gf16_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_scalar_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf256 *B  = gf256_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf256_matrix_random(B, nrows, ncols);
    gf256_matrix_add_scalar(C1, scalar, B, nrows, ncols);
    gf256_matrix_add_scalar_u256(C2, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_scalar_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_add_scalar_u256_v2() {
    uint32_t ret = 0;
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf256 *A  = gf256_matrix_alloc(nrows, ncols);
    gf256 *B  = gf256_matrix_alloc(nrows, ncols);
    gf256 *C1 = gf256_matrix_alloc(nrows, ncols);
    gf256 *C2 = gf256_matrix_alloc(nrows, ncols);

    const gf256 scalar = 2;
    gf256_matrix_random(A, nrows, ncols);
    gf256_matrix_random(B, nrows, ncols);
    gf256_matrix_add_scalar_v2(C1, A, scalar, B, nrows, ncols);
    gf256_matrix_add_scalar_u256_v2(C2, A, scalar, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf256 t1 = gf256_matrix_get(C1, nrows, i, j);
            const gf256 t2 = gf256_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                gf256_matrix_print(C1, nrows, ncols);
                gf256_matrix_print(C2, nrows, ncols);
                printf("test_matrix_add_scalar_u256_v2 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }
    finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

uint32_t test_matrix_mul_gf16_1_u256() {
    uint32_t ret = 0;
    const uint32_t shapes[][3] = {{166, 195, 1}, {37, 11, 5}, {100, 7, 3}};
    for (uint32_t s = 0; s < 3; s++) {
        const uint32_t nrows1 = shapes[s][0];
        const uint32_t ncols1 = shapes[s][1];
        const uint32_t ncols2 = shapes[s][2];

        gf16 *A   = gf16_matrix_alloc (nrows1, ncols1);
        gf256 *B  = gf256_matrix_alloc(ncols1, ncols2);
        gf256 *C1 = gf256_matrix_alloc(nrows1, ncols2);
        gf256 *C2 = gf256_matrix_alloc(nrows1, ncols2);

        gf16_matrix_random(A, nrows1, ncols1);
        gf256_matrix_random(B, ncols1, ncols2);
        gf256_matrix_random(C2, nrows1, ncols2);
        gf256_matrix_mul_gf16_1(C1, A, B, nrows1, ncols1, ncols2);
        gf256_matrix_mul_gf16_1_u256(C2, A, B, nrows1, ncols1, ncols2);

        for (uint32_t i = 0; i < nrows1; i++) {
            for (uint32_t j = 0; j < ncols2; j++) {
                const gf256 t1 = gf256_matrix_get(C1, nrows1, i, j);
                const gf256 t2 = gf256_matrix_get(C2, nrows1, i, j);
                if (t1 != t2) {
                    printf("test_matrix_mul_gf16_1_u256 error: %d %d\n", i, j);
                    ret = 1;
                    goto finish;
                }
            }
        }

    finish:
        free(A); free(B); free(C1); free(C2);
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_matrix_mul_gf16_2_u256() {
    uint32_t ret = 0;
    const uint32_t nrows1 = 15;
    const uint32_t ncols1 = 15;
    const uint32_t ncols2 = 15;

    gf16 *a   = gf16_matrix_alloc (ncols1, ncols2);
    gf256 *b  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf16_matrix_random(a, ncols1, ncols2);
    gf256_matrix_random(b, nrows1, ncols1);
    gf256_matrix_mul_gf16_2(c1, b, a, nrows1, ncols1, ncols2);
    gf256_matrix_mul_gf16_2_u256(c2, b, a, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_mul_gf16_2_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_mul_gf2_1_u256() {
    uint32_t ret = 0;
    const uint32_t nrows1 = 476;
    const uint32_t ncols1 = 2024;
    const uint32_t ncols2 = 1;

    gf2 *a    = gf2_matrix_alloc(nrows1, ncols1);
    gf256 *b  = gf256_matrix_alloc(ncols1, ncols2);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf2_matrix_random(a, nrows1, ncols1);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_mul_gf2_1(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_mul_gf2_1_u256(c2, a, b, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                printf("test_matrix_mul_gf2_1_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_mul_gf2_2_u256() {
    uint32_t ret = 0;
    const uint32_t shapes[][3] = {{11, 11, 1}, {40, 9, 1}, {13, 7, 6}};
    for (uint32_t s = 0; s < 3; s++) {
        const uint32_t nrows1 = shapes[s][0];
        const uint32_t ncols1 = shapes[s][1];
        const uint32_t ncols2 = shapes[s][2];

        gf2 *a    = gf2_matrix_alloc (ncols1, ncols2);
        gf256 *b  = gf256_matrix_alloc(nrows1, ncols1);
        gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
        gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

        gf2_matrix_random(a, ncols1, ncols2);
        gf256_matrix_random(b, nrows1, ncols1);
        gf256_matrix_random(c2, nrows1, ncols2);
        gf256_matrix_mul_gf2_2(c1, b, a, nrows1, ncols1, ncols2);
        gf256_matrix_mul_gf2_2_u256(c2, b, a, nrows1, ncols1, ncols2);

        for (uint32_t i = 0; i < nrows1; i++) {
            for (uint32_t j = 0; j < ncols2; j++) {
                const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
                const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
                if (t1 != t2) {
                    printf("test_matrix_mul_gf2_2_u256 error: %d %d\n", i, j);
                    ret = 1;
                    goto finish;
                }
            }
        }

    finish:
        free(a); free(b); free(c1); free(c2);
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_matrix_mul_8x8xC_u256() {
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
    gf256_matrix_mul(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_mul_8x8xC_u256(c2, a, b, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_mul_8x8xC_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_mul_16x16xC_u256() {
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
    gf256_matrix_mul(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_mul_16x16xC_u256(c2, a, b, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_mul_16x16xC_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_mul_le32xBxle16_u256() {
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
    gf256_matrix_mul(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_mul_le32xBxle16_u256(c2, a, b, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_mul_le32xBxle16_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_mul_u256() {
    uint32_t ret = 0;
    const uint32_t nrows1 = 50;
    const uint32_t ncols1 = 5;
    const uint32_t ncols2 = 45;

    gf256 *b  = gf256_matrix_alloc(ncols1, ncols2);
    gf256 *a  = gf256_matrix_alloc(nrows1, ncols1);
    gf256 *c1 = gf256_matrix_alloc(nrows1, ncols2);
    gf256 *c2 = gf256_matrix_alloc(nrows1, ncols2);

    gf256_matrix_random(a, nrows1, ncols1);
    gf256_matrix_random(b, ncols1, ncols2);
    gf256_matrix_mul(c1, a, b, nrows1, ncols1, ncols2);
    gf256_matrix_mul_u256(c2, a, b, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf256 t1 = gf256_matrix_get(c1, nrows1, i, j);
            const gf256 t2 = gf256_matrix_get(c2, nrows1, i, j);
            if (t1 != t2) {
                gf256_matrix_print(c1, nrows1, ncols2);
                gf256_matrix_print(c2, nrows1, ncols2);
                printf("test_matrix_mul_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

uint32_t test_matrix_map_gf2_u256() {
    uint32_t ret = 0;
    const uint32_t nrows = 13;
    const uint32_t ncols = 9;
    gf2 *B    = gf2_matrix_alloc(nrows, ncols);
    gf256 *C  = gf256_matrix_alloc(nrows, ncols);

    gf2_matrix_random(B, nrows, ncols);
    gf256_matrix_map_gf2_u256(C, B, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            if (gf256_matrix_get(C, nrows, i, j) != gf2_matrix_get(B, nrows, i, j)) {
                printf("test_matrix_map_gf2_u256 error: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(B); free(C);
    return ret;
}

uint32_t test_vector_set_to_gf16_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
        gf16 *A = gf16_vector_alloc(N);
        gf256 *B1 = gf256_vector_alloc(N);
        gf256 *B2 = gf256_vector_alloc(N);
        gf256 *B3 = gf256_vector_alloc(N);

        gf16_vector_random(A, N);
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
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_vector_set_to_gf2_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
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
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_vector_add_gf2_2_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
        gf2 *A = gf2_vector_alloc(N);
        gf256 *B = gf256_vector_alloc(N);
        gf256 *C1 = gf256_vector_alloc(N);
        gf256 *C2 = gf256_vector_alloc(N);

        gf2_vector_random(A, N);
        gf256_vector_random(B, N);
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
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_vector_add_gf16_2_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
        gf16  *A  = gf16_vector_alloc(N);
        gf256 *B  = gf256_vector_alloc(N);
        gf256 *C1 = gf256_vector_alloc(N);
        gf256 *C2 = gf256_vector_alloc(N);

        gf16_vector_random(A, N);
        gf256_vector_random(B, N);
        gf256_vector_add_2_gf16(C1, B, A, N);
        gf256_vector_add_2_gf16_u256(C2, B, A, N);

        for (uint32_t i = 0; i < N; i++) {
            if (C1[i] != C2[i]) {
                gf256_vector_print(C1, N);
                gf256_vector_print(C2, N);
                printf("test_vector_add_2_gf16_u256 error %d\n", i);
                ret = 1;
                break;
            }
        }

        free(A); free(B); free(C1); free(C2);
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_vector_scalar_gf2_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
        gf2 *A = gf2_vector_alloc(N);
        gf256 *C1 = gf256_vector_alloc(N);
        gf256 *C2 = gf256_vector_alloc(N);

        const gf256 B = rand();
        gf2_vector_random(A, N);
        gf256_vector_scalar_gf2(C1, B, A, N);
        gf256_vector_scalar_gf2_u256(C2, B, A, N);

        for (uint32_t i = 0; i < N; i++) {
            if (C1[i] != C2[i]) {
                gf256_vector_print(C1, N);
                gf256_vector_print(C2, N);
                printf("test_vector_scalar_gf2_u256 error %d\n", i);
                ret = 1;
                break;
            }
        }

        free(A); free(C1); free(C2);
        if (ret) { return ret; }
    }
    return ret;
}

uint32_t test_vector_scalar_gf16_u256() {
    uint32_t ret = 0;
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t N = test_lengths[t];
        gf16  *A  = gf16_vector_alloc(N);
        gf256 *C1 = gf256_vector_alloc(N);
        gf256 *C2 = gf256_vector_alloc(N);

        const gf256 B = rand();
        gf16_vector_random(A, N);
        gf256_vector_scalar_gf16(C1, B, A, N);
        gf256_vector_scalar_gf16_u256(C2, B, A, N);

        for (uint32_t i = 0; i < N; i++) {
            if (C1[i] != C2[i]) {
                gf256_vector_print(C1, N);
                gf256_vector_print(C2, N);
                printf("test_vector_scalar_gf16_u256 error %d\n", i);
                ret = 1;
                break;
            }
        }

        free(A); free(C1); free(C2);
        if (ret) { return ret; }
    }
    return ret;
}

/// NOTE: `gf256v_mul_u32_u256` requires each 32bit limb of `b` to be < 256
int check_gf256v_mul_u32_u256() {
    v256 v1 = {0}, v2 = {0}, v3;
    for (uint32_t i = 0; i < 256; ++i) {
        for (uint32_t j = 0; j < 256; ++j) {
            for (uint32_t k = 0; k < 32; ++k) { v1.v8[k] = (uint8_t)(i + k); }
            for (uint32_t l = 0; l < 8; ++l)  { v2.v32[l] = (j + 7*l) & 0xFF; }

            v3.v256 = gf256v_mul_u32_u256(v1.v256, v2.v256);
            for (uint32_t k = 0; k < 32; ++k) {
                if (v3.v8[k] != gf256_mul(v1.v8[k], v2.v32[k/4])) {
                    printf("check_gf256v_mul_u32_u256 error: %u %u %u\n", i, j, k);
                    return 1;
                }
            }
        }
    }
    return 0;
}

int check_gf256v_sqr_u256() {
    v256 v1, v2;
    for (uint32_t i = 0; i < 256; ++i) {
        for (uint32_t k = 0; k < 32; ++k) { v1.v8[k] = (uint8_t)(i + 13*k); }
        v2.v256 = gf256v_sqr_u256(v1.v256);
        for (uint32_t k = 0; k < 32; ++k) {
            if (v2.v8[k] != gf256_mul(v1.v8[k], v1.v8[k])) {
                printf("check_gf256v_sqr_u256 error: %u %u\n", i, k);
                return 1;
            }
        }
    }
    return 0;
}

int check_gf256v_mul_scalar_u256() {
    v256 a, c1, c2, c3;
    for (uint32_t k = 0; k < 32; k++) { a.v8[k] = (uint8_t)(k*7 + 1); }

    for (uint32_t b = 0; b < 256; b++) {
        const __m256i b1 = _mm256_set1_epi8(b);
        c1.v256 = gf256v_mul_u256(a.v256, b1);
        c2.v256 = gf256v_mul_scalar_u256(a.v256, b);
        c3.v256 = gf256v_mul_scalar_u256_non_ct(a.v256, b);
        for (uint32_t k = 0; k < 32; k++) {
            if ((c1.v8[k] != c2.v8[k]) || (c1.v8[k] != c3.v8[k])) {
                printf("check_gf256v_mul_scalar_u256 error: %u %u\n", b, k);
                return 1;
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_NEON
/// checks gf256v_{add,sub,mul}_u128 lane-wise against the scalar code
uint32_t test_arith_u128() {
    uint8_t a[16], b[16], c[16], d[16], e[16];
    for (uint32_t i = 0; i < 256; i++) {
        for (uint32_t j = 0; j < 256; j++) {
            for (uint32_t k = 0; k < 16; k++) {
                a[k] = (uint8_t)(i + k);
                b[k] = (uint8_t)(j*3 + k);
            }
            const uint8x16_t ta = vld1q_u8(a), tb = vld1q_u8(b);
            vst1q_u8(c, gf256v_add_u128(ta, tb));
            vst1q_u8(d, gf256v_sub_u128(ta, tb));
            vst1q_u8(e, gf256v_mul_u128(ta, tb));
            for (uint32_t k = 0; k < 16; k++) {
                if ((c[k] != gf256_add(a[k], b[k])) ||
                    (d[k] != gf256_sub(a[k], b[k])) ||
                    (e[k] != gf256_mul(a[k], b[k]))) {
                    printf("test_arith_u128 error: %u %u %u\n", i, j, k);
                    return 1;
                }
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_AVX512
/// checks gf256v_{add,sub,mul}_u512 lane-wise against the scalar code
uint32_t test_arith_u512() {
    uint8_t a[64], b[64], c[64], d[64], e[64];
    for (uint32_t i = 0; i < 256; i++) {
        for (uint32_t j = 0; j < 256; j++) {
            for (uint32_t k = 0; k < 64; k++) {
                a[k] = (uint8_t)(i + k);
                b[k] = (uint8_t)(j*3 + k);
            }
            const __m512i ta = _mm512_loadu_si512((const void *)a);
            const __m512i tb = _mm512_loadu_si512((const void *)b);
            _mm512_storeu_si512((void *)c, gf256v_add_u512(ta, tb));
            _mm512_storeu_si512((void *)d, gf256v_sub_u512(ta, tb));
            _mm512_storeu_si512((void *)e, gf256v_mul_u512(ta, tb));
            for (uint32_t k = 0; k < 64; k++) {
                if ((c[k] != gf256_add(a[k], b[k])) ||
                    (d[k] != gf256_sub(a[k], b[k])) ||
                    (e[k] != gf256_mul(a[k], b[k]))) {
                    printf("test_arith_u512 error: %u %u %u\n", i, j, k);
                    return 1;
                }
            }
        }
    }
    return 0;
}
#endif

int main() {
    srand(1);

    if (test_arith()) { return 1; }
    if (test_vector()) { return 1; }
    if (test_matrix()) { return 1; }

#ifdef USE_AVX2
    if (test_arith_u256()) { return 1; }
    if (test_vector_simd("vector_u256", gf256_vector_add_u256, gf256_vector_add_u256_v2,
                         gf256_vector_scalar_add_u256, gf256_vector_scalar_add_u256_v2,
                         gf256_vector_mul_acc_u256)) { return 1; }
    if (test_matrix_simd("matrix_u256", gf256_matrix_add_u256, gf256_matrix_add_scalar_u256,
                         gf256_matrix_mul_u256, gf256_matrix_add_mul_u256)) { return 1; }

    if (test_matrix_map_gf16_to_gf256_u256()) { return 1; }
    if (test_matrix_map_gf2_u256()) { return 1; }
    if (test_matrix_add_u256()) { return 1; }
    if (test_matrix_add_gf2_u256()) { return 1; }
    if (test_matrix_add_gf16_u256()) { return 1; }
    if (test_matrix_add_scalar_gf2_u256()) { return 1; }
    if (test_matrix_add_scalar_gf16_u256()) { return 1; }
    if (test_matrix_add_scalar_u256()) { return 1; }
    if (test_matrix_add_scalar_u256_v2()) { return 1; }
    if (test_matrix_mul_gf2_1_u256()) { return 1; }
    if (test_matrix_mul_gf2_2_u256()) { return 1; }
    if (test_matrix_mul_gf16_1_u256()) { return 1; }
    if (test_matrix_mul_gf16_2_u256()) { return 1; }
    if (test_matrix_mul_8x8xC_u256()) { return 1; }
    if (test_matrix_mul_16x16xC_u256()) { return 1; }
    if (test_matrix_mul_le32xBxle16_u256()) { return 1; }
    if (test_matrix_mul_u256()) { return 1; }

    if (test_vector_set_to_gf16_u256()) { return 1; }
    if (test_vector_set_to_gf2_u256()) { return 1; }
    if (test_vector_add_gf2_2_u256()) { return 1; }
    if (test_vector_add_gf16_2_u256()) { return 1; }
    if (test_vector_scalar_gf2_u256()) { return 1; }
    if (test_vector_scalar_gf16_u256()) { return 1; }

    if (check_gf256v_mul_scalar_u256()) { return 1; }
    if (check_gf256v_mul_u32_u256()) { return 1; }
    if (check_gf256v_sqr_u256()) { return 1; }
#endif

#ifdef USE_NEON
    if (test_arith_u128()) { return 1; }
    if (test_vector_simd("vector_u128", gf256_vector_add_u128, gf256_vector_add_u128_v2,
                         gf256_vector_scalar_add_u128, gf256_vector_scalar_add_u128_v2,
                         gf256_vector_mul_acc_u128)) { return 1; }
    if (test_matrix_simd("matrix_u128", gf256_matrix_add_u128, gf256_matrix_add_scalar_u128,
                         gf256_matrix_mul_u128, gf256_matrix_add_mul_u128)) { return 1; }
#endif

#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
    if (test_vector_simd("vector_u512", gf256_vector_add_u512, gf256_vector_add_u512_v2,
                         gf256_vector_scalar_add_u512, gf256_vector_scalar_add_u512_v2,
                         gf256_vector_mul_acc_u512)) { return 1; }
    if (test_matrix_simd("matrix_u512", gf256_matrix_add_u512, gf256_matrix_add_scalar_u512,
                         gf256_matrix_mul_u512, gf256_matrix_add_mul_u512)) { return 1; }
#endif

    printf("All good\n");
    return 0;
}
