#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"


const uint64_t N = 1u << 10u;

static const uint32_t test_lengths[] = {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// {n_rows1, n_cols1, n_cols2}
static const uint32_t test_shapes[][3] = {{1, 1, 1}, {3, 5, 2}, {17, 9, 4}, {33, 40, 7}, {8, 8, 8}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

static int vec_eq(const gf2to256 *a,
                  const gf2to256 *b,
                  const uint32_t n) {
    return memcmp(a, b, n * sizeof(gf2to256)) == 0;
}

int test_arith() {
    gf2to256 a, b, c1, c2, c3, zero, one;
    gf2to256_set_zero(zero);
    gf2to256_set_zero(one); one[0] = 1;
    for (uint64_t i = 0; i < N; i++) {
        gf2to256_set_random(a);
        gf2to256_set_random(b);

        gf2to256_mul(c1, a, b);
        gf2to256_mul_v2(c2, a, b);
        if (!gf2to256_is_equal(c1, c2)) {
            printf("test_arith: mul != mul_v2\n");
            return 1;
        }

        gf2to256_mul(c2, b, a);
        if (!gf2to256_is_equal(c1, c2)) {
            printf("test_arith: mul not commutative\n");
            return 1;
        }

        // aliasing
        gf2to256_copy(c2, a);
        gf2to256_mul(c2, c2, b);
        gf2to256_copy(c3, b);
        gf2to256_mul(c3, a, c3);
        if (!gf2to256_is_equal(c1, c2) || !gf2to256_is_equal(c1, c3)) {
            printf("test_arith: mul aliasing\n");
            return 1;
        }

        gf2to256_add(c1, a, b);
        gf2to256_sub(c2, c1, b);
        if (!gf2to256_is_equal(c2, a)) {
            printf("test_arith: sub(add(a, b), b) != a\n");
            return 1;
        }

        gf2to256_neg(c1, a);
        gf2to256_add(c2, a, c1);
        if (!gf2to256_is_equal(c2, zero)) {
            printf("test_arith: a + neg(a) != 0\n");
            return 1;
        }
    }

    for (uint64_t i = 0; i < 16; i++) {
        gf2to256_set_random(a);
        gf2to256_inv(b, a);
        gf2to256_mul(c1, a, b);
        if (!gf2to256_is_equal(c1, one)) {
            printf("test_arith: inv(a)*a != 1\n");
            return 1;
        }
    }

    gf2to256_inv(b, zero);
    if (!gf2to256_is_equal(b, zero)) {
        printf("test_arith: inv(0) != 0\n");
        return 1;
    }

    return 0;
}

int test_vector() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        gf2to256 *a = gf2to256_vector_alloc(n);
        gf2to256 *b = gf2to256_vector_alloc(n);
        gf2to256 *r1 = gf2to256_vector_alloc(n);
        gf2to256 *r2 = gf2to256_vector_alloc(n);
        gf2to256 s, t, acc1, acc2;
        gf2 *g = gf2_vector_alloc(n);

        gf2to256_vector_random(a, n);
        gf2to256_vector_random(b, n);
        gf2_vector_random(g, n);
        gf2to256_set_random(s);

        for (uint32_t i = 0; i < n; i++) {
            gf2to256 x, y;
            gf2to256_set_random(x);
            gf2to256_vector_set(r1, i, x);
            gf2to256_vector_get(y, r1, i);
            if (!gf2to256_is_equal(x, y)) {
                printf("test_vector: get/set\n");
                return 1;
            }
        }

        // add
        gf2to256_vector_copy(r1, a, n);
        gf2to256_vector_add(r1, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf2to256_add(t, a[i], b[i]);
            if (!gf2to256_is_equal(t, r1[i])) {
                printf("test_vector: add %u\n", n);
                return 1;
            }
        }

        // scalar_add
        gf2to256_vector_copy(r1, a, n);
        gf2to256_vector_scalar_add(r1, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf2to256_mul(t, s, b[i]);
            gf2to256_add(t, t, a[i]);
            if (!gf2to256_is_equal(t, r1[i])) {
                printf("test_vector: scalar_add %u\n", n);
                return 1;
            }
        }

        // scalar_add_v2
        gf2to256_vector_zero(r2, n);
        gf2to256_vector_scalar_add_v2(r2, a, s, b, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_v2 %u\n", n);
            return 1;
        }

        // mul_acc
        gf2to256_set_zero(acc1);
        for (uint32_t i = 0; i < n; i++) {
            gf2to256_mul(t, a[i], b[i]);
            gf2to256_add(acc1, acc1, t);
        }
        gf2to256_vector_mul_acc(acc2, a, b, n);
        if (!gf2to256_is_equal(acc1, acc2)) {
            printf("test_vector: mul_acc %u\n", n);
            return 1;
        }

        // gf2 sub field
        gf2to256_vector_copy(r1, a, n);
        gf2to256_vector_add_gf2(r1, g, n);
        for (uint32_t i = 0; i < n; i++) {
            gf2to256_copy(t, a[i]);
            t[0] ^= gf2_vector_get(g, i);
            if (!gf2to256_is_equal(t, r1[i])) {
                printf("test_vector: add_gf2 %u\n", n);
                return 1;
            }
        }

#ifdef USE_AVX2
        gf2to256_vector_copy(r2, a, n);
        gf2to256_vector_add_gf2_u256(r2, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: add_gf2_u256 %u\n", n);
            return 1;
        }

        gf2to256_vector_zero(r2, n);
        gf2to256_vector_add_gf2_u256_v2(r2, a, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: add_gf2_u256_v2 %u\n", n);
            return 1;
        }

        gf2to256_vector_copy(r1, a, n);
        gf2to256_vector_scalar_add_gf2_v3(r1, s, g, n);
        gf2to256_vector_copy(r2, a, n);
        gf2to256_vector_scalar_add_gf2_u256_v2(r2, s, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_gf2_u256_v2 %u\n", n);
            return 1;
        }

        gf2to256_vector_copy(r1, a, n);
        gf2to256_vector_scalar_add_gf2(r1, 1, b, n);
        gf2to256_vector_copy(r2, a, n);
        gf2to256_vector_scalar_add_gf2_u256(r2, 1, b, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_gf2_u256 %u\n", n);
            return 1;
        }
#endif

        free(a); free(b); free(r1); free(r2); free(g);
    }

    return 0;
}

/// tests the SIMD vector functions with suffix X against the reference
/// implementation. One extra element is allocated to detect out-of-bound writes.
#define DEFINE_TEST_VECTOR_SIMD(X)                                              \
int test_vector_##X() {                                                         \
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {                            \
        const uint32_t n = test_lengths[l];                                     \
        gf2to256 *a  = gf2to256_vector_alloc(n + 1);                            \
        gf2to256 *b  = gf2to256_vector_alloc(n + 1);                            \
        gf2to256 *r1 = gf2to256_vector_alloc(n + 1);                            \
        gf2to256 *r2 = gf2to256_vector_alloc(n + 1);                            \
        gf2to256 s, acc1, acc2;                                                 \
        gf2to256_vector_random(a, n + 1);                                       \
        gf2to256_vector_random(b, n + 1);                                       \
        gf2to256_vector_random(r1, n + 1);                                      \
        gf2to256_set_random(s);                                                 \
                                                                                \
        gf2to256_vector_copy(r2, r1, n + 1);                                    \
        gf2to256_vector_add(r1, a, n);                                          \
        gf2to256_vector_add_##X(r2, a, n);                                      \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": add %u\n", n);                          \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_vector_copy(r1, a, n);                                         \
        gf2to256_vector_add(r1, b, n);                                          \
        gf2to256_vector_add_##X##_v2(r2, a, b, n);                              \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": add_v2 %u\n", n);                       \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_vector_scalar_add(r1, s, b, n);                                \
        gf2to256_vector_scalar_add_##X(r2, s, b, n);                            \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": scalar_add %u\n", n);                   \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_vector_scalar_add_v2(r1, a, s, b, n);                          \
        gf2to256_vector_scalar_add_##X##_v2(r2, a, s, b, n);                    \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": scalar_add_v2 %u\n", n);                \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_vector_mul_acc(acc1, a, b, n);                                 \
        gf2to256_vector_mul_acc_##X(acc2, a, b, n);                             \
        if (!gf2to256_is_equal(acc1, acc2)) {                                   \
            printf("test_vector_" #X ": mul_acc %u\n", n);                      \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        free(a); free(b); free(r1); free(r2);                                   \
    }                                                                           \
    return 0;                                                                   \
}

int test_matrix() {
    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t nr = test_shapes[l][0], nk = test_shapes[l][1], nc = test_shapes[l][2];
        gf2to256 *A = gf2to256_matrix_alloc(nr, nk);
        gf2to256 *B = gf2to256_matrix_alloc(nk, nc);
        gf2to256 *C = gf2to256_matrix_alloc(nr, nc);
        gf2to256 *I = gf2to256_matrix_alloc(nr, nr);
        gf2to256 *R1 = gf2to256_matrix_alloc(nr, nk);
        gf2to256 *R2 = gf2to256_matrix_alloc(nr, nk);
        gf2to256 *T = gf2to256_matrix_alloc(nk, nr);
        gf2to256 *M1 = gf2to256_matrix_alloc(nr, nc);
        gf2to256 *M2 = gf2to256_matrix_alloc(nr, nc);
        gf2to256 s, x, y;
        gf2 *G = gf2_matrix_alloc(nr, nk);

        gf2to256_matrix_random(A, nr, nk);
        gf2to256_matrix_random(B, nk, nc);
        gf2to256_matrix_random(C, nr, nc);
        gf2_matrix_random(G, nr, nk);
        gf2to256_set_random(s);

        // get/set
        gf2to256_matrix_copy(R1, A, nr, nk);
        gf2to256_set_random(x);
        gf2to256_matrix_set(R1, nr, nr - 1, nk - 1, x);
        gf2to256_matrix_get(y, R1, nr, nr - 1, nk - 1);
        if (!gf2to256_is_equal(x, y) || !gf2to256_is_equal(R1[nr*nk - 1], x)) {
            printf("test_matrix: get/set\n");
            return 1;
        }

        // id * A = A
        gf2to256_matrix_id(I, nr, nr);
        gf2to256_matrix_mul(R1, I, A, nr, nr, nk);
        if (!vec_eq(R1, A, nr*nk)) {
            printf("test_matrix: id\n");
            return 1;
        }

        // transpose
        gf2to256_matrix_transpose(T, A, nr, nk);
        gf2to256_matrix_get(x, A, nr, nr - 1, 0);
        gf2to256_matrix_get(y, T, nk, 0, nr - 1);
        if (!gf2to256_is_equal(x, y)) {
            printf("test_matrix: transpose entry\n");
            return 1;
        }
        gf2to256_matrix_transpose(R1, T, nk, nr);
        if (!vec_eq(R1, A, nr*nk)) {
            printf("test_matrix: transpose\n");
            return 1;
        }

        // add_mul == mul + add
        gf2to256_matrix_mul(M1, A, B, nr, nk, nc);
        gf2to256_matrix_add(M1, M1, C, nr, nc);
        gf2to256_matrix_copy(M2, C, nr, nc);
        gf2to256_matrix_add_mul(M2, A, B, nr, nk, nc);
        if (!vec_eq(M1, M2, nr*nc)) {
            printf("test_matrix: add_mul\n");
            return 1;
        }

        // add_scalar
        gf2to256_matrix_copy(R1, A, nr, nk);
        gf2to256_matrix_add_scalar(R1, s, A, nr, nk);
        gf2to256_matrix_add_scalar_v2(R2, A, s, A, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_scalar\n");
            return 1;
        }
        for (uint32_t i = 0; i < nr*nk; i++) {
            gf2to256_mul(x, s, A[i]);
            gf2to256_add(x, x, A[i]);
            if (!gf2to256_is_equal(x, R1[i])) {
                printf("test_matrix: add_scalar entry\n");
                return 1;
            }
        }

#ifdef USE_AVX2
        gf2to256_matrix_add_gf2(R1, G, A, nr, nk);
        gf2to256_matrix_add_gf2_u256(R2, G, A, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_gf2_u256\n");
            return 1;
        }

        gf2to256_matrix_copy(R1, A, nr, nk);
        gf2to256_matrix_add_scalar_gf2(R1, s, G, nr, nk);
        gf2to256_matrix_copy(R2, A, nr, nk);
        gf2to256_matrix_add_scalar_gf2_u256(R2, s, G, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_scalar_gf2_u256\n");
            return 1;
        }
#endif

        free(A); free(B); free(C); free(I); free(R1); free(R2);
        free(T); free(M1); free(M2); free(G);
    }
    return 0;
}

/// tests the SIMD matrix functions with suffix X against the reference
#define DEFINE_TEST_MATRIX_SIMD(X)                                              \
int test_matrix_##X() {                                                         \
    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {                             \
        const uint32_t nr = test_shapes[l][0], nk = test_shapes[l][1],          \
                       nc = test_shapes[l][2];                                  \
        gf2to256 *A = gf2to256_matrix_alloc(nr, nk);                            \
        gf2to256 *A2 = gf2to256_matrix_alloc(nr, nk);                           \
        gf2to256 *B = gf2to256_matrix_alloc(nk, nc);                            \
        gf2to256 *R1 = gf2to256_matrix_alloc(nr, nk);                           \
        gf2to256 *R2 = gf2to256_matrix_alloc(nr, nk);                           \
        gf2to256 *M1 = gf2to256_matrix_alloc(nr, nc);                           \
        gf2to256 *M2 = gf2to256_matrix_alloc(nr, nc);                           \
        gf2to256 s;                                                             \
        gf2to256_matrix_random(A, nr, nk);                                      \
        gf2to256_matrix_random(A2, nr, nk);                                     \
        gf2to256_matrix_random(B, nk, nc);                                      \
        gf2to256_matrix_random(M1, nr, nc);                                     \
        gf2to256_matrix_copy(M2, M1, nr, nc);                                   \
        gf2to256_set_random(s);                                                 \
                                                                                \
        gf2to256_matrix_add(R1, A, A2, nr, nk);                                 \
        gf2to256_matrix_add_##X(R2, A, A2, nr, nk);                             \
        if (!vec_eq(R1, R2, nr*nk)) {                                           \
            printf("test_matrix_" #X ": add\n");                                \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_matrix_add_scalar(R1, s, A2, nr, nk);                          \
        gf2to256_matrix_add_scalar_##X(R2, s, A2, nr, nk);                      \
        if (!vec_eq(R1, R2, nr*nk)) {                                           \
            printf("test_matrix_" #X ": add_scalar\n");                         \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_matrix_add_mul(M1, A, B, nr, nk, nc);                          \
        gf2to256_matrix_add_mul_##X(M2, A, B, nr, nk, nc);                      \
        if (!vec_eq(M1, M2, nr*nc)) {                                           \
            printf("test_matrix_" #X ": add_mul\n");                            \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf2to256_matrix_mul(M1, A, B, nr, nk, nc);                              \
        gf2to256_matrix_mul_##X(M2, A, B, nr, nk, nc);                          \
        if (!vec_eq(M1, M2, nr*nc)) {                                           \
            printf("test_matrix_" #X ": mul\n");                                \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        free(A); free(A2); free(B); free(R1); free(R2); free(M1); free(M2);     \
    }                                                                           \
    return 0;                                                                   \
}

#ifdef USE_AVX2
/// out = in1 + in2 and out += in on the AVX2 vector API
uint32_t test_vector_add() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        gf2to256 *a = gf2to256_vector_alloc(n);
        gf2to256 *b = gf2to256_vector_alloc(n);
        gf2to256 *c = gf2to256_vector_alloc(n);
        gf2to256 t;
        gf2to256_vector_random(a, n);
        gf2to256_vector_random(b, n);

        gf2to256_vector_add_u256_v2(c, a, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf2to256_add(t, a[i], b[i]);
            if (!gf2to256_is_equal(t, c[i])) {
                printf("test_vector_add: %u\n", n);
                return 1;
            }
        }

        // c = a + b + b = a
        gf2to256_vector_add_u256(c, b, n);
        if (!vec_eq(a, c, n)) {
            printf("test_vector_add (2): %u\n", n);
            return 1;
        }

        free(a); free(b); free(c);
    }
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

        if (!gf2to256_is_equal(c1, c2)) {
            printf("test_vector_mul: %llx %llx\n",
                   (unsigned long long)c1[0], (unsigned long long)c2[0]);
            return 1;
        }

        if (!gf2to256_is_equal(c1, c3)) {
            printf("test_vector_mul(3): %llx %llx\n",
                   (unsigned long long)c1[0], (unsigned long long)c3[0]);
            return 1;
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

        gf2to256_add(c, a, b);
        _mm256_storeu_si256((__m256i *)t1, gf2to256v_add_u256(aa, bb));
        if (!gf2to256_is_equal(c, t1)) {
            printf("test_mul_u256: add\n");
            return 1;
        }

        _mm256_storeu_si256((__m256i *)t1, gf2to256v_sub_u256(aa, bb));
        if (!gf2to256_is_equal(c, t1)) {
            printf("test_mul_u256: sub\n");
            return 1;
        }
    }

    return 0;
}

DEFINE_TEST_VECTOR_SIMD(u256)
DEFINE_TEST_MATRIX_SIMD(u256)
#endif

#ifdef USE_AVX512
int test_arith_u512() {
    gf2to256 a[2], b[2], c[2], t[2];
    for (uint64_t i = 0; i < N; i++) {
        gf2to256_vector_random(a, 2);
        gf2to256_vector_random(b, 2);
        const __m512i aa = _mm512_loadu_si512((const void *)a);
        const __m512i bb = _mm512_loadu_si512((const void *)b);

        gf2to256_mul(c[0], a[0], b[0]);
        gf2to256_mul(c[1], a[1], b[1]);
        _mm512_storeu_si512((void *)t, gf2to256v_mul_u512(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u512: mul\n");
            return 1;
        }

        gf2to256_add(c[0], a[0], b[0]);
        gf2to256_add(c[1], a[1], b[1]);
        _mm512_storeu_si512((void *)t, gf2to256v_add_u512(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u512: add\n");
            return 1;
        }

        _mm512_storeu_si512((void *)t, gf2to256v_sub_u512(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u512: sub\n");
            return 1;
        }
    }
    return 0;
}

DEFINE_TEST_VECTOR_SIMD(u512)
DEFINE_TEST_MATRIX_SIMD(u512)
#endif

#ifdef USE_NEON
int test_arith_u128() {
    gf2to256 a, b, c, t;
    for (uint64_t i = 0; i < N; i++) {
        gf2to256_set_random(a);
        gf2to256_set_random(b);
        const uint64x2x2_t aa = gf2to256v_load_u128(a);
        const uint64x2x2_t bb = gf2to256v_load_u128(b);

        gf2to256_mul(c, a, b);
        gf2to256v_store_u128(t, gf2to256v_mul_u128(aa, bb));
        if (!gf2to256_is_equal(c, t)) {
            printf("test_arith_u128: mul\n");
            return 1;
        }

#ifdef __ARM_FEATURE_CRYPTO
        gf2to256v_mul_neon_v2(t, a, b);
        if (!gf2to256_is_equal(c, t)) {
            printf("test_arith_u128: mul_neon_v2\n");
            return 1;
        }
#endif

        gf2to256_add(c, a, b);
        gf2to256v_store_u128(t, gf2to256v_add_u128(aa, bb));
        if (!gf2to256_is_equal(c, t)) {
            printf("test_arith_u128: add\n");
            return 1;
        }

        gf2to256v_store_u128(t, gf2to256v_sub_u128(aa, bb));
        if (!gf2to256_is_equal(c, t)) {
            printf("test_arith_u128: sub\n");
            return 1;
        }
    }
    return 0;
}

DEFINE_TEST_VECTOR_SIMD(u128)
DEFINE_TEST_MATRIX_SIMD(u128)
#endif

int main() {
    srand(0x2256);
    if (test_arith()) { return 1; }
    if (test_vector()) { return 1; }
    if (test_matrix()) { return 1; }

#ifdef USE_AVX2
    if (test_mul_u256()) { return 1; }

    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_vector_u256()) { return 1; }
    if (test_matrix_u256()) { return 1; }
#endif

#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
    if (test_vector_u512()) { return 1; }
    if (test_matrix_u512()) { return 1; }
#endif

#ifdef USE_NEON
    if (test_arith_u128()) { return 1; }
    if (test_vector_u128()) { return 1; }
    if (test_matrix_u128()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
