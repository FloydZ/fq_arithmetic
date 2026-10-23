#include <stdint.h>
#include <stdbool.h>
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

static int vec_eq(const gf16to32 *a,
                  const gf16to32 *b,
                  const uint32_t n) {
    return memcmp(a, b, n * sizeof(gf16to32)) == 0;
}

/// \return random nibble packed gf16 buffer of `bytes` bytes
static gf16 *random_gf16_buffer(const uint32_t bytes) {
    gf16 *r = (gf16 *)malloc(bytes);
    for (uint32_t i = 0; i < bytes; i++) {
        r[i] = (gf16)rand();
    }
    return r;
}

bool test_add() {
    gf16to32 a, b, c;
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        gf16to32_set_u(a, 0);
        gf16to32_set_u(b, i);
        gf16to32_add(c, a, b);

        if (gf16to32_cmp(c, b) != 0) {
            printf("error gf16to32_add\n");
            return 1;
        }
    }

    return 0;
}

bool test_mul() {
    gf16to32 a, b, c1, d;
    gf16to32_set_u(d, 1);
    for (uint32_t i = 2; i < 1u << 12; ++i) {
        gf16to32_set_u(b, i);
        gf16to32_inv(a, b);
        gf16to32_mul(c1, a, b);

        if (gf16to32_cmp(c1, d) != 0) {
            printf("error gf16to32_mul\n");
            return 1;
        }
    }

    return 0;
}

int test_arith() {
    gf16to32 a, b, c, t1, t2, t3, zero, one;
    gf16to32_set_zero(zero);
    gf16to32_set_u(one, 1);

    // X^16 * X^16 = X^32 = B*X^28 + X^21 + F*X^2 + 9
    static const uint8_t x32[16] = {0x09, 0x0F, 0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0, 0, 0x0B, 0};
    gf16to32_set_zero(a); a[8] = 0x01;
    gf16to32_mul(t1, a, a);
    if (!gf16to32_is_equal(t1, x32)) {
        printf("test_arith: X^32\n");
        return 1;
    }

    for (uint64_t i = 0; i < N; i++) {
        gf16to32_set_random(a);
        gf16to32_set_random(b);
        gf16to32_set_random(c);

        gf16to32_mul(t1, a, b);
        gf16to32_mul(t2, b, a);
        if (!gf16to32_is_equal(t1, t2)) {
            printf("test_arith: mul not commutative\n");
            return 1;
        }

        // a*(b + c) = a*b + a*c
        gf16to32_add(t2, b, c);
        gf16to32_mul(t2, a, t2);
        gf16to32_mul(t3, a, c);
        gf16to32_add(t3, t3, t1);
        if (!gf16to32_is_equal(t2, t3)) {
            printf("test_arith: mul not distributive\n");
            return 1;
        }

        // (a*b)*c = a*(b*c)
        gf16to32_mul(t2, t1, c);
        gf16to32_mul(t3, b, c);
        gf16to32_mul(t3, a, t3);
        if (!gf16to32_is_equal(t2, t3)) {
            printf("test_arith: mul not associative\n");
            return 1;
        }

        // mul_gf16 = mul with an element of the sub field
        const gf16 s = rand() & 0xF;
        gf16to32_set_zero(t3); t3[0] = s;
        gf16to32_mul(t2, a, t3);
        gf16to32_mul_gf16(t3, a, s);
        if (!gf16to32_is_equal(t2, t3)) {
            printf("test_arith: mul_gf16\n");
            return 1;
        }

        gf16to32_add(t1, a, b);
        gf16to32_sub(t2, t1, b);
        if (!gf16to32_is_equal(t2, a)) {
            printf("test_arith: sub(add(a, b), b) != a\n");
            return 1;
        }

        gf16to32_neg(t1, a);
        gf16to32_add(t2, a, t1);
        if (!gf16to32_is_equal(t2, zero)) {
            printf("test_arith: a + neg(a) != 0\n");
            return 1;
        }
    }

    for (uint64_t i = 0; i < 16; i++) {
        gf16to32_set_random(a);
        gf16to32_inv(b, a);
        gf16to32_mul(c, a, b);
        if (!gf16to32_is_equal(c, one)) {
            printf("test_arith: inv(a)*a != 1\n");
            return 1;
        }
    }

    gf16to32_inv(b, zero);
    if (!gf16to32_is_equal(b, zero)) {
        printf("test_arith: inv(0) != 0\n");
        return 1;
    }
    return 0;
}

int test_vector() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        gf16to32 *a = gf16to32_vector_alloc(n);
        gf16to32 *b = gf16to32_vector_alloc(n);
        gf16to32 *r1 = gf16to32_vector_alloc(n);
        gf16to32 *r2 = gf16to32_vector_alloc(n);
        gf16to32 s, t, acc1, acc2;
        gf16 *g = random_gf16_buffer((n + 1) / 2);
        const gf16 gs = rand() & 0xF;

        gf16to32_vector_random(a, n);
        gf16to32_vector_random(b, n);
        gf16to32_set_random(s);

        for (uint32_t i = 0; i < n; i++) {
            gf16to32 x, y;
            gf16to32_set_random(x);
            gf16to32_vector_set(r1, i, x);
            gf16to32_vector_get(y, r1, i);
            if (!gf16to32_is_equal(x, y)) {
                printf("test_vector: get/set\n");
                return 1;
            }
        }

        // add
        gf16to32_vector_copy(r1, a, n);
        gf16to32_vector_add(r1, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf16to32_add(t, a[i], b[i]);
            if (!gf16to32_is_equal(t, r1[i])) {
                printf("test_vector: add %u\n", n);
                return 1;
            }
        }

        // scalar_add
        gf16to32_vector_copy(r1, a, n);
        gf16to32_vector_scalar_add(r1, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf16to32_mul(t, s, b[i]);
            gf16to32_add(t, t, a[i]);
            if (!gf16to32_is_equal(t, r1[i])) {
                printf("test_vector: scalar_add %u\n", n);
                return 1;
            }
        }

        // scalar_add_v2
        gf16to32_vector_zero(r2, n);
        gf16to32_vector_scalar_add_v2(r2, a, s, b, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_v2 %u\n", n);
            return 1;
        }

        // mul_acc
        gf16to32_set_zero(acc1);
        for (uint32_t i = 0; i < n; i++) {
            gf16to32_mul(t, a[i], b[i]);
            gf16to32_add(acc1, acc1, t);
        }
        gf16to32_vector_mul_acc(acc2, a, b, n);
        if (!gf16to32_is_equal(acc1, acc2)) {
            printf("test_vector: mul_acc %u\n", n);
            return 1;
        }

        // gf16 sub field
        gf16to32_vector_copy(r1, a, n);
        gf16to32_vector_add_gf16(r1, g, n);
        for (uint32_t i = 0; i < n; i++) {
            gf16to32_copy(t, a[i]);
            t[0] ^= gf16_vector_get(g, i);
            if (!gf16to32_is_equal(t, r1[i])) {
                printf("test_vector: add_gf16 %u\n", n);
                return 1;
            }
        }

#ifdef USE_AVX2
        gf16to32_vector_copy(r2, a, n);
        gf16to32_vector_add_gf16_u128(r2, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: add_gf16_u128 %u\n", n);
            return 1;
        }

        gf16to32_vector_zero(r2, n);
        gf16to32_vector_add_gf16_u256_v2(r2, a, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: add_gf16_u256_v2 %u\n", n);
            return 1;
        }

        gf16to32_vector_copy(r1, a, n);
        gf16to32_vector_scalar_add_gf16_v3(r1, s, g, n);
        gf16to32_vector_copy(r2, a, n);
        gf16to32_vector_scalar_add_gf16_u256_v2(r2, s, g, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_gf16_u256_v2 %u\n", n);
            return 1;
        }

        gf16to32_vector_copy(r1, a, n);
        gf16to32_vector_scalar_add_gf16(r1, gs, b, n);
        gf16to32_vector_copy(r2, a, n);
        gf16to32_vector_scalar_add_gf16_u256(r2, gs, b, n);
        if (!vec_eq(r1, r2, n)) {
            printf("test_vector: scalar_add_gf16_u256 %u\n", n);
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
        gf16to32 *a  = gf16to32_vector_alloc(n + 1);                            \
        gf16to32 *b  = gf16to32_vector_alloc(n + 1);                            \
        gf16to32 *r1 = gf16to32_vector_alloc(n + 1);                            \
        gf16to32 *r2 = gf16to32_vector_alloc(n + 1);                            \
        gf16to32 s, acc1, acc2;                                                 \
        gf16to32_vector_random(a, n + 1);                                       \
        gf16to32_vector_random(b, n + 1);                                       \
        gf16to32_vector_random(r1, n + 1);                                      \
        gf16to32_set_random(s);                                                 \
                                                                                \
        gf16to32_vector_copy(r2, r1, n + 1);                                    \
        gf16to32_vector_add(r1, a, n);                                          \
        gf16to32_vector_add_##X(r2, a, n);                                      \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": add %u\n", n);                          \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_vector_copy(r1, a, n);                                         \
        gf16to32_vector_add(r1, b, n);                                          \
        gf16to32_vector_add_##X##_v2(r2, a, b, n);                              \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": add_v2 %u\n", n);                       \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_vector_scalar_add(r1, s, b, n);                                \
        gf16to32_vector_scalar_add_##X(r2, s, b, n);                            \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": scalar_add %u\n", n);                   \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_vector_scalar_add_v2(r1, a, s, b, n);                          \
        gf16to32_vector_scalar_add_##X##_v2(r2, a, s, b, n);                    \
        if (!vec_eq(r1, r2, n + 1)) {                                           \
            printf("test_vector_" #X ": scalar_add_v2 %u\n", n);                \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_vector_mul_acc(acc1, a, b, n);                                 \
        gf16to32_vector_mul_acc_##X(acc2, a, b, n);                             \
        if (!gf16to32_is_equal(acc1, acc2)) {                                   \
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
        gf16to32 *A = gf16to32_matrix_alloc(nr, nk);
        gf16to32 *B = gf16to32_matrix_alloc(nk, nc);
        gf16to32 *C = gf16to32_matrix_alloc(nr, nc);
        gf16to32 *I = gf16to32_matrix_alloc(nr, nr);
        gf16to32 *R1 = gf16to32_matrix_alloc(nr, nk);
        gf16to32 *R2 = gf16to32_matrix_alloc(nr, nk);
        gf16to32 *T = gf16to32_matrix_alloc(nk, nr);
        gf16to32 *M1 = gf16to32_matrix_alloc(nr, nc);
        gf16to32 *M2 = gf16to32_matrix_alloc(nr, nc);
        gf16to32 s, x, y;
        gf16 *G = random_gf16_buffer(((nr + 1) / 2) * nk);

        gf16to32_matrix_random(A, nr, nk);
        gf16to32_matrix_random(B, nk, nc);
        gf16to32_matrix_random(C, nr, nc);
        gf16to32_set_random(s);

        // get/set
        gf16to32_matrix_copy(R1, A, nr, nk);
        gf16to32_set_random(x);
        gf16to32_matrix_set(R1, nr, nr - 1, nk - 1, x);
        gf16to32_matrix_get(y, R1, nr, nr - 1, nk - 1);
        if (!gf16to32_is_equal(x, y) || !gf16to32_is_equal(R1[nr*nk - 1], x)) {
            printf("test_matrix: get/set\n");
            return 1;
        }

        // id * A = A
        gf16to32_matrix_id(I, nr, nr);
        gf16to32_matrix_mul(R1, I, A, nr, nr, nk);
        if (!vec_eq(R1, A, nr*nk)) {
            printf("test_matrix: id\n");
            return 1;
        }

        // transpose
        gf16to32_matrix_transpose(T, A, nr, nk);
        gf16to32_matrix_get(x, A, nr, nr - 1, 0);
        gf16to32_matrix_get(y, T, nk, 0, nr - 1);
        if (!gf16to32_is_equal(x, y)) {
            printf("test_matrix: transpose entry\n");
            return 1;
        }
        gf16to32_matrix_transpose(R1, T, nk, nr);
        if (!vec_eq(R1, A, nr*nk)) {
            printf("test_matrix: transpose\n");
            return 1;
        }

        // add_mul == mul + add
        gf16to32_matrix_mul(M1, A, B, nr, nk, nc);
        gf16to32_matrix_add(M1, M1, C, nr, nc);
        gf16to32_matrix_copy(M2, C, nr, nc);
        gf16to32_matrix_add_mul(M2, A, B, nr, nk, nc);
        if (!vec_eq(M1, M2, nr*nc)) {
            printf("test_matrix: add_mul\n");
            return 1;
        }

        // add_scalar
        gf16to32_matrix_copy(R1, A, nr, nk);
        gf16to32_matrix_add_scalar(R1, s, A, nr, nk);
        gf16to32_matrix_add_scalar_v2(R2, A, s, A, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_scalar\n");
            return 1;
        }
        for (uint32_t i = 0; i < nr*nk; i++) {
            gf16to32_mul(x, s, A[i]);
            gf16to32_add(x, x, A[i]);
            if (!gf16to32_is_equal(x, R1[i])) {
                printf("test_matrix: add_scalar entry\n");
                return 1;
            }
        }

#ifdef USE_AVX2
        gf16to32_matrix_add_gf16(R1, G, A, nr, nk);
        gf16to32_matrix_add_gf16_u256(R2, G, A, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_gf16_u256\n");
            return 1;
        }

        gf16to32_matrix_copy(R1, A, nr, nk);
        gf16to32_matrix_add_scalar_gf16(R1, s, G, nr, nk);
        gf16to32_matrix_copy(R2, A, nr, nk);
        gf16to32_matrix_add_scalar_gf16_u256(R2, s, G, nr, nk);
        if (!vec_eq(R1, R2, nr*nk)) {
            printf("test_matrix: add_scalar_gf16_u256\n");
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
        gf16to32 *A = gf16to32_matrix_alloc(nr, nk);                            \
        gf16to32 *A2 = gf16to32_matrix_alloc(nr, nk);                           \
        gf16to32 *B = gf16to32_matrix_alloc(nk, nc);                            \
        gf16to32 *R1 = gf16to32_matrix_alloc(nr, nk);                           \
        gf16to32 *R2 = gf16to32_matrix_alloc(nr, nk);                           \
        gf16to32 *M1 = gf16to32_matrix_alloc(nr, nc);                           \
        gf16to32 *M2 = gf16to32_matrix_alloc(nr, nc);                           \
        gf16to32 s;                                                             \
        gf16to32_matrix_random(A, nr, nk);                                      \
        gf16to32_matrix_random(A2, nr, nk);                                     \
        gf16to32_matrix_random(B, nk, nc);                                      \
        gf16to32_matrix_random(M1, nr, nc);                                     \
        gf16to32_matrix_copy(M2, M1, nr, nc);                                   \
        gf16to32_set_random(s);                                                 \
                                                                                \
        gf16to32_matrix_add(R1, A, A2, nr, nk);                                 \
        gf16to32_matrix_add_##X(R2, A, A2, nr, nk);                             \
        if (!vec_eq(R1, R2, nr*nk)) {                                           \
            printf("test_matrix_" #X ": add\n");                                \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_matrix_add_scalar(R1, s, A2, nr, nk);                          \
        gf16to32_matrix_add_scalar_##X(R2, s, A2, nr, nk);                      \
        if (!vec_eq(R1, R2, nr*nk)) {                                           \
            printf("test_matrix_" #X ": add_scalar\n");                         \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_matrix_add_mul(M1, A, B, nr, nk, nc);                          \
        gf16to32_matrix_add_mul_##X(M2, A, B, nr, nk, nc);                      \
        if (!vec_eq(M1, M2, nr*nc)) {                                           \
            printf("test_matrix_" #X ": add_mul\n");                            \
            return 1;                                                           \
        }                                                                       \
                                                                                \
        gf16to32_matrix_mul(M1, A, B, nr, nk, nc);                              \
        gf16to32_matrix_mul_##X(M2, A, B, nr, nk, nc);                          \
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
int test_arith_u256() {
    gf16to32 a[2], b[2], c[2], t[2];
    for (uint64_t i = 0; i < N; i++) {
        gf16to32_vector_random(a, 2);
        gf16to32_vector_random(b, 2);
        const __m256i aa = _mm256_loadu_si256((const __m256i *)a);
        const __m256i bb = _mm256_loadu_si256((const __m256i *)b);

        gf16to32_mul(c[0], a[0], b[0]);
        gf16to32_mul(c[1], a[1], b[1]);
        _mm256_storeu_si256((__m256i *)t, gf16to32v_mul_u256(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u256: mul\n");
            return 1;
        }

        // SSE
        _mm_storeu_si128((__m128i *)t[0], gf16to32v_mul_u128(_mm_loadu_si128((const __m128i *)a[0]),
                                                             _mm_loadu_si128((const __m128i *)b[0])));
        if (!gf16to32_is_equal(c[0], t[0])) {
            printf("test_arith_u256: mul_u128 (sse)\n");
            return 1;
        }

        gf16to32_add(c[0], a[0], b[0]);
        gf16to32_add(c[1], a[1], b[1]);
        _mm256_storeu_si256((__m256i *)t, gf16to32v_add_u256(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u256: add\n");
            return 1;
        }

        _mm256_storeu_si256((__m256i *)t, gf16to32v_sub_u256(aa, bb));
        if (!vec_eq(c, t, 2)) {
            printf("test_arith_u256: sub\n");
            return 1;
        }

        // multiplication with an element of GF(16)
        const gf16 s = rand() & 0xF;
        gf16to32_mul_gf16(c[0], a[0], s);
        gf16to32v_mul_gf16_u128(t[0], a[0], s);
        if (!gf16to32_is_equal(c[0], t[0])) {
            printf("test_arith_u256: mul_gf16_u128\n");
            return 1;
        }

        // nibble-wise GF(16) multiplication
        _mm256_storeu_si256((__m256i *)t, gf16to32v_gf16_mul_u256(aa, bb));
        for (uint32_t k = 0; k < 16; k++) {
            const uint8_t e = gf16_mul(a[1][k] & 0xF, b[1][k] & 0xF) |
                              (gf16_mul(a[1][k] >> 4, b[1][k] >> 4) << 4);
            if (t[1][k] != e) {
                printf("test_arith_u256: gf16_mul_u256\n");
                return 1;
            }
        }
    }
    return 0;
}

DEFINE_TEST_VECTOR_SIMD(u256)
DEFINE_TEST_MATRIX_SIMD(u256)
#endif

#ifdef USE_AVX512
int test_arith_u512() {
    gf16to32 a[4], b[4], c[4], t[4];
    for (uint64_t i = 0; i < N; i++) {
        gf16to32_vector_random(a, 4);
        gf16to32_vector_random(b, 4);
        const __m512i aa = _mm512_loadu_si512((const void *)a);
        const __m512i bb = _mm512_loadu_si512((const void *)b);

        for (uint32_t k = 0; k < 4; k++) {
            gf16to32_mul(c[k], a[k], b[k]);
        }
        _mm512_storeu_si512((void *)t, gf16to32v_mul_u512(aa, bb));
        if (!vec_eq(c, t, 4)) {
            printf("test_arith_u512: mul\n");
            return 1;
        }

        for (uint32_t k = 0; k < 4; k++) {
            gf16to32_add(c[k], a[k], b[k]);
        }
        _mm512_storeu_si512((void *)t, gf16to32v_add_u512(aa, bb));
        if (!vec_eq(c, t, 4)) {
            printf("test_arith_u512: add\n");
            return 1;
        }

        _mm512_storeu_si512((void *)t, gf16to32v_sub_u512(aa, bb));
        if (!vec_eq(c, t, 4)) {
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
    gf16to32 a, b, c, t;
    for (uint64_t i = 0; i < N; i++) {
        gf16to32_set_random(a);
        gf16to32_set_random(b);
        const uint8x16_t aa = vld1q_u8(a);
        const uint8x16_t bb = vld1q_u8(b);

        gf16to32_mul(c, a, b);
        vst1q_u8(t, gf16to32v_mul_u128(aa, bb));
        if (!gf16to32_is_equal(c, t)) {
            printf("test_arith_u128: mul\n");
            return 1;
        }

        gf16to32_add(c, a, b);
        vst1q_u8(t, gf16to32v_add_u128(aa, bb));
        if (!gf16to32_is_equal(c, t)) {
            printf("test_arith_u128: add\n");
            return 1;
        }

        vst1q_u8(t, gf16to32v_sub_u128(aa, bb));
        if (!gf16to32_is_equal(c, t)) {
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
    srand(0x1632);
    if (test_add()) { return 1; }
    if (test_mul()) { return 1; }
    if (test_arith()) { return 1; }
    if (test_vector()) { return 1; }
    if (test_matrix()) { return 1; }

#ifdef USE_AVX2
    if (test_arith_u256()) { return 1; }
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
