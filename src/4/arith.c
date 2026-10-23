#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"
#define TEST_SIZE 19999

/// vector lengths (in elements) used by the vector tests
static const uint32_t test_lengths[] = {
    1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
    255, 256, 257, 511, 512, 513, 1000, 1025
};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// matrix shapes {n_rows1, n_cols1, n_cols2} used by the matrix tests
static const uint32_t test_shapes[][3] = {
    {1, 1, 1}, {3, 5, 2}, {4, 4, 4}, {17, 9, 33}, {33, 40, 7}, {64, 64, 64},
    {100, 3, 70}, {129, 65, 9}, {300, 20, 1}
};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

/// fills `n` bytes with random data (including the unused bits)
static void fill_raw(uint8_t *a, const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) { a[i] = rand(); }
}

void print_mul_table() {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const ff_t c = gf4_mul(i, j);
            printf("%d = %d * %d\n", c, i, j);
        }
    }
}
bool test1() {
	if (gf4_add(0, 1) != 1) { return false; }
	if (gf4_add(0, 2) != 2) { return false; }
	if (gf4_add(0, 3) != 3) { return false; }
	if (gf4_add(0, 0) != 0) { return false; }

    if (gf4_mul(0, 0) != 0) { return false; }
    if (gf4_mul(0, 1) != 0) { return false; }
    if (gf4_mul(0, 2) != 0) { return false; }
    if (gf4_mul(0, 3) != 0) { return false; }
	return true;
}

/// tests add, sub, neg, inv and the packed byte multiplication
bool test_arith() {
    for (gf4 a = 0; a < 4; a++) {
        if (gf4_add(a, gf4_neg(a)) != 0) { printf("error test_arith: neg\n"); return false; }
        if (a && (gf4_mul(a, gf4_inv(a)) != 1)) { printf("error test_arith: inv %d\n", a); return false; }
        for (gf4 b = 0; b < 4; b++) {
            if (gf4_sub(gf4_add(a, b), b) != a) { printf("error test_arith: sub\n"); return false; }
            if (gf4_mul(a, b) != gf4_mul(b, a)) { printf("error test_arith: mul comm\n"); return false; }
            for (gf4 c = 0; c < 4; c++) {
                // distributivity and associativity
                if (gf4_mul(a, gf4_add(b, c)) != gf4_add(gf4_mul(a, b), gf4_mul(a, c))) {
                    printf("error test_arith: distributivity\n"); return false;
                }
                if (gf4_mul(a, gf4_mul(b, c)) != gf4_mul(gf4_mul(a, b), c)) {
                    printf("error test_arith: associativity\n"); return false;
                }
            }
        }
    }
    if (gf4_inv(0) != 0) { printf("error test_arith: inv(0)\n"); return false; }

    // all pairs of bytes
    for (uint32_t a = 0; a < 256; a++) {
        for (uint32_t b = 0; b < 256; b++) {
            const uint8_t c = gf4v_mul_u8(a, b);
            for (uint32_t i = 0; i < 4; i++) {
                if (((c >> (2*i)) & 3u) != gf4_mul((a >> (2*i)) & 3u, (b >> (2*i)) & 3u)) {
                    printf("error test_arith: mul_u8\n"); return false;
                }
            }
        }
    }
    return true;
}

bool test_mul_u64(){
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j1 = 0; j1 < 4; ++j1) {
            for (uint32_t j2 = 0; j2 < 4; ++j2) {
                uint64_t a = ((uint64_t)j1) << (i * 2u);
                uint64_t b = ((uint64_t)j2) << (i * 2u);

                const ff_t t1 = gf4v_mul_u64(a, b) >> (i*2);
                const ff_t t2 = gf4_mul(j1, j2);
                const ff_t t3 = gf4_mul_v2(j1, j2);
                if (t1 != t2) {
                    return false;
                }

                if (t2 != t3) {
                    return false;
                }
            }
        }
    }

    return true;
}

/// checks the reference vector functions against their element wise definition
bool test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        const uint32_t nb = gf4_vector_size(n);
        gf4 *a = (gf4 *)malloc(nb), *b = (gf4 *)malloc(nb),
            *r = (gf4 *)malloc(nb), *o = (gf4 *)malloc(nb);
        fill_raw(a, nb); fill_raw(b, nb); fill_raw(o, nb);

        for (gf4 c = 0; c < 4; c++) {
            memcpy(r, o, nb);
            gf4_vector_scalar_add(r, c, a, n);
            for (uint32_t i = 0; i < 4*nb; i++) {
                const gf4 e = (i < n) ? (gf4_vector_get(o, i) ^ gf4_mul(c, gf4_vector_get(a, i)))
                                      : gf4_vector_get(o, i);
                if (gf4_vector_get(r, i) != e) {
                    printf("error test_vector_ref: scalar_add %u %u\n", n, i);
                    return false;
                }
            }

            memcpy(r, o, nb);
            gf4_vector_scalar_add_v2(r, a, c, b, n);
            for (uint32_t i = 0; i < 4*nb; i++) {
                const gf4 e = (i < n) ? (gf4_vector_get(a, i) ^ gf4_mul(c, gf4_vector_get(b, i)))
                                      : gf4_vector_get(o, i);
                if (gf4_vector_get(r, i) != e) {
                    printf("error test_vector_ref: scalar_add_v2 %u %u\n", n, i);
                    return false;
                }
            }
        }

        memcpy(r, o, nb);
        gf4_vector_add(r, a, n);
        gf4 acc = 0;
        for (uint32_t i = 0; i < 4*nb; i++) {
            const gf4 e = (i < n) ? (gf4_vector_get(o, i) ^ gf4_vector_get(a, i))
                                  : gf4_vector_get(o, i);
            if (gf4_vector_get(r, i) != e) {
                printf("error test_vector_ref: add %u %u\n", n, i);
                return false;
            }
            if (i < n) { acc ^= gf4_mul(gf4_vector_get(a, i), gf4_vector_get(b, i)); }
        }

        if (gf4_vector_mul_acc(a, b, n) != acc) {
            printf("error test_vector_ref: mul_acc %u\n", n);
            return false;
        }

        memcpy(r, o, nb);
        gf4_vector_copy(r, a, n);
        for (uint32_t i = 0; i < 4*nb; i++) {
            const gf4 e = (i < n) ? gf4_vector_get(a, i) : gf4_vector_get(o, i);
            if (gf4_vector_get(r, i) != e) {
                printf("error test_vector_ref: copy %u %u\n", n, i);
                return false;
            }
        }

        // get/set
        memcpy(r, o, nb);
        for (uint32_t i = 0; i < n; i++) {
            gf4_vector_set(r, i, gf4_vector_get(a, i));
        }
        for (uint32_t i = 0; i < 4*nb; i++) {
            const gf4 e = (i < n) ? gf4_vector_get(a, i) : gf4_vector_get(o, i);
            if (gf4_vector_get(r, i) != e) {
                printf("error test_vector_ref: set %u %u\n", n, i);
                return false;
            }
        }

        free(a); free(b); free(r); free(o);
    }
    return true;
}

typedef void (*vec_add_t)(gf4 *, const gf4 *, uint32_t);
typedef void (*vec_add_v2_t)(gf4 *, const gf4 *, const gf4 *, uint32_t);
typedef void (*vec_scalar_add_t)(gf4 *, gf4, const gf4 *, uint32_t);
typedef void (*vec_scalar_add_v2_t)(gf4 *, const gf4 *, gf4, const gf4 *, uint32_t);
typedef gf4  (*vec_mul_acc_t)(const gf4 *, const gf4 *, uint32_t);

/// compares the simd vector functions against the reference functions
bool test_vector_simd(const char *name,
                      vec_add_t add,
                      vec_add_v2_t add_v2,
                      vec_scalar_add_t scalar_add,
                      vec_scalar_add_v2_t scalar_add_v2,
                      vec_mul_acc_t mul_acc) {
    bool ret = true;
    for (uint32_t l = 0; (l < NR_TEST_LENGTHS) && ret; l++) {
        const uint32_t n = test_lengths[l];
        const uint32_t nb = gf4_vector_size(n);
        // exact sizes, s.t. the address sanitizer catches out of bound accesses
        gf4 *a  = (gf4 *)malloc(nb), *b  = (gf4 *)malloc(nb),
            *r1 = (gf4 *)malloc(nb), *r2 = (gf4 *)malloc(nb);
        fill_raw(a, nb); fill_raw(b, nb); fill_raw(r1, nb);
        memcpy(r2, r1, nb);

        gf4_vector_add(r1, a, n);
        add(r2, a, n);
        if (memcmp(r1, r2, nb)) { printf("error %s: add %u\n", name, n); ret = false; }

        gf4_vector_add_v2(r1, a, b, n);
        add_v2(r2, a, b, n);
        if (memcmp(r1, r2, nb)) { printf("error %s: add_v2 %u\n", name, n); ret = false; }

        for (gf4 c = 0; c < 4; c++) {
            gf4_vector_scalar_add(r1, c, a, n);
            scalar_add(r2, c, a, n);
            if (memcmp(r1, r2, nb)) { printf("error %s: scalar_add %u\n", name, n); ret = false; }

            gf4_vector_scalar_add_v2(r1, b, c, a, n);
            scalar_add_v2(r2, b, c, a, n);
            if (memcmp(r1, r2, nb)) { printf("error %s: scalar_add_v2 %u\n", name, n); ret = false; }
        }

        if (gf4_vector_mul_acc(a, b, n) != mul_acc(a, b, n)) {
            printf("error %s: mul_acc %u\n", name, n);
            ret = false;
        }

        free(a); free(b); free(r1); free(r2);
    }
    return ret;
}

/// \return true if m1 == m2
static bool matrix_eq(const gf4 *m1, const gf4 *m2, const uint32_t nrows, const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            if (gf4_matrix_get(m1, nrows, i, j) != gf4_matrix_get(m2, nrows, i, j)) { return false; }
        }
    }
    return true;
}

/// checks the reference matrix functions: id*A == A, (A^T)^T == A,
/// transpose against get/set, add_mul == mul + add, add_scalar, copy
bool test_matrix_ref() {
    bool ret = true;
    for (uint32_t s = 0; (s < NR_TEST_SHAPES) && ret; s++) {
        const uint32_t nrows = test_shapes[s][0],
                       ncols = test_shapes[s][1],
                       ncols2 = test_shapes[s][2];
        gf4 *I  = gf4_matrix_alloc(nrows, nrows);
        gf4 *A  = gf4_matrix_alloc(nrows, ncols);
        gf4 *B  = gf4_matrix_alloc(ncols, ncols2);
        gf4 *C1 = gf4_matrix_alloc(nrows, ncols2);
        gf4 *C2 = gf4_matrix_alloc(nrows, ncols2);
        gf4 *C3 = gf4_matrix_alloc(nrows, ncols2);
        gf4 *T  = gf4_matrix_alloc(ncols, nrows);
        gf4 *A2 = gf4_matrix_alloc(nrows, ncols);

        gf4_matrix_random(A, nrows, ncols);
        gf4_matrix_random(B, ncols, ncols2);
        gf4_matrix_random(C1, nrows, ncols2);

        // id * A == A
        gf4_matrix_id(I, nrows, nrows);
        gf4_matrix_mul(A2, I, A, nrows, nrows, ncols);
        if (!matrix_eq(A, A2, nrows, ncols)) {
            printf("error test_matrix_ref: id %u %u\n", nrows, ncols);
            ret = false;
        }

        // transpose
        gf4_matrix_transpose(T, A, nrows, ncols);
        for (uint32_t i = 0; i < nrows; i++) {
            for (uint32_t j = 0; j < ncols; j++) {
                if (gf4_matrix_get(T, ncols, j, i) != gf4_matrix_get(A, nrows, i, j)) {
                    printf("error test_matrix_ref: transpose %u %u\n", nrows, ncols);
                    ret = false;
                }
            }
        }
        gf4_matrix_zero(A2, nrows, ncols);
        gf4_matrix_transpose(A2, T, ncols, nrows);
        if (!matrix_eq(A, A2, nrows, ncols)) {
            printf("error test_matrix_ref: transpose^2 %u %u\n", nrows, ncols);
            ret = false;
        }

        // C2 = C1 + A*B, C3 = C1; C3 += A*B
        gf4_matrix_mul(C2, A, B, nrows, ncols, ncols2);
        gf4_matrix_add(C2, C2, C1, nrows, ncols2);
        gf4_matrix_copy(C3, C1, nrows, ncols2);
        gf4_matrix_add_mul(C3, A, B, nrows, ncols, ncols2);
        if (!matrix_eq(C2, C3, nrows, ncols2)) {
            printf("error test_matrix_ref: add_mul %u %u %u\n", nrows, ncols, ncols2);
            ret = false;
        }

        // C3 = C1 + c*A*B, element wise
        for (gf4 c = 0; c < 4; c++) {
            gf4_matrix_mul(C2, A, B, nrows, ncols, ncols2);
            gf4_matrix_copy(C3, C1, nrows, ncols2);
            gf4_matrix_add_scalar(C3, c, C2, nrows, ncols2);
            for (uint32_t i = 0; i < nrows; i++) {
                for (uint32_t j = 0; j < ncols2; j++) {
                    const gf4 e = gf4_matrix_get(C1, nrows, i, j) ^ gf4_mul(c, gf4_matrix_get(C2, nrows, i, j));
                    if (gf4_matrix_get(C3, nrows, i, j) != e) {
                        printf("error test_matrix_ref: add_scalar %u %u %u\n", nrows, ncols, ncols2);
                        ret = false;
                    }
                }
            }
        }

        free(I); free(A); free(B); free(C1); free(C2); free(C3); free(T); free(A2);
    }
    return ret;
}

typedef void (*mat_add_t)(gf4 *, const gf4 *, const gf4 *, uint32_t, uint32_t);
typedef void (*mat_add_scalar_t)(gf4 *, gf4, const gf4 *, uint32_t, uint32_t);
typedef void (*mat_mul_t)(gf4 *, const gf4 *, const gf4 *, uint32_t, uint32_t, uint32_t);

/// compares the simd matrix functions against the reference functions
bool test_matrix_simd(const char *name,
                      mat_add_t add,
                      mat_add_scalar_t add_scalar,
                      mat_mul_t mul,
                      mat_mul_t add_mul) {
    bool ret = true;
    for (uint32_t s = 0; (s < NR_TEST_SHAPES) && ret; s++) {
        const uint32_t nrows = test_shapes[s][0],
                       ncols = test_shapes[s][1],
                       ncols2 = test_shapes[s][2];
        gf4 *A  = gf4_matrix_alloc(nrows, ncols);
        gf4 *A2 = gf4_matrix_alloc(nrows, ncols);
        gf4 *B  = gf4_matrix_alloc(ncols, ncols2);
        gf4 *C1 = gf4_matrix_alloc(nrows, ncols2);
        gf4 *C2 = gf4_matrix_alloc(nrows, ncols2);
        gf4 *D1 = gf4_matrix_alloc(nrows, ncols);
        gf4 *D2 = gf4_matrix_alloc(nrows, ncols);

        gf4_matrix_random(A, nrows, ncols);
        gf4_matrix_random(A2, nrows, ncols);
        gf4_matrix_random(B, ncols, ncols2);

        gf4_matrix_add(D1, A, A2, nrows, ncols);
        add(D2, A, A2, nrows, ncols);
        if (!matrix_eq(D1, D2, nrows, ncols)) { printf("error %s: add %u %u\n", name, nrows, ncols); ret = false; }

        for (gf4 c = 0; c < 4; c++) {
            gf4_matrix_add_scalar(D1, c, A, nrows, ncols);
            add_scalar(D2, c, A, nrows, ncols);
            if (!matrix_eq(D1, D2, nrows, ncols)) { printf("error %s: add_scalar %u %u\n", name, nrows, ncols); ret = false; }
        }

        gf4_matrix_mul(C1, A, B, nrows, ncols, ncols2);
        gf4_matrix_random(C2, nrows, ncols2);
        mul(C2, A, B, nrows, ncols, ncols2);
        if (!matrix_eq(C1, C2, nrows, ncols2)) { printf("error %s: mul %u %u %u\n", name, nrows, ncols, ncols2); ret = false; }

        gf4_matrix_random(C1, nrows, ncols2);
        gf4_matrix_copy(C2, C1, nrows, ncols2);
        gf4_matrix_add_mul(C1, A, B, nrows, ncols, ncols2);
        add_mul(C2, A, B, nrows, ncols, ncols2);
        if (!matrix_eq(C1, C2, nrows, ncols2)) { printf("error %s: add_mul %u %u %u\n", name, nrows, ncols, ncols2); ret = false; }

        free(A); free(A2); free(B); free(C1); free(C2); free(D1); free(D2);
    }
    return ret;
}

#ifdef USE_AVX2
bool test_mul_u256() {
    uint64_t tmp[4] = {0};
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j1 = 0; j1 < 4; ++j1) {
            for (uint32_t j2 = 0; j2 < 4; ++j2) {
                uint64_t a = ((uint64_t)j1) << (i * 2u);
                uint64_t b = ((uint64_t)j2) << (i * 2u);

                __m256i t1 = {(long long)a,0,0,0};
                __m256i t2 = {(long long)b,0,0,0};

                const __m256i c = gf4v_mul_u256(t1, t2);
                _mm256_storeu_si256((__m256i *)tmp, c);

                const ff_t g = gf4_mul(j1, j2);
                if ((tmp[0] >> (i*2u)) != g) {
                    printf("mul_u256 failed\n");
                    return false;
                }
            }
        }
    }

    return true;
}

bool test_scalar_u256() {
    uint64_t tmp[4] = {0};
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j1 = 0; j1 < 4; ++j1) {
            for (uint32_t j2 = 0; j2 < 4; ++j2) {
                uint64_t a = ((uint64_t)j1) << (i * 2u);

                __m256i t1 = {(long long)a,0,0,0};

                const __m256i c = gf4v_scalar_u256(t1, j2);
                _mm256_storeu_si256((__m256i *)tmp, c);

                const ff_t g = gf4_mul(j1, j2);
                if ((tmp[0] >> (i*2u)) != g) {
                    printf("scalar_u256 failed\n");
                    return false;
                }
            }
        }
    }

    return true;
}
#endif

int main() {
    srand(0x4);
	if (!test1()) { return -1; }
    if (!test_mul_u64()) { return -1; }
    if (!test_arith()) { return -1; }
    if (!test_vector_ref()) { return -1; }
    if (!test_matrix_ref()) { return -1; }

#ifdef USE_AVX2
    if (!test_mul_u256()) { return -1; }
    if (!test_scalar_u256()) { return -1; }
    if (!test_vector_simd("vector_u256", gf4_vector_add_u256, gf4_vector_add_u256_v2,
                          gf4_vector_scalar_add_u256, gf4_vector_scalar_add_u256_v2,
                          gf4_vector_mul_acc_u256)) { return -1; }
    if (!test_matrix_simd("matrix_u256", gf4_matrix_add_u256, gf4_matrix_add_scalar_u256,
                          gf4_matrix_mul_u256, gf4_matrix_add_mul_u256)) { return -1; }
#endif

#ifdef USE_NEON
    if (!test_vector_simd("vector_u128", gf4_vector_add_u128, gf4_vector_add_u128_v2,
                          gf4_vector_scalar_add_u128, gf4_vector_scalar_add_u128_v2,
                          gf4_vector_mul_acc_u128)) { return -1; }
    if (!test_matrix_simd("matrix_u128", gf4_matrix_add_u128, gf4_matrix_add_scalar_u128,
                          gf4_matrix_mul_u128, gf4_matrix_add_mul_u128)) { return -1; }
#endif

#ifdef USE_AVX512
    if (!test_vector_simd("vector_u512", gf4_vector_add_u512, gf4_vector_add_u512_v2,
                          gf4_vector_scalar_add_u512, gf4_vector_scalar_add_u512_v2,
                          gf4_vector_mul_acc_u512)) { return -1; }
    if (!test_matrix_simd("matrix_u512", gf4_matrix_add_u512, gf4_matrix_add_scalar_u512,
                          gf4_matrix_mul_u512, gf4_matrix_add_mul_u512)) { return -1; }
#endif

    printf("all good\n");
	return 0;
}
