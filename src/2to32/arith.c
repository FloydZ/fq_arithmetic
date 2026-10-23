#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint64_t N = 1u<<10u;

static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
static const uint32_t nr_test_lengths = sizeof(test_lengths)/sizeof(test_lengths[0]);
/// {n_rows1, n_cols1, n_cols2}
static const uint32_t test_shapes[][3] = {{1,1,1},{3,5,2},{17,9,4},{33,40,7},{8,8,8},{16,3,17}};
static const uint32_t nr_test_shapes = sizeof(test_shapes)/sizeof(test_shapes[0]);

static gf2to32 rand_elem(void) {
    return (gf2to32)rand() ^ ((gf2to32)rand() << 16u);
}

static int vector_cmp(const gf2to32 *a,
                      const gf2to32 *b,
                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (a[i] != b[i]) { return 1; }
    }
    return 0;
}

uint32_t test_add() {
    for (uint64_t a = 0; a < N; ++a) {
        for (uint64_t b = 0; b < N; ++b) {
            uint32_t i = rand();
            uint32_t j = rand();

            const uint32_t c1 = gf2to32_mul(i, j);
            const uint32_t c2 = gf2to32_mul_v2(i, j);

            if (c1 != c2){
                printf("error: test add: %d %d\n", c1, c2);
                return 1;
            }
        }
    }

    return 0;
}

/// a^(2^32) = a holds only if MODULUS is irreducible
uint32_t test_mul_field() {
    for (uint64_t t = 0; t < N; ++t) {
        const gf2to32 a = rand() ^ ((uint32_t)rand() << 16u);
        gf2to32 b = a;
        for (uint32_t k = 0; k < 32; ++k) {
            b = gf2to32_mul(b, b);
        }

        if (a != b) {
            printf("error: test mul field: %u %u\n", a, b);
            return 1;
        }
    }

    return 0;
}

/// tests add, sub, neg, inv
int test_arith_ref() {
    if (gf2to32_inv(0) != 0) {
        printf("error: test_arith_ref: inv(0)\n");
        return 1;
    }

    for (uint64_t t = 0; t < N; ++t) {
        const gf2to32 a = rand_elem();
        const gf2to32 b = rand_elem();
        if (gf2to32_add(a, gf2to32_neg(a)) != 0) {
            printf("error: test_arith_ref: neg %x\n", a);
            return 1;
        }
        if (gf2to32_sub(gf2to32_add(a, b), b) != a) {
            printf("error: test_arith_ref: sub %x %x\n", a, b);
            return 1;
        }
        if (a && (gf2to32_mul(a, gf2to32_inv(a)) != 1)) {
            printf("error: test_arith_ref: inv %x\n", a);
            return 1;
        }
    }

    return 0;
}

/// tests vector_mul_acc, vector_scalar_add_v2, vector_get/set
int test_vector_ref() {
    const uint32_t n = 77;
    gf2to32 *a = gf2to32_vector_alloc(n);
    gf2to32 *b = gf2to32_vector_alloc(n);
    gf2to32 *c1 = gf2to32_vector_alloc(n);
    gf2to32 *c2 = gf2to32_vector_alloc(n);
    int ret = 0;

    for (uint32_t t = 0; t < 64; t++) {
        gf2to32_vector_random(a, n);
        gf2to32_vector_random(b, n);
        const gf2to32 s = rand_elem();

        gf2to32 acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            acc ^= gf2to32_mul(gf2to32_vector_get(a, i), gf2to32_vector_get(b, i));
        }
        if (acc != gf2to32_vector_mul_acc(a, b, n)) {
            printf("error: test_vector_ref: mul_acc\n");
            ret = 1; goto finish;
        }

        gf2to32_vector_copy(c1, a, n);
        gf2to32_vector_scalar_add(c1, s, b, n);
        gf2to32_vector_scalar_add_v2(c2, a, s, b, n);
        if (vector_cmp(c1, c2, n)) {
            printf("error: test_vector_ref: scalar_add_v2\n");
            ret = 1; goto finish;
        }

        gf2to32_vector_set(c1, t % n, s);
        if (gf2to32_vector_get(c1, t % n) != s) {
            printf("error: test_vector_ref: get/set\n");
            ret = 1; goto finish;
        }
    }

finish:
    free(a); free(b); free(c1); free(c2);
    return ret;
}

/// tests matrix_id, matrix_mul, matrix_add_mul, matrix_transpose, matrix_get/set
int test_matrix_ref() {
    int ret = 0;
    for (uint32_t s = 0; s < nr_test_shapes; s++) {
        const uint32_t r = test_shapes[s][0], k = test_shapes[s][1], c = test_shapes[s][2];
        gf2to32 *A  = gf2to32_matrix_alloc(r, k);
        gf2to32 *AT = gf2to32_matrix_alloc(k, r);
        gf2to32 *A2 = gf2to32_matrix_alloc(r, k);
        gf2to32 *Ik = gf2to32_matrix_alloc(k, k);
        gf2to32 *Ir = gf2to32_matrix_alloc(r, r);
        gf2to32 *B  = gf2to32_matrix_alloc(k, c);
        gf2to32 *C1 = gf2to32_matrix_alloc(r, c);
        gf2to32 *C2 = gf2to32_matrix_alloc(r, c);
        gf2to32 *T  = gf2to32_matrix_alloc(r, c);

        gf2to32_matrix_random(A, r, k);
        gf2to32_matrix_random(B, k, c);
        gf2to32_matrix_random(C1, r, c);
        gf2to32_matrix_id(Ik, k, k);
        gf2to32_matrix_id(Ir, r, r);

        gf2to32_matrix_mul(A2, A, Ik, r, k, k);
        if (vector_cmp(A, A2, r*k)) {
            printf("error: test_matrix_ref: A*I\n");
            ret = 1; goto finish;
        }
        gf2to32_matrix_mul(A2, Ir, A, r, r, k);
        if (vector_cmp(A, A2, r*k)) {
            printf("error: test_matrix_ref: I*A\n");
            ret = 1; goto finish;
        }

        gf2to32_matrix_transpose(AT, A, r, k);
        gf2to32_matrix_transpose(A2, AT, k, r);
        if (vector_cmp(A, A2, r*k) ||
            gf2to32_matrix_get(AT, k, k-1, 0) != gf2to32_matrix_get(A, r, 0, k-1)) {
            printf("error: test_matrix_ref: transpose\n");
            ret = 1; goto finish;
        }

        gf2to32_matrix_copy(C2, C1, r, c);
        gf2to32_matrix_add_mul(C1, A, B, r, k, c);
        gf2to32_matrix_mul(T, A, B, r, k, c);
        gf2to32_matrix_add(C2, C2, T, r, c);
        if (vector_cmp(C1, C2, r*c)) {
            printf("error: test_matrix_ref: add_mul\n");
            ret = 1; goto finish;
        }

        const gf2to32 v = rand_elem();
        gf2to32_matrix_set(C1, r, r-1, c-1, v);
        if (gf2to32_matrix_get(C1, r, r-1, c-1) != v || C1[r*c - 1] != v) {
            printf("error: test_matrix_ref: get/set\n");
            ret = 1; goto finish;
        }

    finish:
        free(A); free(AT); free(A2); free(Ik); free(Ir);
        free(B); free(C1); free(C2); free(T);
        if (ret) { return ret; }
    }

    return 0;
}

/// collection of the vectorized base functions of one SIMD target
typedef struct {
    const char *name;
    void (*vector_add)(gf2to32 *, const gf2to32 *, uint32_t);
    void (*vector_add_v2)(gf2to32 *, const gf2to32 *, const gf2to32 *, uint32_t);
    void (*vector_scalar_add)(gf2to32 *, gf2to32, const gf2to32 *, uint32_t);
    void (*vector_scalar_add_v2)(gf2to32 *, const gf2to32 *, gf2to32, const gf2to32 *, uint32_t);
    gf2to32 (*vector_mul_acc)(const gf2to32 *, const gf2to32 *, uint32_t);
    void (*matrix_add)(gf2to32 *, const gf2to32 *, const gf2to32 *, uint32_t, uint32_t);
    void (*matrix_add_scalar)(gf2to32 *, gf2to32, const gf2to32 *, uint32_t, uint32_t);
    void (*matrix_mul)(gf2to32 *, const gf2to32 *, const gf2to32 *, uint32_t, uint32_t, uint32_t);
    void (*matrix_add_mul)(gf2to32 *, const gf2to32 *, const gf2to32 *, uint32_t, uint32_t, uint32_t);
} simd_impl;

static const simd_impl simd_impls[] = {
#ifdef USE_AVX2
    {"u256", gf2to32_vector_add_u256, gf2to32_vector_add_u256_v2,
     gf2to32_vector_scalar_add_u256, gf2to32_vector_scalar_add_u256_v2, gf2to32_vector_mul_acc_u256,
     gf2to32_matrix_add_u256, gf2to32_matrix_add_scalar_u256, gf2to32_matrix_mul_u256, gf2to32_matrix_add_mul_u256},
#endif
#ifdef USE_AVX512
    {"u512", gf2to32_vector_add_u512, gf2to32_vector_add_u512_v2,
     gf2to32_vector_scalar_add_u512, gf2to32_vector_scalar_add_u512_v2, gf2to32_vector_mul_acc_u512,
     gf2to32_matrix_add_u512, gf2to32_matrix_add_scalar_u512, gf2to32_matrix_mul_u512, gf2to32_matrix_add_mul_u512},
#endif
#ifdef USE_NEON
    {"u128", gf2to32_vector_add_u128, gf2to32_vector_add_u128_v2,
     gf2to32_vector_scalar_add_u128, gf2to32_vector_scalar_add_u128_v2, gf2to32_vector_mul_acc_u128,
     gf2to32_matrix_add_u128, gf2to32_matrix_add_scalar_u128, gf2to32_matrix_mul_u128, gf2to32_matrix_add_mul_u128},
#endif
    {NULL}
};

/// compares the vectorized vector functions against the reference ones
int test_vector_simd(const simd_impl *impl) {
    for (uint32_t l = 0; l < nr_test_lengths; l++) {
        const uint32_t n = test_lengths[l];
        gf2to32 *a  = gf2to32_vector_alloc(n);
        gf2to32 *b  = gf2to32_vector_alloc(n);
        gf2to32 *c1 = gf2to32_vector_alloc(n);
        gf2to32 *c2 = gf2to32_vector_alloc(n);
        const char *err = NULL;

        for (uint32_t t = 0; t < 16; t++) {
            gf2to32_vector_random(a, n);
            gf2to32_vector_random(b, n);
            gf2to32_vector_random(c1, n);
            gf2to32_vector_copy(c2, c1, n);
            const gf2to32 s = rand_elem();

            gf2to32_vector_add(c1, a, n);
            impl->vector_add(c2, a, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add"; break; }

            gf2to32_vector_copy(c1, a, n);
            gf2to32_vector_add(c1, b, n);
            impl->vector_add_v2(c2, a, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add_v2"; break; }

            gf2to32_vector_scalar_add(c1, s, a, n);
            impl->vector_scalar_add(c2, s, a, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add"; break; }

            gf2to32_vector_scalar_add_v2(c1, a, s, b, n);
            impl->vector_scalar_add_v2(c2, a, s, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add_v2"; break; }

            if (gf2to32_vector_mul_acc(a, b, n) != impl->vector_mul_acc(a, b, n)) {
                err = "vector_mul_acc"; break;
            }
        }

        free(a); free(b); free(c1); free(c2);
        if (err) {
            printf("error: test_vector_simd(%s): %s, n=%u\n", impl->name, err, n);
            return 1;
        }
    }

    return 0;
}

/// compares the vectorized matrix functions against the reference ones
int test_matrix_simd(const simd_impl *impl) {
    for (uint32_t s = 0; s < nr_test_shapes; s++) {
        const uint32_t r = test_shapes[s][0], k = test_shapes[s][1], c = test_shapes[s][2];
        gf2to32 *A  = gf2to32_matrix_alloc(r, k);
        gf2to32 *D  = gf2to32_matrix_alloc(r, k);
        gf2to32 *D1 = gf2to32_matrix_alloc(r, k);
        gf2to32 *D2 = gf2to32_matrix_alloc(r, k);
        gf2to32 *B  = gf2to32_matrix_alloc(k, c);
        gf2to32 *C1 = gf2to32_matrix_alloc(r, c);
        gf2to32 *C2 = gf2to32_matrix_alloc(r, c);
        const char *err = NULL;

        for (uint32_t t = 0; t < 8; t++) {
            gf2to32_matrix_random(A, r, k);
            gf2to32_matrix_random(D, r, k);
            gf2to32_matrix_random(B, k, c);
            gf2to32_matrix_random(C2, r, c);
            const gf2to32 sc = rand_elem();

            gf2to32_matrix_add(D1, A, D, r, k);
            impl->matrix_add(D2, A, D, r, k);
            if (vector_cmp(D1, D2, r*k)) { err = "matrix_add"; break; }

            gf2to32_matrix_add_scalar(D1, sc, A, r, k);
            impl->matrix_add_scalar(D2, sc, A, r, k);
            if (vector_cmp(D1, D2, r*k)) { err = "matrix_add_scalar"; break; }

            gf2to32_matrix_mul(C1, A, B, r, k, c);
            impl->matrix_mul(C2, A, B, r, k, c);
            if (vector_cmp(C1, C2, r*c)) { err = "matrix_mul"; break; }

            gf2to32_matrix_add_mul(C1, D, B, r, k, c);
            impl->matrix_add_mul(C2, D, B, r, k, c);
            if (vector_cmp(C1, C2, r*c)) { err = "matrix_add_mul"; break; }
        }

        free(A); free(D); free(D1); free(D2); free(B); free(C1); free(C2);
        if (err) {
            printf("error: test_matrix_simd(%s): %s, shape=%ux%ux%u\n", impl->name, err, r, k, c);
            return 1;
        }
    }

    return 0;
}

#ifdef USE_AVX2
uint32_t test_vector_add() {
    const uint32_t n = 123;
    gf2to32 *a  = gf2to32_vector_alloc(n);
    gf2to32 *c1 = gf2to32_vector_alloc(n);
    gf2to32 *c2 = gf2to32_vector_alloc(n);
    int ret = 0;

    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a, n);
        gf2to32_vector_random(c1, n);
        gf2to32_vector_copy(c2, c1, n);
        gf2to32_vector_add(c1, a, n);
        gf2to32_vector_add_u256(c2, a, n);
        if (vector_cmp(c1, c2, n)) {
            printf("error test_vector_add\n");
            ret = 1;
            break;
        }
    }

    free(a); free(c1); free(c2);
    return ret;
}

uint32_t test_vector_mul() {
    uint32_t tmp1[8] = {0};
    for (uint64_t a = 0; a < N; ++a) {
        for (uint64_t b = 0; b < N; ++b) {
            uint32_t i = rand();
            uint32_t j = rand();
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = i; }
            const __m256i aa = _mm256_loadu_si256((const __m256i *)tmp1);
            for (uint32_t k = 0; k < 8; ++k) { tmp1[k] = j; }
            const __m256i bb = _mm256_loadu_si256((const __m256i *)tmp1);

            const uint32_t c1 = gf2to32_mul(i, j);
            const __m256i c2 = gf2to32v_mul_u256(aa, bb);
            _mm256_storeu_si256((__m256i *)tmp1, c2);

            for (uint32_t k = 0; k < 8; ++k) {
                if (c1 != tmp1[k]){
                    printf("error: test vector mul: %d %d\n", c1, tmp1[k]);
                    return 1;
                }
            }

            const __m256i c3 = gf2to32v_mul_u256_v2(aa, bb);
            _mm256_storeu_si256((__m256i *)tmp1, c3);

            for (uint32_t k = 0; k < 8; ++k) {
                if (c1 != tmp1[k]){
                    printf("error: test vector mul (2): %d %d\n", c1, tmp1[k]);
                    return 1;
                }
            }
        }
    }
    return 0;
}

/// element-wise check of gf2to32v_{add,sub,mul}_u256 with distinct lanes
int test_arith_u256() {
    gf2to32 a[8], b[8], c[8];
    for (uint64_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 8; i++) { a[i] = rand_elem(); b[i] = rand_elem(); }
        const __m256i va = _mm256_loadu_si256((const __m256i *)a);
        const __m256i vb = _mm256_loadu_si256((const __m256i *)b);

        _mm256_storeu_si256((__m256i *)c, gf2to32v_add_u256(va, vb));
        for (uint32_t i = 0; i < 8; i++) { if (c[i] != gf2to32_add(a[i], b[i])) { printf("error: test_arith_u256: add\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to32v_sub_u256(va, vb));
        for (uint32_t i = 0; i < 8; i++) { if (c[i] != gf2to32_sub(a[i], b[i])) { printf("error: test_arith_u256: sub\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to32v_mul_u256(va, vb));
        for (uint32_t i = 0; i < 8; i++) { if (c[i] != gf2to32_mul(a[i], b[i])) { printf("error: test_arith_u256: mul\n"); return 1; } }
    }
    return 0;
}

int test_vector_add_gf2_u256() {
    const uint32_t n = 123;

    gf2to32 *a1 = gf2to32_vector_alloc(n);
    gf2to32 *a2 = gf2to32_vector_alloc(n);
    gf2 *b = gf2_vector_alloc(n);

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a1, n);
        gf2_vector_random(b, n);
        gf2to32_vector_copy(a2, a1, n);
        gf2to32_vector_add_gf2(a1, b, n);
        gf2to32_vector_add_gf2_u256(a2, b, n);

        for (uint32_t k = 0; k < n; k++) {
            if (a1[k] != a2[k]) {
                printf("error test_vector_add_gf2_u256\n");
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
        free(a1); free(a2); free(b);
    return ret;
}

int test_vector_scalar_add_gf2_u256_v2() {
    const uint32_t n = 123;

    gf2to32 *a1 = gf2to32_vector_alloc(n);
    gf2to32 *a2 = gf2to32_vector_alloc(n);
    gf2 *b = gf2_vector_alloc(n);

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a1, n);
        gf2_vector_random(b, n);
        gf2to32_vector_copy(a2, a1, n);
        const gf2to32 s = rand_elem();
        gf2to32_vector_scalar_add_gf2_v2(a1, s, b, n);
        gf2to32_vector_scalar_add_gf2_u256_v2(a2, s, b, n);

        if (vector_cmp(a1, a2, n)) {
            printf("error test_vector_scalar_add_gf2_u256_v2\n");
            ret = 1;
            break;
        }
    }

    free(a1); free(a2); free(b);
    return ret;
}

int test_vector_scalar_add_u256() {
    const uint32_t n = 123;

    gf2to32 *a1 = gf2to32_vector_alloc(n);
    gf2to32 *c1 = gf2to32_vector_alloc(n);
    gf2to32 *c2 = gf2to32_vector_alloc(n);

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to32_vector_random(a1, n);
        gf2to32_vector_random(c1, n);
        gf2to32_vector_copy(c2, c1, n);
        gf2to32 a = rand();

        gf2to32_vector_scalar_add(c1, a, a1, n);
        gf2to32_vector_scalar_add_u256(c2, a, a1, n);

        for (uint32_t k = 0; k < n; k++) {
            if (c1[k] != c2[k]) {
                printf("error test_vector_scalar_add_u256\n");
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
        free(a1); free(c1); free(c2);
    return ret;
}

#endif

#ifdef USE_AVX512
/// element-wise check of gf2to32v_{add,sub,mul}_u512 with distinct lanes
int test_arith_u512() {
    gf2to32 a[16], b[16], c[16];
    for (uint64_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 16; i++) { a[i] = rand_elem(); b[i] = rand_elem(); }
        const __m512i va = _mm512_loadu_si512((const void *)a);
        const __m512i vb = _mm512_loadu_si512((const void *)b);

        _mm512_storeu_si512((void *)c, gf2to32v_add_u512(va, vb));
        for (uint32_t i = 0; i < 16; i++) { if (c[i] != gf2to32_add(a[i], b[i])) { printf("error: test_arith_u512: add\n"); return 1; } }
        _mm512_storeu_si512((void *)c, gf2to32v_sub_u512(va, vb));
        for (uint32_t i = 0; i < 16; i++) { if (c[i] != gf2to32_sub(a[i], b[i])) { printf("error: test_arith_u512: sub\n"); return 1; } }
        _mm512_storeu_si512((void *)c, gf2to32v_mul_u512(va, vb));
        for (uint32_t i = 0; i < 16; i++) { if (c[i] != gf2to32_mul(a[i], b[i])) { printf("error: test_arith_u512: mul\n"); return 1; } }
    }
    return 0;
}
#endif

#ifdef USE_NEON
/// element-wise check of gf2to32v_{add,sub,mul}_u128 with distinct lanes
int test_arith_u128() {
    gf2to32 a[4], b[4], c[4];
    for (uint64_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 4; i++) { a[i] = rand_elem(); b[i] = rand_elem(); }
        const uint32x4_t va = vld1q_u32(a);
        const uint32x4_t vb = vld1q_u32(b);

        vst1q_u32(c, gf2to32v_add_u128(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to32_add(a[i], b[i])) { printf("error: test_arith_u128: add\n"); return 1; } }
        vst1q_u32(c, gf2to32v_sub_u128(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to32_sub(a[i], b[i])) { printf("error: test_arith_u128: sub\n"); return 1; } }
        vst1q_u32(c, gf2to32v_mul_u128(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to32_mul(a[i], b[i])) { printf("error: test_arith_u128: mul\n"); return 1; } }
    }
    return 0;
}
#endif

int main() {
    srand(42);
    if (test_add()) { return 1; }
    if (test_mul_field()) { return 1; }
    if (test_arith_ref()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }

#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_arith_u256()) { return 1; }
    if (test_vector_add_gf2_u256()) { return 1; }
    if (test_vector_scalar_add_gf2_u256_v2()) { return 1; }
    if (test_vector_scalar_add_u256()) { return 1; }
#endif
#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
#endif
#ifdef USE_NEON
    if (test_arith_u128()) { return 1; }
#endif

    for (const simd_impl *impl = simd_impls; impl->name; impl++) {
        if (test_vector_simd(impl)) { return 1; }
        if (test_matrix_simd(impl)) { return 1; }
    }

    printf("all good\n");
    return 0;
}
