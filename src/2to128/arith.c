#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint64_t N = 1u<<10u;

static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
static const uint32_t nr_test_lengths = sizeof(test_lengths)/sizeof(test_lengths[0]);
/// {n_rows1, n_cols1, n_cols2}
static const uint32_t test_shapes[][3] = {{1,1,1},{3,5,2},{17,9,4},{33,40,7},{8,8,8},{16,3,17}};
static const uint32_t nr_test_shapes = sizeof(test_shapes)/sizeof(test_shapes[0]);

static gf2to128 rand_elem(void) {
    return ((gf2to128)rand() <<  0u) ^ ((gf2to128)rand() << 31u) ^
           ((gf2to128)rand() << 62u) ^ ((gf2to128)rand() << 93u) ^
           ((gf2to128)rand() << 124u);
}

static int vector_cmp(const gf2to128 *a,
                      const gf2to128 *b,
                      const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (a[i] != b[i]) { return 1; }
    }
    return 0;
}

/// a^(2^128) = a holds only if MODULUS is irreducible
int test_mul_field() {
    for (uint64_t t = 0; t < 64; ++t) {
        const gf2to128 a = rand_elem();
        gf2to128 b = a;
        for (uint32_t k = 0; k < 128; ++k) {
            b = gf2to128_mul(b, b);
        }

        if (a != b) {
            printf("error: test mul field\n");
            return 1;
        }
    }

    return 0;
}

/// tests add, sub, neg, inv
int test_arith_ref() {
    if (gf2to128_inv(0) != 0) {
        printf("error: test_arith_ref: inv(0)\n");
        return 1;
    }

    for (uint64_t t = 0; t < N; ++t) {
        const gf2to128 a = rand_elem();
        const gf2to128 b = rand_elem();
        if (gf2to128_add(a, gf2to128_neg(a)) != 0) {
            printf("error: test_arith_ref: neg\n");
            return 1;
        }
        if (gf2to128_sub(gf2to128_add(a, b), b) != a) {
            printf("error: test_arith_ref: sub\n");
            return 1;
        }
        if (gf2to128_mul(a, b) != gf2to128_mul_v2(a, b)) {
            printf("error: test_arith_ref: mul_v2\n");
            return 1;
        }
        if (gf2to128_mul_gf2(a, 1) != a || gf2to128_mul_gf2(a, 0) != 0) {
            printf("error: test_arith_ref: mul_gf2\n");
            return 1;
        }
        // inversion is slow, only test a subset
        if ((t < 128) && a && (gf2to128_mul(a, gf2to128_inv(a)) != 1)) {
            printf("error: test_arith_ref: inv\n");
            return 1;
        }
    }

    return 0;
}

/// tests vector_mul_acc, vector_scalar_add_v2, vector_get/set
int test_vector_ref() {
    const uint32_t n = 77;
    gf2to128 *a = gf2to128_vector_alloc(n);
    gf2to128 *b = gf2to128_vector_alloc(n);
    gf2to128 *c1 = gf2to128_vector_alloc(n);
    gf2to128 *c2 = gf2to128_vector_alloc(n);
    int ret = 0;

    for (uint32_t t = 0; t < 64; t++) {
        gf2to128_vector_random(a, n);
        gf2to128_vector_random(b, n);
        const gf2to128 s = rand_elem();

        gf2to128 acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            acc ^= gf2to128_mul(gf2to128_vector_get(a, i), gf2to128_vector_get(b, i));
        }
        if (acc != gf2to128_vector_mul_acc(a, b, n)) {
            printf("error: test_vector_ref: mul_acc\n");
            ret = 1; goto finish;
        }

        gf2to128_vector_copy(c1, a, n);
        gf2to128_vector_scalar_add(c1, s, b, n);
        gf2to128_vector_scalar_add_v2(c2, a, s, b, n);
        if (vector_cmp(c1, c2, n)) {
            printf("error: test_vector_ref: scalar_add_v2\n");
            ret = 1; goto finish;
        }

        gf2to128_vector_set(c1, t % n, s);
        if (gf2to128_vector_get(c1, t % n) != s) {
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
        gf2to128 *A  = gf2to128_matrix_alloc(r, k);
        gf2to128 *AT = gf2to128_matrix_alloc(k, r);
        gf2to128 *A2 = gf2to128_matrix_alloc(r, k);
        gf2to128 *Ik = gf2to128_matrix_alloc(k, k);
        gf2to128 *Ir = gf2to128_matrix_alloc(r, r);
        gf2to128 *B  = gf2to128_matrix_alloc(k, c);
        gf2to128 *C1 = gf2to128_matrix_alloc(r, c);
        gf2to128 *C2 = gf2to128_matrix_alloc(r, c);
        gf2to128 *T  = gf2to128_matrix_alloc(r, c);

        gf2to128_matrix_random(A, r, k);
        gf2to128_matrix_random(B, k, c);
        gf2to128_matrix_random(C1, r, c);
        gf2to128_matrix_id(Ik, k, k);
        gf2to128_matrix_id(Ir, r, r);

        gf2to128_matrix_mul(A2, A, Ik, r, k, k);
        if (vector_cmp(A, A2, r*k)) {
            printf("error: test_matrix_ref: A*I\n");
            ret = 1; goto finish;
        }
        gf2to128_matrix_mul(A2, Ir, A, r, r, k);
        if (vector_cmp(A, A2, r*k)) {
            printf("error: test_matrix_ref: I*A\n");
            ret = 1; goto finish;
        }

        gf2to128_matrix_transpose(AT, A, r, k);
        gf2to128_matrix_transpose(A2, AT, k, r);
        if (vector_cmp(A, A2, r*k) ||
            gf2to128_matrix_get(AT, k, k-1, 0) != gf2to128_matrix_get(A, r, 0, k-1)) {
            printf("error: test_matrix_ref: transpose\n");
            ret = 1; goto finish;
        }

        gf2to128_matrix_copy(C2, C1, r, c);
        gf2to128_matrix_add_mul(C1, A, B, r, k, c);
        gf2to128_matrix_mul(T, A, B, r, k, c);
        gf2to128_matrix_add(C2, C2, T, r, c);
        if (vector_cmp(C1, C2, r*c)) {
            printf("error: test_matrix_ref: add_mul\n");
            ret = 1; goto finish;
        }

        const gf2to128 v = rand_elem();
        gf2to128_matrix_set(C1, r, r-1, c-1, v);
        if (gf2to128_matrix_get(C1, r, r-1, c-1) != v || C1[r*c - 1] != v) {
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
    void (*vector_add)(gf2to128 *, const gf2to128 *, uint32_t);
    void (*vector_add_v2)(gf2to128 *, const gf2to128 *, const gf2to128 *, uint32_t);
    void (*vector_scalar_add)(gf2to128 *, gf2to128, const gf2to128 *, uint32_t);
    void (*vector_scalar_add_v2)(gf2to128 *, const gf2to128 *, gf2to128, const gf2to128 *, uint32_t);
    gf2to128 (*vector_mul_acc)(const gf2to128 *, const gf2to128 *, uint32_t);
    void (*matrix_add)(gf2to128 *, const gf2to128 *, const gf2to128 *, uint32_t, uint32_t);
    void (*matrix_add_scalar)(gf2to128 *, gf2to128, const gf2to128 *, uint32_t, uint32_t);
    void (*matrix_mul)(gf2to128 *, const gf2to128 *, const gf2to128 *, uint32_t, uint32_t, uint32_t);
    void (*matrix_add_mul)(gf2to128 *, const gf2to128 *, const gf2to128 *, uint32_t, uint32_t, uint32_t);
} simd_impl;

static const simd_impl simd_impls[] = {
#ifdef USE_AVX2
    {"u256", gf2to128_vector_add_u256, gf2to128_vector_add_u256_v2,
     gf2to128_vector_scalar_add_u256, gf2to128_vector_scalar_add_u256_v2, gf2to128_vector_mul_acc_u256,
     gf2to128_matrix_add_u256, gf2to128_matrix_add_scalar_u256, gf2to128_matrix_mul_u256, gf2to128_matrix_add_mul_u256},
#endif
#ifdef USE_AVX512
    {"u512", gf2to128_vector_add_u512, gf2to128_vector_add_u512_v2,
     gf2to128_vector_scalar_add_u512, gf2to128_vector_scalar_add_u512_v2, gf2to128_vector_mul_acc_u512,
     gf2to128_matrix_add_u512, gf2to128_matrix_add_scalar_u512, gf2to128_matrix_mul_u512, gf2to128_matrix_add_mul_u512},
#endif
#ifdef USE_NEON
    {"u128", gf2to128_vector_add_u128, gf2to128_vector_add_u128_v2,
     gf2to128_vector_scalar_add_u128, gf2to128_vector_scalar_add_u128_v2, gf2to128_vector_mul_acc_u128,
     gf2to128_matrix_add_u128, gf2to128_matrix_add_scalar_u128, gf2to128_matrix_mul_u128, gf2to128_matrix_add_mul_u128},
#endif
    {NULL}
};

/// compares the vectorized vector functions against the reference ones
int test_vector_simd(const simd_impl *impl) {
    for (uint32_t l = 0; l < nr_test_lengths; l++) {
        const uint32_t n = test_lengths[l];
        gf2to128 *a  = gf2to128_vector_alloc(n);
        gf2to128 *b  = gf2to128_vector_alloc(n);
        gf2to128 *c1 = gf2to128_vector_alloc(n);
        gf2to128 *c2 = gf2to128_vector_alloc(n);
        const char *err = NULL;

        for (uint32_t t = 0; t < 16; t++) {
            gf2to128_vector_random(a, n);
            gf2to128_vector_random(b, n);
            gf2to128_vector_random(c1, n);
            gf2to128_vector_copy(c2, c1, n);
            const gf2to128 s = rand_elem();

            gf2to128_vector_add(c1, a, n);
            impl->vector_add(c2, a, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add"; break; }

            gf2to128_vector_copy(c1, a, n);
            gf2to128_vector_add(c1, b, n);
            impl->vector_add_v2(c2, a, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add_v2"; break; }

            gf2to128_vector_scalar_add(c1, s, a, n);
            impl->vector_scalar_add(c2, s, a, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add"; break; }

            gf2to128_vector_scalar_add_v2(c1, a, s, b, n);
            impl->vector_scalar_add_v2(c2, a, s, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add_v2"; break; }

            if (gf2to128_vector_mul_acc(a, b, n) != impl->vector_mul_acc(a, b, n)) {
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
        gf2to128 *A  = gf2to128_matrix_alloc(r, k);
        gf2to128 *D  = gf2to128_matrix_alloc(r, k);
        gf2to128 *D1 = gf2to128_matrix_alloc(r, k);
        gf2to128 *D2 = gf2to128_matrix_alloc(r, k);
        gf2to128 *B  = gf2to128_matrix_alloc(k, c);
        gf2to128 *C1 = gf2to128_matrix_alloc(r, c);
        gf2to128 *C2 = gf2to128_matrix_alloc(r, c);
        const char *err = NULL;

        for (uint32_t t = 0; t < 4; t++) {
            gf2to128_matrix_random(A, r, k);
            gf2to128_matrix_random(D, r, k);
            gf2to128_matrix_random(B, k, c);
            gf2to128_matrix_random(C2, r, c);
            const gf2to128 sc = rand_elem();

            gf2to128_matrix_add(D1, A, D, r, k);
            impl->matrix_add(D2, A, D, r, k);
            if (vector_cmp(D1, D2, r*k)) { err = "matrix_add"; break; }

            gf2to128_matrix_add_scalar(D1, sc, A, r, k);
            impl->matrix_add_scalar(D2, sc, A, r, k);
            if (vector_cmp(D1, D2, r*k)) { err = "matrix_add_scalar"; break; }

            gf2to128_matrix_mul(C1, A, B, r, k, c);
            impl->matrix_mul(C2, A, B, r, k, c);
            if (vector_cmp(C1, C2, r*c)) { err = "matrix_mul"; break; }

            gf2to128_matrix_add_mul(C1, D, B, r, k, c);
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
int test_expand() {
    uint8_t in = 0x21;
    gf2to128 tmp[8];
    __m256i out[4];
    gf2to128v_expand_gf2_x8_u256(out, in);
    for (uint32_t i = 0; i < 4; ++i) {
        _mm256_storeu_si256((__m256i *)(tmp + 2*i), out[i]);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        const gf2to128 t1 = tmp[i];
        const gf2to128 t2 = (in >> i) & 1u;

        if (t1 != t2) {
            printf("error test_expand: %d\n", i);
            return 1;
        }
    }

    return 0;
}

int test_vector_mul() {
    for (uint64_t i = 0; i < N; i++) {
        const gf2to128 a = rand() ^ ((gf2to128)rand() << 64u);
        const gf2to128 b = rand() ^ ((gf2to128)rand() << 64u);

        const gf2to128 c1 = gf2to128_mul(a, b);
        const gf2to128 c2 = gf2to128_mul_u128(a, b);
        if (c1 != c2) {
            printf("test_vector_mul: %llx %llx\n",
               (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }

        const gf2to128 c3 = gf2to128_mul_v2(a, b);
        if (c3 != c1) {
            printf("test_vector_mul(v2): %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c3);
            return 1;
        }
    }
    return 0;
}

int test_vector_mul_u256() {
    __uint128_t vals1[2], vals2[2];
    for (uint64_t i = 0; i < N; i++) {
        const gf2to128 a = rand() ^ ((gf2to128)rand() << 64u);
        const gf2to128 b = rand() ^ ((gf2to128)rand() << 64u);
        vals1[0] = a; vals1[1] = a;
        vals2[0] = b; vals2[1] = b;
        const gf2to128 c1 = gf2to128_mul(a, b);

        const __m256i a_ = _mm256_loadu_si256((__m256i *)vals1);
        const __m256i b_ = _mm256_loadu_si256((__m256i *)vals2);
        const __m256i c_ = gf2to128v_mul_u256(a_, b_);
        _mm256_storeu_si256((__m256i *)vals1, c_);
        gf2to128 c2 = vals1[0];
        gf2to128 c3 = vals1[1];

        if (c1 != c2) {
            printf("test_vector_mul_u256: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
        if (c1 != c3) {
            printf("test_vector_mul_u256_2: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c3);
            return 1;
        }
    }
    return 0;
}

/// element-wise check of gf2to128v_{add,sub,mul}_u256 with distinct lanes
int test_arith_u256() {
    gf2to128 a[2], b[2], c[2];
    for (uint64_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 2; i++) { a[i] = rand_elem(); b[i] = rand_elem(); }
        const __m256i va = _mm256_loadu_si256((const __m256i *)a);
        const __m256i vb = _mm256_loadu_si256((const __m256i *)b);

        _mm256_storeu_si256((__m256i *)c, gf2to128v_add_u256(va, vb));
        for (uint32_t i = 0; i < 2; i++) { if (c[i] != gf2to128_add(a[i], b[i])) { printf("error: test_arith_u256: add\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to128v_sub_u256(va, vb));
        for (uint32_t i = 0; i < 2; i++) { if (c[i] != gf2to128_sub(a[i], b[i])) { printf("error: test_arith_u256: sub\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to128v_mul_u256(va, vb));
        for (uint32_t i = 0; i < 2; i++) { if (c[i] != gf2to128_mul(a[i], b[i])) { printf("error: test_arith_u256: mul\n"); return 1; } }
    }
    return 0;
}

int test_vector_add_gf2_u256() {
    const uint32_t n = 3;

    gf2to128 *a1 = gf2to128_vector_alloc(n);
    gf2to128 *a2 = gf2to128_vector_alloc(n);
    gf2 *b = gf2_vector_alloc(n);

    // gf2_vector_zero(b, n);
    int ret = 0;
    for (uint32_t t = 0; t < 100; t++) {
        gf2to128_vector_random(a1, n);
        gf2_vector_random(b, n);
        gf2to128_vector_copy(a2, a1, n);
        gf2to128_vector_add_gf2(a1, b, n);
        gf2to128_vector_add_gf2_u256(a2, b, n);

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

int test_vector_scalar_add_u256_v2() {
    const uint32_t n = 2;

    gf2to128 *in1 = gf2to128_vector_alloc(n);
    gf2to128 *in2 = gf2to128_vector_alloc(n);
    gf2to128 *out1 = gf2to128_vector_alloc(n);
    gf2to128 *out2 = gf2to128_vector_alloc(n);
    gf2to128 a = 2;

    int ret = 0;
    for (uint32_t t = 0; t < N; t++) {
        gf2to128_vector_random(in1, n);
        gf2to128_vector_random(in2, n);
        gf2to128_vector_scalar_add_v2(out1, in1, a, in2, n);
        gf2to128_vector_scalar_add_u256_v2(out2, in1, a, in2, n);

        for (uint32_t k = 0; k < n; k++) {
            if (out1[k] != out2[k]) {
                printf("error test_vector_scalar_add_u256_v2\n");
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(in1); free(in2); free(out1); free(out2);
    return ret;
}


int test_matrix_add_gf2_u256_v2() {
    const uint32_t n = 9, m = 1;
    gf2 *A = gf2_matrix_alloc(n, m);
    gf2to128 *B = gf2to128_matrix_alloc(n, m);
    gf2to128 *C1 = gf2to128_matrix_alloc(n, m);
    gf2to128 *C2 = gf2to128_matrix_alloc(n, m);
    int ret = 0;

    for (uint32_t t = 0; t < N; t++) {
        gf2_matrix_random(A, n, m);
        gf2to128_matrix_random(B, n, m);
        gf2to128_matrix_add_gf2(C1, A, B, n, m);
        gf2to128_matrix_add_gf2_u256(C2, A, B, n, m);

        for (uint32_t k = 0; k < n; k++) {
            if (C1[k] != C2[k]) {
                printf("error test_matrix_add_gf2_u256_v2()\n");
                gf2to128_matrix_print(C1, n, m);
                gf2to128_matrix_print(C2, n, m);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(A); free(B); free(C1); free(C2);
    return ret;
}

/// checks the GF(2) subfield functions against the reference functions
int test_gf2_u256() {
    for (uint32_t l = 0; l < nr_test_lengths; l++) {
        const uint32_t n = test_lengths[l];
        gf2to128 *a  = gf2to128_vector_alloc(n);
        gf2to128 *c1 = gf2to128_vector_alloc(n);
        gf2to128 *c2 = gf2to128_vector_alloc(n);
        gf2 *b = gf2_vector_alloc(n);
        const char *err = NULL;

        for (uint32_t t = 0; t < 16; t++) {
            gf2to128_vector_random(a, n);
            gf2to128_vector_random(c1, n);
            gf2to128_vector_copy(c2, c1, n);
            gf2_vector_random(b, n);
            const gf2to128 s = rand_elem();
            const gf2 s2 = rand() & 1u;

            gf2to128_vector_add_gf2(c1, b, n);
            gf2to128_vector_add_gf2_u256(c2, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add_gf2_u256"; break; }

            gf2to128_vector_copy(c1, a, n);
            gf2to128_vector_add_gf2(c1, b, n);
            gf2to128_vector_add_gf2_u256_v2(c2, a, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_add_gf2_u256_v2"; break; }

            gf2to128_vector_scalar_add_gf2_v3(c1, s, b, n);
            gf2to128_vector_scalar_add_gf2_u256_v2(c2, s, b, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add_gf2_u256_v2"; break; }

            gf2to128_vector_scalar_add_gf2(c1, s2, a, n);
            gf2to128_vector_scalar_add_gf2_u256(c2, s2, a, n);
            if (vector_cmp(c1, c2, n)) { err = "vector_scalar_add_gf2_u256"; break; }
        }

        free(a); free(c1); free(c2); free(b);
        if (err) {
            printf("error: test_gf2_u256: %s, n=%u\n", err, n);
            return 1;
        }
    }

    return 0;
}
#endif

#ifdef USE_AVX512
/// element-wise check of gf2to128v_{add,sub,mul}_u512 with distinct lanes
int test_arith_u512() {
    gf2to128 a[4], b[4], c[4];
    for (uint64_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 4; i++) { a[i] = rand_elem(); b[i] = rand_elem(); }
        const __m512i va = _mm512_loadu_si512((const void *)a);
        const __m512i vb = _mm512_loadu_si512((const void *)b);

        _mm512_storeu_si512((void *)c, gf2to128v_add_u512(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to128_add(a[i], b[i])) { printf("error: test_arith_u512: add\n"); return 1; } }
        _mm512_storeu_si512((void *)c, gf2to128v_sub_u512(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to128_sub(a[i], b[i])) { printf("error: test_arith_u512: sub\n"); return 1; } }
        _mm512_storeu_si512((void *)c, gf2to128v_mul_u512(va, vb));
        for (uint32_t i = 0; i < 4; i++) { if (c[i] != gf2to128_mul(a[i], b[i])) { printf("error: test_arith_u512: mul\n"); return 1; } }
    }
    return 0;
}
#endif

#ifdef USE_NEON
/// check of gf2to128v_{add,sub,mul}_u128
int test_arith_u128() {
    for (uint64_t t = 0; t < N; ++t) {
        const gf2to128 a = rand_elem(), b = rand_elem();
        const uint64x2_t va = gf2to128_to_u64x2(a);
        const uint64x2_t vb = gf2to128_to_u64x2(b);

        if (gf2to128_from_u64x2(gf2to128v_add_u128(va, vb)) != gf2to128_add(a, b)) { printf("error: test_arith_u128: add\n"); return 1; }
        if (gf2to128_from_u64x2(gf2to128v_sub_u128(va, vb)) != gf2to128_sub(a, b)) { printf("error: test_arith_u128: sub\n"); return 1; }
        if (gf2to128_from_u64x2(gf2to128v_mul_u128(va, vb)) != gf2to128_mul(a, b)) { printf("error: test_arith_u128: mul\n"); return 1; }
    }
    return 0;
}
#endif

int main() {
    srand(42);
    if (test_mul_field()) { return 1; }
    if (test_arith_ref()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }

#ifdef USE_AVX2
    if (test_expand()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_vector_mul_u256()) { return 1; }
    if (test_arith_u256()) { return 1; }
    if (test_vector_add_gf2_u256()) { return 1; }
    if (test_vector_scalar_add_u256_v2()) { return 1; }
    if (test_matrix_add_gf2_u256_v2()) { return 1; }
    if (test_gf2_u256()) { return 1; }
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

    // gf2to128 c2 = mul(a, 2);
    // printf("%llx\n", c2);
	printf("all good\n");
	return 0;
}
