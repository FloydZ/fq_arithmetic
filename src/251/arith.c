#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))
static const uint32_t test_shapes[][3] = {{1,1,1},{3,5,2},{17,9,4},{33,40,7},{64,3,65},{2,70,1}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

int test_scalar() {
    for (uint32_t a = 0; a < PRIME; a++) {
        if (gf251_add(a, gf251_neg(a)) != 0) {
            printf("error gf251_neg\n");
            return 1;
        }
        if ((a != 0) && (gf251_mul(a, gf251_inv(a)) != 1)) {
            printf("error gf251_inv\n");
            return 1;
        }
        for (uint32_t b = 0; b < PRIME; b++) {
            if (gf251_add(a, b) != (a + b) % PRIME ||
                gf251_sub(a, b) != (a + PRIME - b) % PRIME ||
                gf251_mul(a, b) != (a * b) % PRIME ||
                gf251_sub(gf251_add(a, b), b) != a) {
                printf("error gf251_add/sub/mul\n");
                return 1;
            }
        }
    }
    if (gf251_inv(0) != 0) {
        printf("error gf251_inv(0)\n");
        return 1;
    }
    return 0;
}

/// checks all ref vector functions against naive implementations
int test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        gf251 *a = gf251_vector_alloc(n), *b = gf251_vector_alloc(n),
              *c = gf251_vector_alloc(n), *d = gf251_vector_alloc(n);
        gf251_vector_random(a, n);
        gf251_vector_random(b, n);
        gf251_vector_copy(c, a, n);
        for (uint32_t i = 0; i < n; i++) {
            if (gf251_vector_get(c, i) != a[i] || a[i] >= PRIME) {
                printf("error gf251_vector_copy/random\n");
                return 1;
            }
        }

        const gf251 s = rand() % PRIME;
        gf251_vector_add(c, b, n);
        gf251_vector_scalar_add_v2(d, a, s, b, n);
        uint32_t acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            acc = (acc + a[i]*b[i]) % PRIME;
            if (c[i] != (a[i] + b[i]) % PRIME || d[i] != (a[i] + s*b[i]) % PRIME) {
                printf("error gf251_vector_add/scalar_add_v2\n");
                return 1;
            }
        }
        if (acc != gf251_vector_mul_acc(a, b, n)) {
            printf("error gf251_vector_mul_acc\n");
            return 1;
        }

        gf251_vector_copy(c, a, n);
        gf251_vector_scalar_add(c, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            if (c[i] != d[i]) {
                printf("error gf251_vector_scalar_add\n");
                return 1;
            }
        }

        gf251_vector_set(c, 0, 7);
        gf251_vector_zero(d, n);
        if (gf251_vector_get(c, 0) != 7 || gf251_vector_mul_acc(d, a, n) != 0) {
            printf("error gf251_vector_set/zero\n");
            return 1;
        }
        free(a); free(b); free(c); free(d);
    }
    return 0;
}

/// checks a set of SIMD vector functions against the ref implementation
int test_vector_simd(const char *name,
                     void (*add)(gf251 *, const gf251 *, const uint32_t),
                     void (*add_v2)(gf251 *, const gf251 *, const gf251 *, const uint32_t),
                     void (*scalar_add)(gf251 *, const gf251, const gf251 *, const uint32_t),
                     void (*scalar_add_v2)(gf251 *, const gf251 *, const gf251, const gf251 *, const uint32_t),
                     gf251 (*mul_acc)(const gf251 *, const gf251 *, const uint32_t)) {
    for (uint32_t l = 0; l < NR_TEST_LENGTHS; l++) {
        const uint32_t n = test_lengths[l];
        gf251 *a = gf251_vector_alloc(n), *b = gf251_vector_alloc(n),
              *c1 = gf251_vector_alloc(n), *c2 = gf251_vector_alloc(n);
        gf251_vector_random(a, n);
        gf251_vector_random(b, n);
        const gf251 s = rand() % PRIME;

        gf251_vector_copy(c1, a, n); gf251_vector_add(c1, b, n);
        gf251_vector_copy(c2, a, n); add(c2, b, n);
        if (memcmp(c1, c2, n)) { printf("error %s: vector_add n=%u\n", name, n); return 1; }

        gf251_vector_random(c2, n);
        add_v2(c2, a, b, n);
        if (memcmp(c1, c2, n)) { printf("error %s: vector_add_v2 n=%u\n", name, n); return 1; }

        gf251_vector_copy(c1, a, n); gf251_vector_scalar_add(c1, s, b, n);
        gf251_vector_copy(c2, a, n); scalar_add(c2, s, b, n);
        if (memcmp(c1, c2, n)) { printf("error %s: vector_scalar_add n=%u\n", name, n); return 1; }

        gf251_vector_random(c2, n);
        scalar_add_v2(c2, a, s, b, n);
        if (memcmp(c1, c2, n)) { printf("error %s: vector_scalar_add_v2 n=%u\n", name, n); return 1; }

        if (gf251_vector_mul_acc(a, b, n) != mul_acc(a, b, n)) {
            printf("error %s: vector_mul_acc n=%u\n", name, n);
            return 1;
        }
        free(a); free(b); free(c1); free(c2);
    }
    return 0;
}

/// checks the ref matrix functions
int test_matrix_ref() {
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {
        const uint32_t n = test_shapes[t][0], k = test_shapes[t][1], m = test_shapes[t][2];
        gf251 *A  = gf251_matrix_alloc(n, k), *At = gf251_matrix_alloc(k, n),
              *A2 = gf251_matrix_alloc(n, k), *B  = gf251_matrix_alloc(k, m),
              *I  = gf251_matrix_alloc(k, k), *C1 = gf251_matrix_alloc(n, m),
              *C2 = gf251_matrix_alloc(n, m), *C3 = gf251_matrix_alloc(n, m);
        gf251_matrix_random(A, n, k);
        gf251_matrix_random(B, k, m);
        gf251_matrix_random(C3, n, m);

        // A*I == A
        gf251_matrix_id(I, k, k);
        gf251_matrix_mul(A2, A, I, n, k, k);
        if (memcmp(A, A2, n*k)) { printf("error gf251_matrix_id/mul\n"); return 1; }

        // (A^T)^T == A
        gf251_matrix_transpose(At, A, n, k);
        gf251_matrix_transpose(A2, At, k, n);
        if (memcmp(A, A2, n*k)) { printf("error gf251_matrix_transpose\n"); return 1; }
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < k; j++) {
                if (gf251_matrix_get(A, n, i, j) != gf251_matrix_get(At, k, j, i)) {
                    printf("error gf251_matrix_transpose 2\n"); return 1;
                }
            }
        }

        // add_mul == mul + add
        gf251_matrix_mul(C1, A, B, n, k, m);
        gf251_matrix_add(C2, C1, C3, n, m);
        gf251_matrix_copy(C1, C3, n, m);
        gf251_matrix_add_mul(C1, A, B, n, k, m);
        if (memcmp(C1, C2, n*m)) { printf("error gf251_matrix_add_mul\n"); return 1; }

        // naive product
        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < m; j++) {
                uint32_t s = 0;
                for (uint32_t l = 0; l < k; l++) {
                    s += A[l*n + i] * B[j*k + l];
                }
                if (gf251_add(s % PRIME, C3[j*n + i]) != C2[j*n + i]) {
                    printf("error gf251_matrix_mul naive\n"); return 1;
                }
            }
        }

        // add_scalar
        const gf251 s = rand() % PRIME;
        gf251_matrix_copy(C1, C3, n, m);
        gf251_matrix_add_scalar(C1, s, C2, n, m);
        for (uint32_t i = 0; i < n*m; i++) {
            if (C1[i] != (C3[i] + s*C2[i]) % PRIME) { printf("error gf251_matrix_add_scalar\n"); return 1; }
        }

        free(A); free(At); free(A2); free(B); free(I); free(C1); free(C2); free(C3);
    }
    return 0;
}

/// checks a set of SIMD matrix functions against the ref implementation
int test_matrix_simd(const char *name,
                     void (*add)(gf251 *, const gf251 *, const gf251 *, const uint32_t, const uint32_t),
                     void (*add_scalar)(gf251 *, const gf251, const gf251 *, const uint32_t, const uint32_t),
                     void (*mul)(gf251 *, const gf251 *, const gf251 *, const uint32_t, const uint32_t, const uint32_t),
                     void (*add_mul)(gf251 *, const gf251 *, const gf251 *, const uint32_t, const uint32_t, const uint32_t)) {
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {
        const uint32_t n = test_shapes[t][0], k = test_shapes[t][1], m = test_shapes[t][2];
        gf251 *A  = gf251_matrix_alloc(n, k), *B  = gf251_matrix_alloc(k, m),
              *C1 = gf251_matrix_alloc(n, m), *C2 = gf251_matrix_alloc(n, m),
              *C3 = gf251_matrix_alloc(n, m), *D  = gf251_matrix_alloc(n, m);
        gf251_matrix_random(A, n, k);
        gf251_matrix_random(B, k, m);
        gf251_matrix_random(C3, n, m);
        gf251_matrix_random(D, n, m);
        const gf251 s = rand() % PRIME;

        gf251_matrix_mul(C1, A, B, n, k, m);
        gf251_matrix_random(C2, n, m);
        mul(C2, A, B, n, k, m);
        if (memcmp(C1, C2, n*m)) { printf("error %s: matrix_mul\n", name); return 1; }

        gf251_matrix_copy(C1, C3, n, m); gf251_matrix_add_mul(C1, A, B, n, k, m);
        gf251_matrix_copy(C2, C3, n, m); add_mul(C2, A, B, n, k, m);
        if (memcmp(C1, C2, n*m)) { printf("error %s: matrix_add_mul\n", name); return 1; }

        gf251_matrix_add(C1, C2, C3, n, m);
        add(C2, C2, C3, n, m);
        if (memcmp(C1, C2, n*m)) { printf("error %s: matrix_add\n", name); return 1; }

        gf251_matrix_copy(C1, C3, n, m); gf251_matrix_add_scalar(C1, s, D, n, m);
        gf251_matrix_copy(C2, C3, n, m); add_scalar(C2, s, D, n, m);
        if (memcmp(C1, C2, n*m)) { printf("error %s: matrix_add_scalar\n", name); return 1; }

        free(A); free(B); free(C1); free(C2); free(C3); free(D);
    }
    return 0;
}

#ifdef USE_AVX2
uint32_t test_vector_add() {
    uint8_t tmp[32];
    for (uint32_t i = 0; i < PRIME; i++) {
        for (uint32_t j = 0; j < PRIME; j++) {
            const gf251 c = gf251_add(i, j);
            const gf251 e = gf251_sub(i, j);
            const __m256i a = _mm256_set1_epi8(i);
            const __m256i b = _mm256_set1_epi8(j);
            const __m256i d = gf251v_add_u256(a, b);

            _mm256_storeu_si256((__m256i *)tmp, d);
            for (uint32_t k = 0; k < 32; k++){
                if (tmp[k] != c) {
                    printf("test_vector_add full\n");
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tmp, gf251v_sub_u256(a, b));
            for (uint32_t k = 0; k < 32; k++){
                if (tmp[k] != e) {
                    printf("test_vector_sub\n");
                    return 1;
                }
            }
        }

    }

    return 0;
}

uint32_t test_vector_mul() {
    uint8_t tmp[32];
    for (uint32_t i = 0; i < PRIME; i++) {
        for (uint32_t j = 0; j < PRIME; j++) {
            const gf251 c = gf251_mul(i, j);
            const __m256i a = _mm256_set1_epi8(i);
            const __m256i b = _mm256_set1_epi8(j);
            const __m256i d = gf251v_mul_u256(a, b);

            _mm256_storeu_si256((__m256i *)tmp, d);
            for (uint32_t k = 0; k < 32; k++){
                if (tmp[k] != c) {
                    printf("test_vector_mul\n");
                    return 1;
                }
            }
        }
    }
    return 0;
}

/// lanes get different values, to detect lane permutations
int test_arith_u256() {
    uint8_t ta[32], tb[32], tc[32];
    for (uint32_t a = 0; a < PRIME; a++) {
        for (uint32_t b = 0; b < PRIME; b++) {
            for (uint32_t k = 0; k < 32; k++) {
                ta[k] = (a + k) % PRIME;
                tb[k] = (b + 7*k) % PRIME;
            }
            const __m256i a256 = _mm256_loadu_si256((const __m256i *)ta);
            const __m256i b256 = _mm256_loadu_si256((const __m256i *)tb);

            _mm256_storeu_si256((__m256i *)tc, gf251v_add_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] != gf251_add(ta[k], tb[k])) { printf("error gf251v_add_u256\n"); return 1; }
            }
            _mm256_storeu_si256((__m256i *)tc, gf251v_sub_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] != gf251_sub(ta[k], tb[k])) { printf("error gf251v_sub_u256\n"); return 1; }
            }
            _mm256_storeu_si256((__m256i *)tc, gf251v_mul_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] != gf251_mul(ta[k], tb[k])) { printf("error gf251v_mul_u256\n"); return 1; }
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_AVX512
int test_arith_u512() {
    uint8_t ta[64], tb[64], tc[64];
    for (uint32_t a = 0; a < PRIME; a++) {
        for (uint32_t b = 0; b < PRIME; b++) {
            for (uint32_t k = 0; k < 64; k++) {
                ta[k] = (a + k) % PRIME;
                tb[k] = (b + 7*k) % PRIME;
            }
            const __m512i a512 = _mm512_loadu_si512((const __m512i *)ta);
            const __m512i b512 = _mm512_loadu_si512((const __m512i *)tb);

            _mm512_storeu_si512((__m512i *)tc, gf251v_add_u512(a512, b512));
            for (uint32_t k = 0; k < 64; k++) {
                if (tc[k] != gf251_add(ta[k], tb[k])) { printf("error gf251v_add_u512\n"); return 1; }
            }
            _mm512_storeu_si512((__m512i *)tc, gf251v_sub_u512(a512, b512));
            for (uint32_t k = 0; k < 64; k++) {
                if (tc[k] != gf251_sub(ta[k], tb[k])) { printf("error gf251v_sub_u512\n"); return 1; }
            }
            _mm512_storeu_si512((__m512i *)tc, gf251v_mul_u512(a512, b512));
            for (uint32_t k = 0; k < 64; k++) {
                if (tc[k] != gf251_mul(ta[k], tb[k])) { printf("error gf251v_mul_u512\n"); return 1; }
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_NEON
int test_arith_u128() {
    uint8_t ta[16], tb[16], tc[16];
    for (uint32_t a = 0; a < PRIME; a++) {
        for (uint32_t b = 0; b < PRIME; b++) {
            for (uint32_t k = 0; k < 16; k++) {
                ta[k] = (a + k) % PRIME;
                tb[k] = (b + 7*k) % PRIME;
            }
            const uint8x16_t a128 = vld1q_u8(ta);
            const uint8x16_t b128 = vld1q_u8(tb);

            vst1q_u8(tc, gf251v_add_u128(a128, b128));
            for (uint32_t k = 0; k < 16; k++) {
                if (tc[k] != gf251_add(ta[k], tb[k])) { printf("error gf251v_add_u128\n"); return 1; }
            }
            vst1q_u8(tc, gf251v_sub_u128(a128, b128));
            for (uint32_t k = 0; k < 16; k++) {
                if (tc[k] != gf251_sub(ta[k], tb[k])) { printf("error gf251v_sub_u128\n"); return 1; }
            }
            vst1q_u8(tc, gf251v_mul_u128(a128, b128));
            for (uint32_t k = 0; k < 16; k++) {
                if (tc[k] != gf251_mul(ta[k], tb[k])) { printf("error gf251v_mul_u128\n"); return 1; }
            }
        }
    }
    return 0;
}
#endif

int main() {
    srand(1);
    if (test_scalar()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_arith_u256()) { return 1; }
    if (test_vector_simd("u256", gf251_vector_add_u256, gf251_vector_add_u256_v2,
                         gf251_vector_scalar_add_u256, gf251_vector_scalar_add_u256_v2,
                         gf251_vector_mul_acc_u256)) { return 1; }
    if (test_matrix_simd("u256", gf251_matrix_add_u256, gf251_matrix_add_scalar_u256,
                         gf251_matrix_mul_u256, gf251_matrix_add_mul_u256)) { return 1; }
#endif
#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
    if (test_vector_simd("u512", gf251_vector_add_u512, gf251_vector_add_u512_v2,
                         gf251_vector_scalar_add_u512, gf251_vector_scalar_add_u512_v2,
                         gf251_vector_mul_acc_u512)) { return 1; }
    if (test_matrix_simd("u512", gf251_matrix_add_u512, gf251_matrix_add_scalar_u512,
                         gf251_matrix_mul_u512, gf251_matrix_add_mul_u512)) { return 1; }
#endif
#ifdef USE_NEON
    if (test_arith_u128()) { return 1; }
    if (test_vector_simd("u128", gf251_vector_add_u128, gf251_vector_add_u128_v2,
                         gf251_vector_scalar_add_u128, gf251_vector_scalar_add_u128_v2,
                         gf251_vector_mul_acc_u128)) { return 1; }
    if (test_matrix_simd("u128", gf251_matrix_add_u128, gf251_matrix_add_scalar_u128,
                         gf251_matrix_mul_u128, gf251_matrix_add_mul_u128)) { return 1; }
#endif
    printf("all good\n");
    return 0;
}
