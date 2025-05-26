#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint64_t N = 1u<<10u;

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

        const gf2to128 c1 = gf2to128_mul_u128(a, b);
        const gf2to128 c2 = gf2to128_mul(a, b);

        if (c1 != c2) {
            printf("test_vector_mul: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
    }
    return 0;
}

int test_vector_mul_u256() {
    __uint128_t vals1[2], vals2[2];
    for (uint64_t i = 0; i < N; i++) {
         const gf2to128 a = i ^ ((gf2to128)i << 64u);
         const gf2to128 b = i ^ ((gf2to128)i << 64u);
        // const gf2to128 a = rand();// ^ ((gf2to128)rand() << 64u);
        // const gf2to128 b = rand();// ^ ((gf2to128)rand() << 64u);
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


#endif

int main() {
#ifdef USE_AVX2
    if (test_expand()) { return 1; }
    if (test_vector_mul()) { return 1; }
    if (test_vector_mul_u256()) { return 1; }
    if (test_vector_add_gf2_u256()) { return 1; }
    if (test_vector_scalar_add_u256_v2()) { return 1; }

    if (test_matrix_add_gf2_u256_v2()) { return 1; }
#endif

    // gf2to128 c2 = mul(a, 2);
    // printf("%llx\n", c2);
	printf("all good\n");
	return 0;
}
