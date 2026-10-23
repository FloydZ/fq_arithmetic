#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../helper.h"
#include "arith.h"

#define N (1u << 12)

/// a^(256^2) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to2 a = rand();
        gf256to2 b = a;
        for (uint32_t k = 0; k < 16; ++k) {
            b = gf256to2_mul(b, b);
        }

        if (a != b) {
            printf("error test_mul_field: %u %u\n", a, b);
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to2 a = rand(), b = rand(), c = rand();
        if (gf256to2_mul(a, 1) != a) {
            printf("error test_mul: a*1\n");
            return 1;
        }
        if (gf256to2_mul(a, b) != gf256to2_mul(b, a)) {
            printf("error test_mul: a*b\n");
            return 1;
        }
        if (gf256to2_mul(gf256to2_mul(a, b), c) != gf256to2_mul(a, gf256to2_mul(b, c))) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }
        if (gf256to2_mul(a, b ^ c) != (gf256to2_mul(a, b) ^ gf256to2_mul(a, c))) {
            printf("error test_mul: a*(b+c)\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
int test_vector_mul() {
    v256 v1, v2, v3;
    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 16; ++i) {
            v1.v16[i] = rand();
            v2.v16[i] = rand();
        }

        v3.v256 = gf256to2v_mul_u256(v1.v256, v2.v256);
        for (uint32_t i = 0; i < 16; ++i) {
            if (v3.v16[i] != gf256to2_mul(v1.v16[i], v2.v16[i])) {
                printf("error test_vector_mul\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

int main() {
    if (test_mul_field()) { return 1; }
    if (test_mul()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
	return 0;
}
