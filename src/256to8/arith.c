#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../helper.h"
#include "arith.h"

#define N (1u << 12)

static gf256to8 gf256to8_random() {
    return ((uint64_t)rand() << 42u) ^ ((uint64_t)rand() << 21u) ^ (uint64_t)rand();
}

/// a^(256^8) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to8 a = gf256to8_random();
        gf256to8 b = a;
        for (uint32_t k = 0; k < 64; ++k) {
            b = gf256to8_mul(b, b);
        }

        if (a != b) {
            printf("error test_mul_field\n");
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to8 a = gf256to8_random(), b = gf256to8_random(), c = gf256to8_random();
        if (gf256to8_mul(a, 1) != a) {
            printf("error test_mul: a*1\n");
            return 1;
        }
        if (gf256to8_mul(a, b) != gf256to8_mul(b, a)) {
            printf("error test_mul: a*b\n");
            return 1;
        }
        if (gf256to8_mul(gf256to8_mul(a, b), c) != gf256to8_mul(a, gf256to8_mul(b, c))) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }
        if (gf256to8_mul(a, b ^ c) != (gf256to8_mul(a, b) ^ gf256to8_mul(a, c))) {
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
        for (uint32_t i = 0; i < 4; ++i) {
            v1.v64[i] = gf256to8_random();
            v2.v64[i] = gf256to8_random();
        }

        v3.v256 = gf256to8v_mul_u256(v1.v256, v2.v256);
        for (uint32_t i = 0; i < 4; ++i) {
            if (v3.v64[i] != gf256to8_mul(v1.v64[i], v2.v64[i])) {
                printf("error test_vector_mul\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

int main () {
    if (test_mul_field()) { return 1; }
    if (test_mul()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
