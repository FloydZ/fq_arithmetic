#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arith.h"

#define N (1u << 10)

static void gf256to16_random(gf256to16 a) {
    for (uint32_t i = 0; i < 16; i++) {
        a[i] = rand();
    }
}

/// a^(256^16) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    gf256to16 a, b;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        memcpy(b, a, 16);
        for (uint32_t k = 0; k < 128; ++k) {
            gf256to16_mul(b, b, b);
        }

        if (memcmp(a, b, 16) != 0) {
            printf("error test_mul_field\n");
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    gf256to16 a, b, c, one = {0}, t1, t2, t3;
    one[0] = 1;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(b);
        gf256to16_random(c);

        gf256to16_mul(t1, a, one);
        if (memcmp(t1, a, 16) != 0) {
            printf("error test_mul: a*1\n");
            return 1;
        }

        gf256to16_mul(t1, a, b);
        gf256to16_mul(t2, b, a);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: a*b\n");
            return 1;
        }

        gf256to16_mul(t2, t1, c);
        gf256to16_mul(t3, b, c);
        gf256to16_mul(t1, a, t3);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }

        gf256to16_add(t3, b, c);
        gf256to16_mul(t1, a, t3);
        gf256to16_mul(t2, a, b);
        gf256to16_mul(t3, a, c);
        gf256to16_add(t2, t2, t3);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: a*(b+c)\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
int test_vector_mul() {
    uint8_t a[32], b[32], c[32];
    gf256to16 d;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(a + 16);
        gf256to16_random(b);
        gf256to16_random(b + 16);

        const __m256i a256 = _mm256_loadu_si256((const __m256i *)a);
        const __m256i b256 = _mm256_loadu_si256((const __m256i *)b);
        _mm256_storeu_si256((__m256i *)c, gf256to16v_mul_u256(a256, b256));

        for (uint32_t i = 0; i < 2; ++i) {
            gf256to16_mul(d, a + 16*i, b + 16*i);
            if (memcmp(d, c + 16*i, 16) != 0) {
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
