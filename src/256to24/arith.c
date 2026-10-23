#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"

static void gf256to24_random(gf256to24 a) {
    for (uint32_t i = 0; i < 24; i++) {
        a[i] = rand();
    }
}

/// X^3 = X + 1
static int test_mul_modulus() {
    gf256to24 x = {0}, t, r, e = {0};
    x[8] = 1;
    gf256to24_mul(t, x, x);
    gf256to24_mul(r, t, x);

    e[0] = 1;
    e[8] = 1;
    if (memcmp(r, e, 24) != 0) {
        printf("error gf256to24_mul: X^3 != X + 1\n");
        return 1;
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
static int test_mul() {
    gf256to24 a, b, c, one = {0}, t1, t2, t3;
    one[0] = 1;
    for (uint32_t i = 0; i < (1u << 12); i++) {
        gf256to24_random(a);
        gf256to24_random(b);
        gf256to24_random(c);

        gf256to24_mul(t1, a, one);
        if (memcmp(t1, a, 24) != 0) {
            printf("error gf256to24_mul: a*1 != a\n");
            return 1;
        }

        gf256to24_mul(t1, a, b);
        gf256to24_mul(t2, b, a);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: a*b != b*a\n");
            return 1;
        }

        gf256to24_mul(t2, t1, c);
        gf256to24_mul(t3, b, c);
        gf256to24_mul(t1, a, t3);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: (a*b)*c != a*(b*c)\n");
            return 1;
        }

        gf256to24_add(t3, b, c);
        gf256to24_mul(t1, a, t3);
        gf256to24_mul(t2, a, b);
        gf256to24_mul(t3, a, c);
        gf256to24_add(t2, t2, t3);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: a*(b+c) != a*b + a*c\n");
            return 1;
        }
    }
    return 0;
}

/// a^(256^24) = a holds only if the extension polynomial is irreducible
static int test_mul_field() {
    gf256to24 a, b;
    for (uint32_t t = 0; t < (1u << 8); t++) {
        gf256to24_random(a);
        memcpy(b, a, 24);
        for (uint32_t k = 0; k < 192; ++k) {
            gf256to24_mul(b, b, b);
        }

        if (memcmp(a, b, 24) != 0) {
            printf("error gf256to24_mul: a^(256^24) != a\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
static int test_vector_mul() {
    uint8_t a[32] = {0}, b[32] = {0}, c[32];
    gf256to24 d;
    for (uint32_t t = 0; t < (1u << 12); t++) {
        gf256to24_random(a);
        gf256to24_random(b);
        // the 4th limb must be ignored
        a[31] = rand();
        b[31] = rand();

        const __m256i a256 = _mm256_loadu_si256((const __m256i *)a);
        const __m256i b256 = _mm256_loadu_si256((const __m256i *)b);
        _mm256_storeu_si256((__m256i *)c, gf256to24v_mul(a256, b256));

        gf256to24_mul(d, a, b);
        if (memcmp(d, c, 24) != 0) {
            printf("error gf256to24v_mul\n");
            return 1;
        }
        for (uint32_t i = 24; i < 32; i++) {
            if (c[i]) {
                printf("error gf256to24v_mul: 4th limb\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

int main () {
    if (test_mul_modulus()) { return 1; }
    if (test_mul()) { return 1; }
    if (test_mul_field()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
