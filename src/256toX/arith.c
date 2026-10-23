#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arith.h"

#define N (1u << 8)

static void gf256toX_random(gf256toX a) {
    for (uint32_t i = 0; i < EXTENSION_DEGREE; i++) {
        a[i] = rand();
    }
}

/// a^(256^EXTENSION_DEGREE) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    gf256toX a, b;
    for (uint32_t t = 0; t < N; ++t) {
        gf256toX_random(a);
        gf256toX_set(b, a);
        for (uint32_t k = 0; k < 8*EXTENSION_DEGREE; ++k) {
            gf256toX_mul(b, b, b);
        }

        if (memcmp(a, b, EXTENSION_DEGREE) != 0) {
            printf("error test_mul_field\n");
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    gf256toX a, b, c, one = {0}, t1, t2, t3;
    one[0] = 1;
    for (uint32_t t = 0; t < N; ++t) {
        gf256toX_random(a);
        gf256toX_random(b);
        gf256toX_random(c);

        gf256toX_mul(t1, a, one);
        if (memcmp(t1, a, EXTENSION_DEGREE) != 0) {
            printf("error test_mul: a*1\n");
            return 1;
        }

        gf256toX_mul(t1, a, b);
        gf256toX_mul(t2, b, a);
        if (memcmp(t1, t2, EXTENSION_DEGREE) != 0) {
            printf("error test_mul: a*b\n");
            return 1;
        }

        gf256toX_mul(t2, t1, c);
        gf256toX_mul(t3, b, c);
        gf256toX_mul(t1, a, t3);
        if (memcmp(t1, t2, EXTENSION_DEGREE) != 0) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }

        gf256toX_add(t3, b, c);
        gf256toX_mul(t1, a, t3);
        gf256toX_mul(t2, a, b);
        gf256toX_mul(t3, a, c);
        gf256toX_add(t2, t2, t3);
        if (memcmp(t1, t2, EXTENSION_DEGREE) != 0) {
            printf("error test_mul: a*(b+c)\n");
            return 1;
        }
    }
    return 0;
}

int main () {
    if (test_mul()) { return 1; }
    if (test_mul_field()) { return 1; }

    printf("all good\n");
    return 0;
}
