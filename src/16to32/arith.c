#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "arith.h"

bool test_add() {
    gf16to32 a, b, c;
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        gf16to32_set_u(a, 0);
        gf16to32_set_u(b, i);
        gf16to32_add(c, a, b);

        if (gf16to32_cmp(c, b) != 0) {
            printf("error gf16to32_add\n");
            return 1;
        }
    }

    return 0;
}

bool test_mul() {
    gf16to32 a, b, c;
    for (uint32_t i = 1; i < 1u << 12; ++i) {
        gf16to32_set_u(a, 1);
        gf16to32_set_u(b, i);
        gf16to32_mul_v3(c, a, b);

        if (gf16to32_cmp(c, b) != 0) {
            printf("error gf16to32_mul\n");
            return 1;
        }
    }

    return 0;
}

int main() {
    // if (test_add()) { return 1; }
    if (test_mul()) { return 1; }

    printf("all good\n");
    return 0;
}
