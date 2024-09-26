#include <stdint.h>
#include <stdbool.h>

#include "arith.h"

bool test_add() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t = gf16to3_add(i, i);
        if (t != 0) {
            return 1;
        }
    }

    return 0;
}

bool test_mul() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t1 = gf16to3_mul(i, 1);
        if (t1 != i) { return 1; }

        const gf16to3 t2 = gf16to3_mul(1, i);
        if (t2 != i) { return 1; }
    }

    return 0;
}
int main() {
    if (test_add()) { return 1; }
    if (test_mul()) { return 1; }
    return 0;
}
