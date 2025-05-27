#include <stdint.h>
#include <stdio.h>

#include "arith.h"

const uint64_t N = 1u << 10;

uint32_t test_add() {
    for (int a = 0; a < N; ++a) {
        for (int b = 0; b < N; ++b) {
            uint32_t i =  rand();
            uint32_t j =  rand();

            const uint64_t  c1 = gf2to64_mul(i, j);

        }
    }
    return 0;
}

#ifdef USE_AVX2
uint32_t test_vector_add() {
    return 0;
}

uint32_t test_vector_mul() {
    return 0;
}
#endif

int main() {
    if (test_add()) { return 1; }

#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
