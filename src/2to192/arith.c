#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

const uint64_t N = 1u << 10u;

#ifdef USE_AVX2

int test_vector_mul() {
    gf2to192 c1, c2, c3, a, b;
    for (uint64_t i = 0; i < N; i++) {
        gf2to192_set_random(a);
        gf2to192_set_random(b);

        gf2to192_mul(c1, a, b);
        gf2to192v_mul_u256_(c2, a, b);
        gf2to192_mul_v2(c3, a, b);

        for (uint32_t t = 0; t < 3; t++) {
            if (c1[t] != c2[t]) {
                printf("test_vector_mul: %llx %llx\n",
                       (unsigned long long)c1, (unsigned long long)c2);
                return 1;
            }
        }

        for (uint32_t t = 0; t < 3; t++) {
            if (c1[t] != c3[t]) {
                printf("test_vector_mul(3): %llx %llx\n",
                       (unsigned long long)c1, (unsigned long long)c2);
                return 1;
            }
        }
    }
    return 0;
}

#endif

int main() {
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
