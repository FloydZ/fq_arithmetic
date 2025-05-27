#include <stdint.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"


#ifdef USE_AVX2
uint32_t test_vector_add() {
    return 0;
}

uint32_t test_vector_mul() {
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if (test_vector_add()) { return 1; }
    if (test_vector_mul()) { return 1; }
#endif
    return 0;
}
