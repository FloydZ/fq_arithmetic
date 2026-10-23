#include <stdint.h>
#include <stdio.h>

#include "arith.h"

int test_scalar() {
    for (uint32_t a = 0; a < PRIME; a++) {
        if (gf31_add(a, gf31_neg(a)) != 0) {
            printf("error gf31_neg\n");
            return 1;
        }
        for (uint32_t b = 0; b < PRIME; b++) {
            if (gf31_add(a, b) != (a + b) % PRIME ||
                gf31_sub(a, b) != (a + PRIME - b) % PRIME ||
                gf31_mul(a, b) != (a * b) % PRIME) {
                printf("error gf31_add/sub/mul\n");
                return 1;
            }
        }
    }
    return 0;
}

#ifdef USE_AVX2
/// NOTE: the vector functions only reduce partially, so the results are
///     compared mod 31
int test_vector() {
    uint8_t ta[32], tb[32], tc[32];
    for (uint32_t a = 0; a < PRIME; a++) {
        for (uint32_t b = 0; b < PRIME; b++) {
            for (uint32_t k = 0; k < 32; k++) {
                ta[k] = (a + k) % PRIME;
                tb[k] = (b + 7*k) % PRIME;
            }
            const __m256i a256 = _mm256_loadu_si256((const __m256i *)ta);
            const __m256i b256 = _mm256_loadu_si256((const __m256i *)tb);

            _mm256_storeu_si256((__m256i *)tc, gf31v_add_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] % PRIME != gf31_add(ta[k], tb[k])) {
                    printf("error gf31v_add_u256\n");
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tc, gf31v_full_add_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] != gf31_add(ta[k], tb[k])) {
                    printf("error gf31v_full_add_u256\n");
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tc, gf31v_mul_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] % PRIME != gf31_mul(ta[k], tb[k])) {
                    printf("error gf31v_mul_u256\n");
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tc, gf31v_full_mul_u256(a256, b256));
            for (uint32_t k = 0; k < 32; k++) {
                if (tc[k] % PRIME != gf31_mul(ta[k], tb[k])) {
                    printf("error gf31v_full_mul_u256\n");
                    return 1;
                }
            }
        }
    }
    return 0;
}
#endif

int main() {
    if (test_scalar()) { return 1; }
#ifdef USE_AVX2
    if (test_vector()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
