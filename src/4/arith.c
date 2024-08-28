#include <stdbool.h>
#include <stdio.h>

#include "arith.h"
#define TEST_SIZE 19999

void print_mul_table() {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const ff_t c = gf4_mul(i, j);
            printf("%d = %d * %d\n", c, i, j);
        }
    }
}
bool test1() {
	if (gf4_add(0, 1) != 1) { return false; }
	if (gf4_add(0, 2) != 2) { return false; }
	if (gf4_add(0, 3) != 3) { return false; }
	if (gf4_add(0, 0) != 0) { return false; }

    if (gf4_mul(0, 0) != 0) { return false; }
    if (gf4_mul(0, 1) != 0) { return false; }
    if (gf4_mul(0, 2) != 0) { return false; }
    if (gf4_mul(0, 3) != 0) { return false; }
	return true;
}

bool test_mul_u64(){
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j1 = 0; j1 < 4; ++j1) {
            for (uint32_t j2 = 0; j2 < 4; ++j2) {
                uint64_t a = ((uint64_t)j1) << (i * 2u);
                uint64_t b = ((uint64_t)j2) << (i * 2u);

                const ff_t t1 = gf4_mul_u64(a, b) >> (i*2);
                const ff_t t2 = gf4_mul(j1, j2);
                const ff_t t3 = gf4_mul_v2(j1, j2);
                if (t1 != t2) {
                    return false;
                }

                if (t2 != t3) {
                    return false;
                }
            }
        }
    }

    return true;
}


#ifdef USE_AVX2
bool test_mul_u256() {
    uint64_t tmp[4] = {0};
    for (uint32_t i = 0; i < 32; ++i) {
        for (uint32_t j1 = 0; j1 < 4; ++j1) {
            for (uint32_t j2 = 0; j2 < 4; ++j2) {
                uint64_t a = ((uint64_t)j1) << (i * 2u);
                uint64_t b = ((uint64_t)j2) << (i * 2u);

                __m256i t1 = {a,0,0,0};
                __m256i t2 = {b,0,0,0};

                const __m256i c = gf4_mul_u256(t1, t2);
                _mm256_store_si256((__m256i *)tmp, c);

                const ff_t g = gf4_mul(j1, j2);
                if ((tmp[0] >> (i*2u)) != g) {
                    return false;
                }
            }
        }
    }

    return true;
}
#endif

int main() {
    print_mul_table();

	if (!test1()) { return -1; }
    if (!test_mul_u64()) { return -1; }

#ifdef USE_AVX2
    if (!test_mul_u256()) { return -1; }
#endif
	return 0;
}
