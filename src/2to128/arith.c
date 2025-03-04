#include <stdint.h>
#include <stdio.h>
#include "arith.h"


#ifdef USE_AVX2
int test_vector_mul() {
    const uint64_t N = 1u<<10u;
    for (uint64_t i = 0; i < N; i++) {
        const eff_t a = i ^ ((eff_t)i << 64u);
        const eff_t b = i ^ ((eff_t)i << 64u);

        const eff_t c1 = mul(a, b);
        const eff_t c2 = mul_avx(a, b);
        const eff_t c3 = mul_avx2(a, b);

        if (c1 != c2) {
            printf("test_vector_mul: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
        if (c1 != c3) {
            printf("test_vector_mul2: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
    }
    return 0;
}

int test_vector_mul_u256() {
    uint64_t vals1[4], vals2[4];
    const uint64_t N = 1u<<10u;
    for (uint64_t i = 0; i < N; i++) {
        const eff_t a = i ^ ((eff_t)i << 64u);
        const eff_t b = i ^ ((eff_t)i << 64u);
        vals1[0] = a; vals1[1] = a; vals1[2] = a; vals1[3] = a;
        vals2[0] = b; vals2[1] = b; vals2[2] = b; vals2[3] = b;
        const eff_t c1 = mul(a, b);


        const __m256 a_ = _mm256_loadu_si256((__m256i *)vals1);
        const __m256 b_ = _mm256_loadu_si256((__m256i *)vals2);
        const __m256 c_ = gf2to128v_mul_u256(a_, b_);
        _mm256_storeu_si256((__m256i *)vals1, c_);
        eff_t c2 = *(eff_t *)(vals1 + 0);
        eff_t c3 = *(eff_t *)(vals1 + 2);

        if (c1 != c2) {
            printf("test_vector_mul_u256: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
        if (c1 != c3) {
            printf("test_vector_mul_u256_2: %llx %llx\n",
                   (unsigned long long)c1, (unsigned long long)c2);
            return 1;
        }
    }
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
    if (test_vector_mul_u256()) { return 1; }
#endif

    // eff_t c2 = mul(a, 2);
    // printf("%llx\n", c2);
	printf("all good\n");
	return 1;
}
