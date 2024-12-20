#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"
#include "bitslice.h"

#include "../16/vector.h"

#ifdef USE_AVX2

uint32_t test_vector_set_to_gf16_u256() {
    uint32_t ret = 0;
    const uint32_t N = 100;
    gf16 *A = gf16_vector_alloc(N); 
    gf256 *B1 = gf256_vector_alloc(N); 
    gf256 *B2 = gf256_vector_alloc(N); 

    gf16_vector_rand(A, N);
    gf256_vector_set_to_gf16(B1, A, N);
    gf256_vector_set_to_gf16_u256(B2, A, N);

    for (uint32_t i = 0; i < N; i++) {
        if (B1[i] != B2[i]) {
            printf("test_vector_set_to_gf16_u256 errer %d\n", i);
            ret = 1;
        }
    }

    return ret;
}

int check_gf256v_mul_u256() {
    v256 v1 = {0}, v2 = {0}, v3;
    for (uint32_t i = 0; i < 8; ++i) {
        v1.v32[i] = rand();
        v2.v32[i] = rand();
    }

    for (uint32_t k = 0; k < 32; ++k) {
        for (uint32_t i = 0; i < 256; ++i) {
            for (uint32_t j = 0; j < 256; ++j) {
                v1.v8[k] = i;
                v2.v8[k] = j;

                v3.v256 = gf256v_mul_u256_v3(v1.v256, v2.v256);
                if (v3.v8[k] != gf256_mul(i, j)) {
                    return 1;
                }
                assert(v3.v8[k] == gf256_mul(i, j));
            }
        }
    }
    return 0;
}

int check_gf256v_sqr_u256() {
    v256 sv1 = {0}, sv3;
    sv1.v32[1] = (210<<16) | (22 << 8);
    sv3.v256 = gf256v_squ_u256(sv1.v256);
    printf("%u %lu\n", sv3.v32[1]>>16, gf256v_squ_u64(210));

    return true;
}

int check_gf256v_mul_u256_v2() {
    v256 v1 = {0}, v2 = {0}, v3, v4;
    v1.v32[0] = 121;
    v2.v32[0] = 12;

    v3.v256 = gf256v_mul_u256_v2(v1.v256, v2.v256);
    v4.v256 = gf256v_mul_u256_v3(v1.v256, v2.v256);
    printf("%u %u\n", v3.v32[0], v4.v32[0]);

    if (check_gf256v_mul_u256()) return 1;
    return 0;
}

int check_gf256v_mul_scalar_u256() {
    const __m256i a = _mm256_set1_epi8(1);

    v256 c1 = {0}, c2 = {0};
    for (uint32_t b = 0; b < (1u<<8u); b++) {
        const __m256i b1 = _mm256_set1_epi8(1);
        c1.v256 = gf256v_mul_u256(a, b1);
        c2.v256 = gf256v_mul_scalar_u256(a, b);
        for (uint32_t k = 0; k < 32; k++) {
            if (c1.v8[k] != c2.v8[k]) {
                printf("error\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    if(test_vector_set_to_gf16_u256()) { return 1; }
    // if(check_gf256v_mul_u256()) { return 1; }
    // if(check_gf256v_mul_scalar_u256()) { return 1; }
#endif
	return 0;
}
