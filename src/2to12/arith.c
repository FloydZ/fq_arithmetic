#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "arith.h"


#ifdef USE_AVX2
uint32_t test_vector_mul() {
    uint16_t tmp[16];
    for (int i = 0; i < 1u << 11; ++i) {
        for (int j = 0; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to12v_mul_u256(a256, b256);
            _mm256_storeu_si256(tmp, c256);
            const ff_t c = gf2to12_mul(i, j);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}
#endif

int main() {
	ff_t a=1u << 11u,
         b=1u << 11u;
	ff_t c1 = gf2to12_mul(a, b);
    ff_t c2 = gf2to12_mul_v2(a, b);
    assert (c1 == c2);
	//printf("%u\n", c1);

#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

	return 0;
}
