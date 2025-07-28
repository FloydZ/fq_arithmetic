#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include "../helper.h"
#include "arith.h"


#ifdef USE_AVX2
int check_gf256to2v_mul_u256() {
    v256 v1 = {0}, v2 = {0}, v3;
    for (uint32_t i = 0; i < 8; ++i) {
        v1.v32[i] = rand();
        v2.v32[i] = rand();
    }

    for (uint32_t k = 0; k < 1/*16*/; ++k) {
        for (uint32_t i = 0; i < 256*256; ++i) {
            for (uint32_t j = 0; j < 256*256; ++j) {
                // if ((i*j) % (256*256) == 0){ continue;}
                v1.v16[k] = i;
                v2.v16[k] = j;

                v3.v256 = gf256to2v_mul_u256(v1.v256, v2.v256);
                uint32_t c = gf256to2_mul(i, j);
                if (v3.v16[k] != c) {
                    return 1;
                }
                assert(v3.v16[k] == c);
            }
        }
    }
    return 0;
}
#endif

int main() {
#ifdef USE_AVX2
    // if (check_gf256to2v_mul_u256()) return 1;
#endif
	return 0;
}
