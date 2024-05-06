#include <stdio.h>
#include <assert.h>
#include "arith.h"
#include "bitslice.h"

#ifdef USE_AVX2
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
#endif

int main() {
    srand(0);

// test squaring
#ifdef USE_AVX2
    v256 sv1 = {0}, sv3;
    sv1.v32[1] = (210<<16) | (22 << 8);
    sv3.v256 = gf256v_squ_u256(sv1.v256);
    printf("%u %lu\n", sv3.v32[1]>>16, gf256v_squ_u64(210));
#endif
	const uint64_t t1 = gf256v_mul_u64   (121, 8);
	const uint64_t t2 = gf256v_mul_u64_v2(121, 8);
    const uint64_t t3 = gf256v_mul_u64_v3(121, 8);
    const uint64_t t7 = gf256_mul_v2(121, 8);
	printf("%lu %lu %lu\n", t1, t2, t3);

#ifdef USE_AVX2
    v256 v1 = {0}, v2 = {0}, v3, v4;
    v1.v32[0] = 121;
    v2.v32[0] = 12;

    v3.v256 = gf256v_mul_u256_v2(v1.v256, v2.v256);
    v4.v256 = gf256v_mul_u256_v3(v1.v256, v2.v256);
    printf("%u %u\n", v3.v32[0], v4.v32[0]);

    if (check_gf256v_mul_u256()) return 1;
#endif

    uint64_t bs_tmp1[64] = {0};
    uint64_t bs_tmp2[64] = {0};
    uint64_t bs_in[64] = {0};
    uint64_t bs_out[64] = {0};
    // TODO not working: orthogonalize(bs_tmp1, bs_in);
    // TODO not working: orthogonalize(bs_tmp2, bs_in);

    // TODO momentan wirfd b complete ignoriert
    bs_tmp1[0] = 2;
    bs_tmp1[8] = 2;
    MUL(bs_out, bs_tmp1, bs_tmp2);

    //real_ortho(bs_out);

    printf("%ul\n", bs_out[0]);
	return 1;
}
