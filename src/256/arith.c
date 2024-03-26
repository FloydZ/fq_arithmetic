#include <stdio.h>
#include "arith.h"
#include "bitslice.h"

int main() {
	const uint64_t t1 = gf256v_mul_u64(121, 12);
	const uint64_t t2 = gf256v_mul_u64_v2(121, 12);
	printf("%lu %lu\n", t1, t2);

#ifdef USE_AVX2
    v256 v1 = {0}, v2 = {0}, v3;
    v1.v32[0] = 121;
    v2.v32[0] = 12;

    v3.v256 = gf256v_mul_u256_v2(v1.v256, v2.v256);
    printf("%u\n", v3.v32[0]);
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
