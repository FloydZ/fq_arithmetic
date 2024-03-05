#include <stdio.h>
#include "arith.h"

int main() {
	const uint64_t t1 = gf256v_mul_u64(121, 12);
	const uint64_t t2 = gf256v_mul_u64_v2(121, 12);
	printf("%lu %lu\n", t1, t2);


    v256 v1 = {0}, v2 = {0}, v3;
    v1.v32[0] = 121;
    v2.v32[0] = 12;

    v3.v256 = gf256v_mul_u256_v2(v1.v256, v2.v256);
    printf("%u\n", v3.v32[0]);

	return 1;
}
