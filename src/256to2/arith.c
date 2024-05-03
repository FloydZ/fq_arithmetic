#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include "../helper.h"
#include "../256/arith.h"
#include "arith.h"

int main() {
    uint64_t a = 2;

#ifdef USE_AVX2
    v256 v1 = {0}, v3;
    v1.v32[0] = a<<4 ;
    v3.v256 = sqr_simd(v1.v256);
    printf("%u:%lu\n", v3.v32[0]>>4, sqr(a));
#endif
	return 1;
}
