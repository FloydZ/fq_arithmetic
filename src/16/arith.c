#include "arith.h"

#define TS 512

uint32_t test_transpose() {
    //uint8_t d[TS]={0}, out[TS]={0};
    //for (uint32_t i = 0; i < TS/4; i++ ) {
    //    d[i] = i%16;
    //}

    //// gf16_matrix_print(d, 16, 32);
    //gf16_transpose_16x16(out +  0, d +   0, 16);
    //gf16_transpose_16x16(out +  8, d + 128, 16);
    //// gf16_matrix_print(out, 32, 16);


#ifdef USE_AVX2
    uint8_t *data1 = calloc(1, 2048);
    uint8_t *data2 = calloc(1, 2048);

    for (uint32_t i = 0; i < 32; i++ ) {
        data1[i] = i%16;
    }

    gf16_matrix_print(data1, 64, 64);
    gf16_transpose_64x64_avx2(data2, data1, 32);
    gf16_matrix_print(data2, 64, 64);
    free(data1); free(data2);
#endif
    return 1;
}

int main() {
    if (test_transpose()) { return 1; }
    return 0;
}
