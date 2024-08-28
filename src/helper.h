#ifndef FQ_ARITHMETIC_HELPER_H
#define FQ_ARITHMETIC_HELPER_H

#ifdef USE_AVX2
#include <immintrin.h>
#endif

typedef union v256_t {
    uint8_t  v8[32];
    uint16_t v16[16];
    uint32_t v32[8];
    uint64_t v64[3];
#ifdef USE_AVX2
    __m256i v256;
#endif
} v256;


#endif //FQ_ARITHMETIC_HELPER_H
