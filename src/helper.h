#ifndef FQ_ARITHMETIC_HELPER_H
#define FQ_ARITHMETIC_HELPER_H

#ifdef USE_AVX2
#include <immintrin.h>
#endif

typedef union v256_t {
    uint32_t v32[8];
#ifdef USE_AVX2
    __m256i v256;
#endif
} v256;
#endif //FQ_ARITHMETIC_HELPER_H
