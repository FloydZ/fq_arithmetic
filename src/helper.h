#ifndef FQ_ARITHMETIC_HELPER_H
#define FQ_ARITHMETIC_HELPER_H

#include <stdint.h>

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

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#ifdef USE_NEON
/// straightforward implementation of intels _pext_u64
/// /param bitmap
/// /param select_bitmap
/// /return
static inline 
uint64_t _pext_u64(const uint64_t bitmap,
                   uint64_t select_bitmap) {
    uint64_t res = 0;
    for (uint64_t bp = 1; select_bitmap != 0; bp += bp) {
        if (bitmap & select_bitmap & -select_bitmap) {
            res |= bp;
        }
        select_bitmap &= (select_bitmap - 1);
    }

    return res;
}
#endif


// a > b -> b - a is negative
// returns 0xFFFFFFFF if true, 0x00000000 if false
static inline uint32_t ct_is_greater_than(int a, int b) {
    int32_t diff = b - a;
    return (uint32_t) (diff >> (8*sizeof(uint32_t)-1));
}

// if a == b -> 0x00000000, else 0xFFFFFFFF
static inline uint32_t ct_compare_32(int a, int b) {
    return (uint32_t)((-(int32_t)(a ^ b)) >> (8*sizeof(uint32_t)-1));
}

// if a == b -> 0x00, else 0xFF
static inline unsigned char ct_compare_8(unsigned char a, unsigned char b) {
    return (int8_t)((-(int32_t)(a ^ b)) >> (8*sizeof(uint32_t)-1));
}

#endif //FQ_ARITHMETIC_HELPER_H
