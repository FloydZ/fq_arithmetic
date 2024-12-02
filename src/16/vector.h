#pragma once

#include <string.h>
#include <stdlib.h>

#include "arith.h"

/// \return v[i]
gf16 gf16_vector_get(const uint8_t *v,
                     const uint32_t i) {
    const uint32_t pos = i >> 1;
    uint8_t b = 0;
    if (i & 1u) {
        b = v[pos] >> 4;
    } else {
        b = v[pos] & 0xF;
    }
    return b;
}

/// v[i] = a
void gf16_vector_set(uint8_t *v,
                     const uint32_t i,
                     const uint8_t a) {
    const uint32_t pos = i >> 1;
    const uint8_t b = a & 0xF;
    if (i & 1u) {
        v[pos] = (v[pos] & 0xF ) ^ (b << 4);
    } else {
        v[pos] = (v[pos] & 0xF0) ^ b;
    }
}

///
/// @param n
/// @return
static inline gf16* gf16_vector_alloc(const uint32_t n) {
    return (gf16 *)malloc(((n+1)/2)*sizeof(gf16));
}

///
/// @param v
/// @param n
static inline void gf16_vector_zero(gf16 *v,
                                    const uint32_t n) {
    memset(v, 0, ((n + 1)/2) *sizeof(gf16));
}

///
/// @param v
/// @param n
static inline void gf16_vector_rand(gf16 *v,
                                       const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        v[i] = rand() & 0xFFF;
    }
}

///
/// @param v1
/// @param v2
/// @param n
static inline void gf16_vector_copy(gf16 *__restrict__ v1,
                                    const gf16 *__restrict__ v2,
                                    const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        v1[i] = v2[i];
    }
}

///
/// @param out += in
/// @param in
/// @param n
static inline void gf16_vector_add(gf16 *out,
                                   const gf16 *in,
                                   const uint32_t n) {
    for (uint32_t i = 0; i < (n+1)/2; i++) {
        out[i] ^= in[i];
    }
}

