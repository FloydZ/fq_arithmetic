#pragma once

#include "arith.h"

///
/// @param n
/// @return
static inline gf2* gf2_vector_alloc(const uint32_t n) {
    return (gf2 *)malloc(((n + 7)/8)*sizeof(gf2));
}

/// @param vector
/// @param i
/// @return vector[i]
static inline gf2 gf2_vector_get(const gf2 *vector,
                                  const uint32_t i) {
    return (vector[i / (8)] >> (i % 8)) & 0x01;
}

/// vector[i] = scalar
/// @param vector
/// @param i 
/// @param scalar 
static inline void gf2_vector_set(gf2 *vector,
                                  const uint32_t i,
                                  const gf2 scalar) {
    const uint8_t mask = 0xff ^ (1 << (i % 8));
    vector[i/8] = (vector[i/8] & mask) ^ (scalar << (i % 8));
}

static inline void gf2_vector_rand(gf2 *arg1,
                                  const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = rand();
    }
}

/// arg1 = arg2 + arg3
/// @param arg1
/// @param arg2
/// @param arg3
/// @param d
static inline void gf2_vector_add(gf2 *arg1,
                                  const gf2 *arg2,
                                  const gf2 *arg3,
                                  const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] ^ arg3[i];
    }
}

/// arg1 = arg2 + scalar*arg3
/// @param arg1
/// @param arg2
/// @param scalar
/// @param arg3
/// @param d
static inline void gf2_vector_add_multiple(gf2 *arg1,
                                           const gf2 *arg2,
                                           const gf2 scalar,
                                           const gf2 *arg3,
                                           const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] ^ (arg3[i] & (-scalar));
    }
}

/// arg1 = scalar*arg2
/// @param arg1
/// @param arg2
/// @param value
/// @param d
static inline void gf2_vector_scalar(gf2 *arg1,
                                     const gf2 *arg2,
                                     const gf2 value,
                                     const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] = arg2[i] & (-value);
    }
}

/// arg1 += scalar*arg2
/// @param arg1
/// @param arg2
/// @param value
/// @param d
static inline void gf2_vector_scalar_add(gf2 *arg1,
                                         const gf2 *arg2,
                                         const gf2 value,
                                         const uint32_t d) {
    for (uint32_t i = 0; i < (d+7)/8; i++) {
        arg1[i] ^= arg2[i] & (-value);
    }
}