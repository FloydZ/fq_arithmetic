#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "vector.h"

// matrices are stored column-major: entry (i, j) is at index j*nrows + i

/// \return zero initialized nrows x ncols matrix
static inline
gf256to16* gf256to16_matrix_alloc(const uint32_t nrows,
                                  const uint32_t ncols) {
    return (gf256to16 *)calloc((size_t)nrows * ncols, sizeof(gf256to16));
}

/// out = m[i, j]
static inline
void gf256to16_matrix_get(gf256to16 out,
                          const gf256to16 *m,
                          const uint32_t nrows,
                          const uint32_t i,
                          const uint32_t j) {
    memcpy(out, m[(size_t)j*nrows + i], sizeof(gf256to16));
}

/// m[i, j] = a
static inline
void gf256to16_matrix_set(gf256to16 *m,
                          const uint32_t nrows,
                          const uint32_t i,
                          const uint32_t j,
                          const gf256to16 a) {
    memcpy(m[(size_t)j*nrows + i], a, sizeof(gf256to16));
}

/// prints the matrix row by row, each element as its 16 bytes (byte 0 first)
static inline
void gf256to16_matrix_print(const gf256to16 *m,
                            const uint32_t nrows,
                            const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            for (uint32_t k = 0; k < 16; k++) {
                printf("%02x", m[(size_t)j*nrows + i][k]);
            }
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

/// m = 0
static inline
void gf256to16_matrix_zero(gf256to16 *m,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    gf256to16_vector_zero(m, nrows*ncols);
}

/// m = random
static inline
void gf256to16_matrix_random(gf256to16 *m,
                             const uint32_t nrows,
                             const uint32_t ncols) {
    gf256to16_vector_random(m, nrows*ncols);
}

/// out = in
static inline
void gf256to16_matrix_copy(gf256to16 *out,
                           const gf256to16 *in,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    gf256to16_vector_copy(out, in, nrows*ncols);
}

/// m = identity (1 on the diagonal, 0 elsewhere)
static inline
void gf256to16_matrix_id(gf256to16 *m,
                         const uint32_t nrows,
                         const uint32_t ncols) {
    gf256to16 one = {0};
    one[0] = 1;
    gf256to16_matrix_zero(m, nrows, ncols);
    for (uint32_t i = 0; i < MIN(nrows, ncols); i++) {
        gf256to16_matrix_set(m, nrows, i, i, one);
    }
}

/// m1 = m2 + m3
static inline
void gf256to16_matrix_add(gf256to16 *m1,
                          const gf256to16 *m2,
                          const gf256to16 *m3,
                          const uint32_t nrows,
                          const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows*ncols; i++) {
        gf256to16_add(m1[i], m2[i], m3[i]);
    }
}

/// m1 += c*m2
static inline
void gf256to16_matrix_add_scalar(gf256to16 *m1,
                                 const gf256to16 c,
                                 const gf256to16 *m2,
                                 const uint32_t nrows,
                                 const uint32_t ncols) {
    gf256to16_vector_scalar_add(m1, c, m2, nrows*ncols);
}

/// r += A*B, A: nrows1 x ncols1, B: ncols1 x ncols2, r: nrows1 x ncols2
/// NOTE: r must not alias A or B
static inline
void gf256to16_matrix_add_mul(gf256to16 *r,
                              const gf256to16 *A,
                              const gf256to16 *B,
                              const uint32_t nrows1,
                              const uint32_t ncols1,
                              const uint32_t ncols2) {
    gf256to16 t, a, b, p;
    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            gf256to16_matrix_get(t, r, nrows1, i, j);
            for (uint32_t k = 0; k < ncols1; k++) {
                gf256to16_matrix_get(a, A, nrows1, i, k);
                gf256to16_matrix_get(b, B, ncols1, k, j);
                gf256to16_mul(p, a, b);
                gf256to16_add(t, t, p);
            }
            gf256to16_matrix_set(r, nrows1, i, j, t);
        }
    }
}

/// r = A*B, A: nrows1 x ncols1, B: ncols1 x ncols2, r: nrows1 x ncols2
/// NOTE: r must not alias A or B
static inline
void gf256to16_matrix_mul(gf256to16 *r,
                          const gf256to16 *A,
                          const gf256to16 *B,
                          const uint32_t nrows1,
                          const uint32_t ncols1,
                          const uint32_t ncols2) {
    gf256to16_matrix_zero(r, nrows1, ncols2);
    gf256to16_matrix_add_mul(r, A, B, nrows1, ncols1, ncols2);
}

/// out = in^T, in: nrows x ncols, out: ncols x nrows
static inline
void gf256to16_matrix_transpose(gf256to16 *out,
                                const gf256to16 *in,
                                const uint32_t nrows,
                                const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf256to16_matrix_set(out, ncols, j, i, in[(size_t)j*nrows + i]);
        }
    }
}

#ifdef USE_AVX2
/// m1 = m2 + m3
static inline
void gf256to16_matrix_add_u256(gf256to16 *m1,
                               const gf256to16 *m2,
                               const gf256to16 *m3,
                               const uint32_t nrows,
                               const uint32_t ncols) {
    gf256to16_vector_add_u256_v2(m1, m2, m3, nrows*ncols);
}

/// m1 += c*m2
static inline
void gf256to16_matrix_add_scalar_u256(gf256to16 *m1,
                                      const gf256to16 c,
                                      const gf256to16 *m2,
                                      const uint32_t nrows,
                                      const uint32_t ncols) {
    gf256to16_vector_scalar_add_u256(m1, c, m2, nrows*ncols);
}

/// r += A*B, computed column wise: r[:, j] += B[k, j] * A[:, k]
static inline
void gf256to16_matrix_add_mul_u256(gf256to16 *r,
                                   const gf256to16 *A,
                                   const gf256to16 *B,
                                   const uint32_t nrows1,
                                   const uint32_t ncols1,
                                   const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        for (uint32_t k = 0; k < ncols1; k++) {
            gf256to16_vector_scalar_add_u256(r + (size_t)j*nrows1, B[(size_t)j*ncols1 + k],
                                             A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// r = A*B
static inline
void gf256to16_matrix_mul_u256(gf256to16 *r,
                               const gf256to16 *A,
                               const gf256to16 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    gf256to16_matrix_zero(r, nrows1, ncols2);
    gf256to16_matrix_add_mul_u256(r, A, B, nrows1, ncols1, ncols2);
}
#endif // USE_AVX2

#ifdef USE_NEON
/// m1 = m2 + m3
static inline
void gf256to16_matrix_add_u128(gf256to16 *m1,
                               const gf256to16 *m2,
                               const gf256to16 *m3,
                               const uint32_t nrows,
                               const uint32_t ncols) {
    gf256to16_vector_add_u128_v2(m1, m2, m3, nrows*ncols);
}

/// m1 += c*m2
static inline
void gf256to16_matrix_add_scalar_u128(gf256to16 *m1,
                                      const gf256to16 c,
                                      const gf256to16 *m2,
                                      const uint32_t nrows,
                                      const uint32_t ncols) {
    gf256to16_vector_scalar_add_u128(m1, c, m2, nrows*ncols);
}

/// r += A*B, computed column wise: r[:, j] += B[k, j] * A[:, k]
static inline
void gf256to16_matrix_add_mul_u128(gf256to16 *r,
                                   const gf256to16 *A,
                                   const gf256to16 *B,
                                   const uint32_t nrows1,
                                   const uint32_t ncols1,
                                   const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        for (uint32_t k = 0; k < ncols1; k++) {
            gf256to16_vector_scalar_add_u128(r + (size_t)j*nrows1, B[(size_t)j*ncols1 + k],
                                             A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// r = A*B
static inline
void gf256to16_matrix_mul_u128(gf256to16 *r,
                               const gf256to16 *A,
                               const gf256to16 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    gf256to16_matrix_zero(r, nrows1, ncols2);
    gf256to16_matrix_add_mul_u128(r, A, B, nrows1, ncols1, ncols2);
}
#endif // USE_NEON

#ifdef USE_AVX512
/// m1 = m2 + m3
static inline
void gf256to16_matrix_add_u512(gf256to16 *m1,
                               const gf256to16 *m2,
                               const gf256to16 *m3,
                               const uint32_t nrows,
                               const uint32_t ncols) {
    gf256to16_vector_add_u512_v2(m1, m2, m3, nrows*ncols);
}

/// m1 += c*m2
static inline
void gf256to16_matrix_add_scalar_u512(gf256to16 *m1,
                                      const gf256to16 c,
                                      const gf256to16 *m2,
                                      const uint32_t nrows,
                                      const uint32_t ncols) {
    gf256to16_vector_scalar_add_u512(m1, c, m2, nrows*ncols);
}

/// r += A*B, computed column wise: r[:, j] += B[k, j] * A[:, k]
static inline
void gf256to16_matrix_add_mul_u512(gf256to16 *r,
                                   const gf256to16 *A,
                                   const gf256to16 *B,
                                   const uint32_t nrows1,
                                   const uint32_t ncols1,
                                   const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        for (uint32_t k = 0; k < ncols1; k++) {
            gf256to16_vector_scalar_add_u512(r + (size_t)j*nrows1, B[(size_t)j*ncols1 + k],
                                             A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// r = A*B
static inline
void gf256to16_matrix_mul_u512(gf256to16 *r,
                               const gf256to16 *A,
                               const gf256to16 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    gf256to16_matrix_zero(r, nrows1, ncols2);
    gf256to16_matrix_add_mul_u512(r, A, B, nrows1, ncols1, ncols2);
}
#endif // USE_AVX512
