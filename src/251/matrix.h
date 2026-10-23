#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "vector.h"

/// NOTE: matrices are stored column-major: entry (i, j) is at m[j*nrows + i]
/// NOTE: the output of `mul`/`add_mul` must not alias the inputs

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return zero initialized nrows x ncols matrix
static inline
gf251* gf251_matrix_alloc(const uint32_t nrows,
                          const uint32_t ncols) {
    return (gf251 *)calloc((size_t)nrows * ncols, sizeof(gf251));
}

/// \param m[in]: matrix
/// \param nrows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
/// \return m[i, j]
static inline
gf251 gf251_matrix_get(const gf251 *m,
                       const uint32_t nrows,
                       const uint32_t i,
                       const uint32_t j) {
    return m[(size_t)j*nrows + i];
}

/// \param m[out]: m[i, j] = a
/// \param nrows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
/// \param a[in]: value
static inline
void gf251_matrix_set(gf251 *m,
                      const uint32_t nrows,
                      const uint32_t i,
                      const uint32_t j,
                      const gf251 a) {
    m[(size_t)j*nrows + i] = a;
}

/// \param m[in]: matrix to print
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_print(const gf251 *m,
                        const uint32_t nrows,
                        const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            printf("%3d ", gf251_matrix_get(m, nrows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param m[out]: = 0
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_zero(gf251 *m,
                       const uint32_t nrows,
                       const uint32_t ncols) {
    memset(m, 0, (size_t)nrows * ncols * sizeof(gf251));
}

/// \param m[out]: random matrix
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_random(gf251 *m,
                         const uint32_t nrows,
                         const uint32_t ncols) {
    gf251_vector_random(m, nrows * ncols);
}

/// \param out[out]: = in
/// \param in[in]: input matrix
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_copy(gf251 *out,
                       const gf251 *in,
                       const uint32_t nrows,
                       const uint32_t ncols) {
    memmove(out, in, (size_t)nrows * ncols * sizeof(gf251));
}

/// \param m[out]: identity matrix (1 on the diagonal, 0 elsewhere)
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_id(gf251 *m,
                     const uint32_t nrows,
                     const uint32_t ncols) {
    gf251_matrix_zero(m, nrows, ncols);
    const uint32_t n = nrows < ncols ? nrows : ncols;
    for (uint32_t i = 0; i < n; i++) {
        gf251_matrix_set(m, nrows, i, i, 1);
    }
}

/// \param m1[out]: = m2 + m3
/// \param m2[in]: input matrix
/// \param m3[in]: input matrix
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_add(gf251 *m1,
                      const gf251 *m2,
                      const gf251 *m3,
                      const uint32_t nrows,
                      const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows * ncols; i++) {
        m1[i] = gf251_add(m2[i], m3[i]);
    }
}

/// \param m1[in/out]: += c*m2
/// \param c[in]: scalar
/// \param m2[in]: input matrix
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
static inline
void gf251_matrix_add_scalar(gf251 *m1,
                             const gf251 c,
                             const gf251 *m2,
                             const uint32_t nrows,
                             const uint32_t ncols) {
    gf251_vector_scalar_add(m1, c, m2, nrows * ncols);
}

/// \param r[out]: = A*B, nrows1 x ncols2
/// \param A[in]: nrows1 x ncols1 matrix
/// \param B[in]: ncols1 x ncols2 matrix
/// \param nrows1[in]: number of rows of A
/// \param ncols1[in]: number of columns of A, rows of B
/// \param ncols2[in]: number of columns of B
static inline
void gf251_matrix_mul(gf251 *r,
                      const gf251 *A,
                      const gf251 *B,
                      const uint32_t nrows1,
                      const uint32_t ncols1,
                      const uint32_t ncols2) {
    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            gf251 t = 0;
            for (uint32_t k = 0; k < ncols1; k++) {
                const gf251 a = gf251_matrix_get(A, nrows1, i, k);
                const gf251 b = gf251_matrix_get(B, ncols1, k, j);
                t = gf251_add(t, gf251_mul(a, b));
            }
            gf251_matrix_set(r, nrows1, i, j, t);
        }
    }
}

/// \param r[in/out]: += A*B, nrows1 x ncols2
/// \param A[in]: nrows1 x ncols1 matrix
/// \param B[in]: ncols1 x ncols2 matrix
/// \param nrows1[in]: number of rows of A
/// \param ncols1[in]: number of columns of A, rows of B
/// \param ncols2[in]: number of columns of B
static inline
void gf251_matrix_add_mul(gf251 *r,
                          const gf251 *A,
                          const gf251 *B,
                          const uint32_t nrows1,
                          const uint32_t ncols1,
                          const uint32_t ncols2) {
    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            gf251 t = gf251_matrix_get(r, nrows1, i, j);
            for (uint32_t k = 0; k < ncols1; k++) {
                const gf251 a = gf251_matrix_get(A, nrows1, i, k);
                const gf251 b = gf251_matrix_get(B, ncols1, k, j);
                t = gf251_add(t, gf251_mul(a, b));
            }
            gf251_matrix_set(r, nrows1, i, j, t);
        }
    }
}

/// \param out[out]: = in^T, ncols x nrows matrix
/// \param in[in]: nrows x ncols matrix
/// \param nrows[in]: number of rows of `in`
/// \param ncols[in]: number of columns of `in`
static inline
void gf251_matrix_transpose(gf251 *out,
                            const gf251 *in,
                            const uint32_t nrows,
                            const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf251_matrix_set(out, ncols, j, i, gf251_matrix_get(in, nrows, i, j));
        }
    }
}


#ifdef USE_AVX2

/// \param m1[out]: = m2 + m3
static inline
void gf251_matrix_add_u256(gf251 *m1,
                           const gf251 *m2,
                           const gf251 *m3,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    gf251_vector_add_u256_v2(m1, m2, m3, nrows * ncols);
}

/// \param m1[in/out]: += c*m2
static inline
void gf251_matrix_add_scalar_u256(gf251 *m1,
                                  const gf251 c,
                                  const gf251 *m2,
                                  const uint32_t nrows,
                                  const uint32_t ncols) {
    gf251_vector_scalar_add_u256(m1, c, m2, nrows * ncols);
}

/// \param r[in/out]: += A*B, computed column wise: r[:,j] += B[k,j] * A[:,k]
static inline
void gf251_matrix_add_mul_u256(gf251 *r,
                               const gf251 *A,
                               const gf251 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        gf251 *rj = r + (size_t)j*nrows1;
        for (uint32_t k = 0; k < ncols1; k++) {
            const gf251 b = gf251_matrix_get(B, ncols1, k, j);
            gf251_vector_scalar_add_u256(rj, b, A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// \param r[out]: = A*B
static inline
void gf251_matrix_mul_u256(gf251 *r,
                           const gf251 *A,
                           const gf251 *B,
                           const uint32_t nrows1,
                           const uint32_t ncols1,
                           const uint32_t ncols2) {
    gf251_matrix_zero(r, nrows1, ncols2);
    gf251_matrix_add_mul_u256(r, A, B, nrows1, ncols1, ncols2);
}
#endif


#ifdef USE_AVX512

/// \param m1[out]: = m2 + m3
static inline
void gf251_matrix_add_u512(gf251 *m1,
                           const gf251 *m2,
                           const gf251 *m3,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    gf251_vector_add_u512_v2(m1, m2, m3, nrows * ncols);
}

/// \param m1[in/out]: += c*m2
static inline
void gf251_matrix_add_scalar_u512(gf251 *m1,
                                  const gf251 c,
                                  const gf251 *m2,
                                  const uint32_t nrows,
                                  const uint32_t ncols) {
    gf251_vector_scalar_add_u512(m1, c, m2, nrows * ncols);
}

/// \param r[in/out]: += A*B, computed column wise: r[:,j] += B[k,j] * A[:,k]
static inline
void gf251_matrix_add_mul_u512(gf251 *r,
                               const gf251 *A,
                               const gf251 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        gf251 *rj = r + (size_t)j*nrows1;
        for (uint32_t k = 0; k < ncols1; k++) {
            const gf251 b = gf251_matrix_get(B, ncols1, k, j);
            gf251_vector_scalar_add_u512(rj, b, A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// \param r[out]: = A*B
static inline
void gf251_matrix_mul_u512(gf251 *r,
                           const gf251 *A,
                           const gf251 *B,
                           const uint32_t nrows1,
                           const uint32_t ncols1,
                           const uint32_t ncols2) {
    gf251_matrix_zero(r, nrows1, ncols2);
    gf251_matrix_add_mul_u512(r, A, B, nrows1, ncols1, ncols2);
}
#endif


#ifdef USE_NEON

/// \param m1[out]: = m2 + m3
static inline
void gf251_matrix_add_u128(gf251 *m1,
                           const gf251 *m2,
                           const gf251 *m3,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    gf251_vector_add_u128_v2(m1, m2, m3, nrows * ncols);
}

/// \param m1[in/out]: += c*m2
static inline
void gf251_matrix_add_scalar_u128(gf251 *m1,
                                  const gf251 c,
                                  const gf251 *m2,
                                  const uint32_t nrows,
                                  const uint32_t ncols) {
    gf251_vector_scalar_add_u128(m1, c, m2, nrows * ncols);
}

/// \param r[in/out]: += A*B, computed column wise: r[:,j] += B[k,j] * A[:,k]
static inline
void gf251_matrix_add_mul_u128(gf251 *r,
                               const gf251 *A,
                               const gf251 *B,
                               const uint32_t nrows1,
                               const uint32_t ncols1,
                               const uint32_t ncols2) {
    for (uint32_t j = 0; j < ncols2; j++) {
        gf251 *rj = r + (size_t)j*nrows1;
        for (uint32_t k = 0; k < ncols1; k++) {
            const gf251 b = gf251_matrix_get(B, ncols1, k, j);
            gf251_vector_scalar_add_u128(rj, b, A + (size_t)k*nrows1, nrows1);
        }
    }
}

/// \param r[out]: = A*B
static inline
void gf251_matrix_mul_u128(gf251 *r,
                           const gf251 *A,
                           const gf251 *B,
                           const uint32_t nrows1,
                           const uint32_t ncols1,
                           const uint32_t ncols2) {
    gf251_matrix_zero(r, nrows1, ncols2);
    gf251_matrix_add_mul_u128(r, A, B, nrows1, ncols1, ncols2);
}
#endif
