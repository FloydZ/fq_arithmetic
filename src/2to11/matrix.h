#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "vector.h"

/// matrices are stored column-major: entry (i, j) is at index j*n_rows + i
#define gf2to11_matrix_bytes_size(x, y) ((x) * (y) * sizeof(gf2to11))

/// \return matrix[i, j]
static inline gf2to11 gf2to11_matrix_get(const gf2to11 *matrix,
                                         const uint32_t n_rows,
                                         const uint32_t i,
                                         const uint32_t j) {
    return matrix[j*n_rows + i];
}

/// matrix[i, j] = a
static inline void gf2to11_matrix_set(gf2to11 *matrix,
                                      const uint32_t n_rows,
                                      const uint32_t i,
                                      const uint32_t j,
                                      const gf2to11 a) {
    matrix[j*n_rows + i] = a;
}

/// \return zero initialized n_rows x n_cols matrix
static inline gf2to11* gf2to11_matrix_alloc(const uint32_t n_rows,
                                            const uint32_t n_cols) {
    return (gf2to11 *)calloc(sizeof(gf2to11), n_rows * n_cols);
}

/// \param matrix[in]: matrix to print
static inline void gf2to11_matrix_print(const gf2to11 *matrix,
                                        const uint32_t n_rows,
                                        const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            printf("%4d ", gf2to11_matrix_get(matrix, n_rows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param[out] matrix = rand()
static inline void gf2to11_matrix_random(gf2to11 *matrix,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = rand() & 0x7FF;
        }
    }
}

/// \param[out] matrix = 0
static inline void gf2to11_matrix_zero(gf2to11 *matrix,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    memset(matrix, 0, gf2to11_matrix_bytes_size(n_rows, n_cols));
}

/// \param[out] matrix = identity (1 on the diagonal, 0 elsewhere)
static inline void gf2to11_matrix_id(gf2to11 *matrix,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to11_matrix_zero(matrix, n_rows, n_cols);
    const uint32_t m = n_rows < n_cols ? n_rows : n_cols;
    for (uint32_t i = 0; i < m; i++) {
        gf2to11_matrix_set(matrix, n_rows, i, i, 1);
    }
}

/// \param[out] matrix1 = matrix2
static inline void gf2to11_matrix_copy(gf2to11 *matrix1,
                                       const gf2to11 *matrix2,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf2to11_matrix_bytes_size(n_rows, n_cols));
}

/// \param[out] out n_cols x n_rows matrix = in^T
/// \param[in] in n_rows x n_cols matrix
static inline void gf2to11_matrix_transpose(gf2to11 *out,
                                            const gf2to11 *in,
                                            const uint32_t n_rows,
                                            const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to11_matrix_set(out, n_cols, j, i, gf2to11_matrix_get(in, n_rows, i, j));
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
static inline void gf2to11_matrix_add(gf2to11 *matrix1,
                                      const gf2to11 *matrix2,
                                      const gf2to11 *matrix3,
                                      const uint32_t n_rows,
                                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// \brief matrix1 += scalar * matrix2
static inline void gf2to11_matrix_add_scalar(gf2to11 *matrix1,
                                             const gf2to11 scalar,
                                             const gf2to11 *matrix2,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] ^= gf2to11_mul(scalar, matrix2[i]);
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result n_rows1 x n_cols2 matrix
/// \param[in] matrix1 n_rows1 x n_cols1 matrix
/// \param[in] matrix2 n_cols1 x n_cols2 matrix
static inline void gf2to11_matrix_mul(gf2to11 *result,
                                      const gf2to11 *matrix1,
                                      const gf2to11 *matrix2,
                                      const uint32_t n_rows1,
                                      const uint32_t n_cols1,
                                      const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to11 entry_i_j = 0;
            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2to11 entry_i_k = gf2to11_matrix_get(matrix1, n_rows1, i, k);
                const gf2to11 entry_k_j = gf2to11_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to11_mul(entry_i_k, entry_k_j);
            }

            gf2to11_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result += matrix1 * matrix2
/// \param[out] result n_rows1 x n_cols2 matrix
/// \param[in] matrix1 n_rows1 x n_cols1 matrix
/// \param[in] matrix2 n_cols1 x n_cols2 matrix
static inline void gf2to11_matrix_add_mul(gf2to11 *result,
                                          const gf2to11 *matrix1,
                                          const gf2to11 *matrix2,
                                          const uint32_t n_rows1,
                                          const uint32_t n_cols1,
                                          const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to11 entry_i_j = gf2to11_matrix_get(result, n_rows1, i, j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2to11 entry_i_k = gf2to11_matrix_get(matrix1, n_rows1, i, k);
                const gf2to11 entry_k_j = gf2to11_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to11_mul(entry_i_k, entry_k_j);
            }

            gf2to11_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

#ifdef USE_AVX2
/// \brief matrix1 = matrix2 + matrix3
static inline void gf2to11_matrix_add_u256(gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const gf2to11 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to11_vector_add_u256_v2(matrix1, matrix2, matrix3, n_rows * n_cols);
}

/// \brief matrix1 += scalar * matrix2
static inline void gf2to11_matrix_add_scalar_u256(gf2to11 *matrix1,
                                                  const gf2to11 scalar,
                                                  const gf2to11 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf2to11_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows * n_cols);
}

/// \brief result += matrix1 * matrix2
static inline void gf2to11_matrix_add_mul_u256(gf2to11 *result,
                                               const gf2to11 *matrix1,
                                               const gf2to11 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    // column-wise axpy, result[:, j] += matrix2[k, j] * matrix1[:, k]
    for (uint32_t j = 0; j < n_cols2; ++j) {
        for (uint32_t k = 0; k < n_cols1; ++k) {
            gf2to11_vector_scalar_add_u256(result + j*n_rows1,
                                           matrix2[j*n_cols1 + k],
                                           matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf2to11_matrix_mul_u256(gf2to11 *result,
                                           const gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to11_matrix_zero(result, n_rows1, n_cols2);
    gf2to11_matrix_add_mul_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

#ifdef USE_AVX512
/// \brief matrix1 = matrix2 + matrix3
static inline void gf2to11_matrix_add_u512(gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const gf2to11 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to11_vector_add_u512_v2(matrix1, matrix2, matrix3, n_rows * n_cols);
}

/// \brief matrix1 += scalar * matrix2
static inline void gf2to11_matrix_add_scalar_u512(gf2to11 *matrix1,
                                                  const gf2to11 scalar,
                                                  const gf2to11 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf2to11_vector_scalar_add_u512(matrix1, scalar, matrix2, n_rows * n_cols);
}

/// \brief result += matrix1 * matrix2
static inline void gf2to11_matrix_add_mul_u512(gf2to11 *result,
                                               const gf2to11 *matrix1,
                                               const gf2to11 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    // column-wise axpy, result[:, j] += matrix2[k, j] * matrix1[:, k]
    for (uint32_t j = 0; j < n_cols2; ++j) {
        for (uint32_t k = 0; k < n_cols1; ++k) {
            gf2to11_vector_scalar_add_u512(result + j*n_rows1,
                                           matrix2[j*n_cols1 + k],
                                           matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf2to11_matrix_mul_u512(gf2to11 *result,
                                           const gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to11_matrix_zero(result, n_rows1, n_cols2);
    gf2to11_matrix_add_mul_u512(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif // USE_AVX512
#endif // USE_AVX2

#ifdef USE_NEON
/// \brief matrix1 = matrix2 + matrix3
static inline void gf2to11_matrix_add_u128(gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const gf2to11 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to11_vector_add_u128_v2(matrix1, matrix2, matrix3, n_rows * n_cols);
}

/// \brief matrix1 += scalar * matrix2
static inline void gf2to11_matrix_add_scalar_u128(gf2to11 *matrix1,
                                                  const gf2to11 scalar,
                                                  const gf2to11 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf2to11_vector_scalar_add_u128(matrix1, scalar, matrix2, n_rows * n_cols);
}

/// \brief result += matrix1 * matrix2
static inline void gf2to11_matrix_add_mul_u128(gf2to11 *result,
                                               const gf2to11 *matrix1,
                                               const gf2to11 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    // column-wise axpy, result[:, j] += matrix2[k, j] * matrix1[:, k]
    for (uint32_t j = 0; j < n_cols2; ++j) {
        for (uint32_t k = 0; k < n_cols1; ++k) {
            gf2to11_vector_scalar_add_u128(result + j*n_rows1,
                                           matrix2[j*n_cols1 + k],
                                           matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf2to11_matrix_mul_u128(gf2to11 *result,
                                           const gf2to11 *matrix1,
                                           const gf2to11 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to11_matrix_zero(result, n_rows1, n_cols2);
    gf2to11_matrix_add_mul_u128(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif // USE_NEON
