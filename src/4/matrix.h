#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"
#include "vector.h"

/// NOTE: matrices are stored column major, each column is a packed vector
/// (see `vector.h`) of `gf4_matrix_bytes_per_column(n_rows)` bytes.
#define gf4_matrix_bytes_per_column(n_rows) (((n_rows) + 3) / 4)
#define gf4_matrix_bytes_size(n_rows, n_cols) ((gf4_matrix_bytes_per_column(n_rows)) * (n_cols))

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return zero initialized n_rows x n_cols matrix
static inline gf4* gf4_matrix_alloc(const uint32_t n_rows,
                                    const uint32_t n_cols) {
    return (gf4 *)calloc(gf4_matrix_bytes_size(n_rows, n_cols) + 1, sizeof(gf4));
}

/// \return matrix[i, j]
static inline gf4 gf4_matrix_get(const gf4 *matrix,
                                 const uint32_t n_rows,
                                 const uint32_t i,
                                 const uint32_t j) {
    return gf4_vector_get(matrix + j*gf4_matrix_bytes_per_column(n_rows), i);
}

/// matrix[i, j] = a
static inline void gf4_matrix_set(gf4 *matrix,
                                  const uint32_t n_rows,
                                  const uint32_t i,
                                  const uint32_t j,
                                  const gf4 a) {
    gf4_vector_set(matrix + j*gf4_matrix_bytes_per_column(n_rows), i, a);
}

/// \param matrix[in]: matrix to print
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_print(const gf4 *matrix,
                                    const uint32_t n_rows,
                                    const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            printf("%u ", (unsigned)gf4_matrix_get(matrix, n_rows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix[out]: = 0
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_zero(gf4 *matrix,
                                   const uint32_t n_rows,
                                   const uint32_t n_cols) {
    memset(matrix, 0, gf4_matrix_bytes_size(n_rows, n_cols));
}

/// NOTE: the unused padding bits of each column are set to zero
/// \param matrix[out]: random matrix
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_random(gf4 *matrix,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        gf4_vector_random(matrix + j*bpc, n_rows);
    }
}

/// \param out[out]: out = in
/// \param in[in]: input matrix
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_copy(gf4 *out,
                                   const gf4 *in,
                                   const uint32_t n_rows,
                                   const uint32_t n_cols) {
    memcpy(out, in, gf4_matrix_bytes_size(n_rows, n_cols));
}

/// \param matrix[out]: identity matrix (1 on the diagonal, 0 elsewhere)
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_id(gf4 *matrix,
                                 const uint32_t n_rows,
                                 const uint32_t n_cols) {
    gf4_matrix_zero(matrix, n_rows, n_cols);
    for (uint32_t i = 0; i < MIN(n_rows, n_cols); i++) {
        gf4_matrix_set(matrix, n_rows, i, i, 1);
    }
}

/// matrix1 = matrix2 + matrix3
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_add(gf4 *matrix1,
                                  const gf4 *matrix2,
                                  const gf4 *matrix3,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    const uint32_t n_bytes = gf4_matrix_bytes_size(n_rows, n_cols);
    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf4_matrix_add_scalar(gf4 *matrix1,
                                         const gf4 scalar,
                                         const gf4 *matrix2,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        gf4_vector_scalar_add(matrix1 + j*bpc, scalar, matrix2 + j*bpc, n_rows);
    }
}

/// result = matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf4_matrix_mul(gf4 *result,
                                  const gf4 *matrix1,
                                  const gf4 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf4 entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf4 entry_i_k = gf4_matrix_get(matrix1, n_rows1, i, k);
                const gf4 entry_k_j = gf4_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf4_mul(entry_i_k, entry_k_j);
            }

            gf4_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// result += matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf4_matrix_add_mul(gf4 *result,
                                      const gf4 *matrix1,
                                      const gf4 *matrix2,
                                      const uint32_t n_rows1,
                                      const uint32_t n_cols1,
                                      const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf4 entry_i_j = gf4_matrix_get(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf4 entry_i_k = gf4_matrix_get(matrix1, n_rows1, i, k);
                const gf4 entry_k_j = gf4_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf4_mul(entry_i_k, entry_k_j);
            }

            gf4_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// out = in^T
/// \param out[out]: n_cols x n_rows matrix
/// \param in[in]: n_rows x n_cols matrix
/// \param n_rows[in]: number of rows of `in`
/// \param n_cols[in]: number of cols of `in`
static inline void gf4_matrix_transpose(gf4 *out,
                                        const gf4 *in,
                                        const uint32_t n_rows,
                                        const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf4_matrix_set(out, n_cols, j, i, gf4_matrix_get(in, n_rows, i, j));
        }
    }
}

#ifdef USE_AVX2

/// matrix1 = matrix2 + matrix3
static inline void gf4_matrix_add_u256(gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const gf4 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    const uint32_t n_bytes = gf4_matrix_bytes_size(n_rows, n_cols);
    uint32_t i = 0;
    for (; (i + 32) <= n_bytes; i += 32) {
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)(matrix2 + i));
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)(matrix3 + i));
        _mm256_storeu_si256((__m256i *)(matrix1 + i), gf4v_add_u256(t2, t3));
    }

    for (; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
static inline void gf4_matrix_add_scalar_u256(gf4 *matrix1,
                                              const gf4 scalar,
                                              const gf4 *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows);
    if (n_rows % 4) {
        // columns are not byte aligned, keep the padding untouched
        for (uint32_t j = 0; j < n_cols; j++) {
            gf4_vector_scalar_add_u256(matrix1 + j*bpc, scalar, matrix2 + j*bpc, n_rows);
        }
        return;
    }

    // the matrix is a single contiguous vector
    gf4_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// result += matrix1 * matrix2
/// column wise: result[:, j] += matrix2[k, j] * matrix1[:, k]
static inline void gf4_matrix_add_mul_u256(gf4 *result,
                                           const gf4 *matrix1,
                                           const gf4 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf4 b = gf4_matrix_get(matrix2, n_cols1, k, j);
            gf4_vector_scalar_add_u256(result + j*bpc, b, matrix1 + k*bpc, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
static inline void gf4_matrix_mul_u256(gf4 *result,
                                       const gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf4_matrix_zero(result, n_rows1, n_cols2);
    gf4_matrix_add_mul_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

#endif

#ifdef USE_NEON

/// matrix1 = matrix2 + matrix3
static inline void gf4_matrix_add_u128(gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const gf4 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    const uint32_t n_bytes = gf4_matrix_bytes_size(n_rows, n_cols);
    uint32_t i = 0;
    for (; (i + 16) <= n_bytes; i += 16) {
        const uint8x16_t t2 = vld1q_u8(matrix2 + i);
        const uint8x16_t t3 = vld1q_u8(matrix3 + i);
        vst1q_u8(matrix1 + i, gf4v_add_u128(t2, t3));
    }

    for (; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
static inline void gf4_matrix_add_scalar_u128(gf4 *matrix1,
                                              const gf4 scalar,
                                              const gf4 *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows);
    if (n_rows % 4) {
        // columns are not byte aligned, keep the padding untouched
        for (uint32_t j = 0; j < n_cols; j++) {
            gf4_vector_scalar_add_u128(matrix1 + j*bpc, scalar, matrix2 + j*bpc, n_rows);
        }
        return;
    }

    // the matrix is a single contiguous vector
    gf4_vector_scalar_add_u128(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// result += matrix1 * matrix2
/// column wise: result[:, j] += matrix2[k, j] * matrix1[:, k]
static inline void gf4_matrix_add_mul_u128(gf4 *result,
                                           const gf4 *matrix1,
                                           const gf4 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf4 b = gf4_matrix_get(matrix2, n_cols1, k, j);
            gf4_vector_scalar_add_u128(result + j*bpc, b, matrix1 + k*bpc, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
static inline void gf4_matrix_mul_u128(gf4 *result,
                                       const gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf4_matrix_zero(result, n_rows1, n_cols2);
    gf4_matrix_add_mul_u128(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

#endif

#ifdef USE_AVX512

/// matrix1 = matrix2 + matrix3
static inline void gf4_matrix_add_u512(gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const gf4 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    const uint32_t n_bytes = gf4_matrix_bytes_size(n_rows, n_cols);
    uint32_t i = 0;
    for (; (i + 64) <= n_bytes; i += 64) {
        const __m512i t2 = _mm512_loadu_si512((const void *)(matrix2 + i));
        const __m512i t3 = _mm512_loadu_si512((const void *)(matrix3 + i));
        _mm512_storeu_si512((void *)(matrix1 + i), gf4v_add_u512(t2, t3));
    }

    for (; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
static inline void gf4_matrix_add_scalar_u512(gf4 *matrix1,
                                              const gf4 scalar,
                                              const gf4 *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows);
    if (n_rows % 4) {
        // columns are not byte aligned, keep the padding untouched
        for (uint32_t j = 0; j < n_cols; j++) {
            gf4_vector_scalar_add_u512(matrix1 + j*bpc, scalar, matrix2 + j*bpc, n_rows);
        }
        return;
    }

    // the matrix is a single contiguous vector
    gf4_vector_scalar_add_u512(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// result += matrix1 * matrix2
/// column wise: result[:, j] += matrix2[k, j] * matrix1[:, k]
static inline void gf4_matrix_add_mul_u512(gf4 *result,
                                           const gf4 *matrix1,
                                           const gf4 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    const uint32_t bpc = gf4_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf4 b = gf4_matrix_get(matrix2, n_cols1, k, j);
            gf4_vector_scalar_add_u512(result + j*bpc, b, matrix1 + k*bpc, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
static inline void gf4_matrix_mul_u512(gf4 *result,
                                       const gf4 *matrix1,
                                       const gf4 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf4_matrix_zero(result, n_rows1, n_cols2);
    gf4_matrix_add_mul_u512(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

#endif
