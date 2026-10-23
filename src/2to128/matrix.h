#pragma once

#include <stdint.h>

#include "arith.h"
#include "vector.h"
#include "../2/matrix.h"

#define gf2to128_matrix_bytes_size(x, y)    ((x) * (y) * sizeof(gf2to128))

/// NOTE: column-major
/// \param m[in]: matrix
/// \param n_rows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
/// \return m[i, j]
static inline
gf2to128 gf2to128_matrix_get(const gf2to128 *m,
                             const uint32_t n_rows,
                             const uint32_t i,
                             const uint32_t j) {
    return m[j*n_rows + i];
}

/// NOTE: column-major
/// \param m[in/out]: matrix, m[i, j] = a
/// \param n_rows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
/// \param a[in]: value to set
static inline
void gf2to128_matrix_set(gf2to128 *m,
                         const uint32_t n_rows,
                         const uint32_t i,
                         const uint32_t j,
                         const gf2to128 a) {
    m[j*n_rows + i] = a;
}

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return pointer to an n_rows*n_cols sized matrix
static inline
gf2to128* gf2to128_matrix_alloc(const uint32_t n_rows,
                                const uint32_t n_cols) {
    return (gf2to128 *) calloc(sizeof(gf2to128), n_rows * n_cols);
}

/// \param matrix[in]: matrix to print
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void
gf2to128_matrix_print(const gf2to128 *matrix,
                      const uint32_t n_rows,
                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to128 t = gf2to128_matrix_get(matrix, n_rows, i, j);
            printf("%016llx%016llx ", (unsigned long long)(t >> 64u), (unsigned long long)t);
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix[out]: = rand()
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to128_matrix_random(gf2to128 *matrix,
                            const uint32_t n_rows,
                            const uint32_t n_cols) {
    gf2to128_vector_random(matrix, n_rows*n_cols);
}

/// \param matrix[out]: = 0
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to128_matrix_zero(gf2to128 *matrix,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = 0;
        }
    }
}

/// \param matrix[out]: = identity (1 on the diagonal, 0 elsewhere)
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to128_matrix_id(gf2to128 *matrix,
                        const uint32_t n_rows,
                        const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            gf2to128_matrix_set(matrix, n_rows, i, j, i == j);
        }
    }
}

/// \param matrix1[out] = matrix2
/// \param matrix2[in]: matrix over gf2to128
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to128_matrix_copy(gf2to128 *matrix1,
                          const gf2to128 *matrix2,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf2to128_matrix_bytes_size(n_rows, n_cols));
}

/// \param out[out]: n_cols x n_rows matrix, out = in^T
/// \param in[in]: n_rows x n_cols matrix
/// \param n_rows[in]: number of rows of `in`
/// \param n_cols[in]: number of cols of `in`
static inline
void gf2to128_matrix_transpose(gf2to128 *__restrict__ out,
                               const gf2to128 *__restrict__ in,
                               const uint32_t n_rows,
                               const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            gf2to128_matrix_set(out, n_cols, j, i, gf2to128_matrix_get(in, n_rows, i, j));
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] matrix3 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add(gf2to128 *matrix1,
                                       const gf2to128 *matrix2,
                                       const gf2to128 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to128_matrix_add_gf2(gf2to128 *matrix1,
                             const gf2 *matrix2,
                             const gf2to128 *matrix3,
                             const uint32_t n_rows,
                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to128 entry1 = gf2to128_matrix_get(matrix3, n_rows, i, j);
            gf2to128 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to128_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add_gf2_v2(gf2to128 *matrix1,
                                              const gf2to128 *matrix2,
                                              const gf2 *matrix3,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to128 entry2;
            const gf2to128 entry1 = gf2to128_matrix_get(matrix2, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix3, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to128_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] scalar scalar over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add_scalar(gf2to128 *matrix1,
                                              const gf2to128 scalar,
                                              const gf2to128 *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to128 entry1 = gf2to128_matrix_get(matrix1, n_rows, i, j);
            const gf2to128 entry2 = gf2to128_matrix_get(matrix2, n_rows, i, j);
            const gf2to128 entry3 = entry1 ^ gf2to128_mul(scalar, entry2);

            gf2to128_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] scalar scalar over gf2to128
/// \param[in] matrix3 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add_scalar_v2(gf2to128 *matrix1,
                                                 const gf2to128 *matrix2,
                                                 const gf2to128 scalar,
                                                 const gf2to128 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to128 entry1 = gf2to128_matrix_get(matrix2, n_rows, i, j);
            const gf2to128 entry2 = gf2to128_matrix_get(matrix3, n_rows, i, j);
            const gf2to128 entry3 = entry1 ^ gf2to128_mul(scalar, entry2);
            gf2to128_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}


/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] scalar scalar over gf2to128
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add_scalar_gf2(gf2to128 *matrix1,
                                                  gf2to128 scalar,
                                                  const gf2 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {

            const gf2to128 entry1 = gf2to128_matrix_get(matrix1, n_rows, i, j);
            const gf2 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            const gf2to128 entry3 = entry1 ^ gf2to128_mul_gf2(scalar, entry2);
            gf2to128_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to128
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to128_matrix_mul_gf2(gf2to128 *result,
                                           const gf2 *matrix1,
                                           const gf2to128 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to128 entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2 entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                gf2to128 entry_k_j = gf2to128_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to128_mul_gf2(entry_k_j, entry_i_k);
            }

            gf2to128_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to128
/// \param[in] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                    respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to128_matrix_mul_gf2_2(gf2to128 *result,
                                             const gf2to128 *matrix1,
                                             const gf2 *matrix2,
                                             const uint32_t n_rows1,
                                             const uint32_t n_cols1,
                                             const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to128 entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2to128 entry_i_k = gf2to128_matrix_get(matrix1, n_rows1, i, k);
                const gf2 entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to128_mul_gf2(entry_i_k, entry_k_j);
            }

            gf2to128_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to128
/// \param[in] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to128_matrix_mul(gf2to128 *result,
                                       const gf2to128 *matrix1,
                                       const gf2to128 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf2to128 entry_i_k, entry_k_j, entry_i_j;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf2to128_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2to128_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to128_mul(entry_i_k, entry_k_j);
            }

            gf2to128_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

///\brief result += matrix1 * matrix2
///\param[out] result Matrix over gf2to128
///\param[in] matrix1 Matrix over gf2to128
///\param[in] matrix2 Matrix over gf2to128
///\param[in] n_rows1 number of rows in matrix1
///\param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                   respectively
///\param[in] n_cols2 number of columns in matrix2
static inline void gf2to128_matrix_add_mul(gf2to128 *result,
                                           const gf2to128 *matrix1,
                                           const gf2to128 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to128 entry_i_k, entry_k_j, entry_i_j;

    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = gf2to128_matrix_get(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf2to128_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2to128_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to128_mul(entry_i_k, entry_k_j);
            }

            gf2to128_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param[out] out Matrix over gf2to128
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to128_matrix_map_gf2(gf2to128 *out,
                                           const gf2 *input,
                                           const uint32_t nrows,
                                           const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; ++i) {
        for (uint32_t j = 0; j < nrows; ++j) {
            const gf2 tmp = gf2_matrix_get(input, nrows, j, i);
            *out = tmp;
            out += 1;
        }
    }
}

#ifdef USE_AVX2
/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] matrix3 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to128_matrix_add_u256(gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const gf2to128 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to128_vector_add_u256_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}


/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to128_matrix_add_gf2_u256(gf2to128 *matrix1,
                                  const gf2 *matrix2,
                                  const gf2to128 *matrix3,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to128_vector_add_gf2_u256_v2(matrix1, matrix3, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t col = 0; col < n_cols; ++col) {
        const gf2 *in2 = matrix2 + col*gf2_col_bytes;
        const gf2to128 *in1 = matrix3 + col*n_rows;
        gf2to128 *out = matrix1 + col*n_rows;
        gf2to128_vector_add_gf2_u256_v2(out, in1, in2, n_rows);
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to128_matrix_add_gf2_u256_v2(gf2to128 *matrix1,
                                     const gf2to128 *matrix2,
                                     const gf2 *matrix3,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to128_matrix_add_gf2_u256(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] scalar scalar over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to128_matrix_add_scalar_u256(gf2to128 *matrix1,
                                     const gf2to128 scalar,
                                     const gf2to128 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to128_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to128
/// \param[in] scalar scalar over gf2to128
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to128_matrix_add_scalar_gf2_u256(gf2to128 *matrix1,
                                                       const gf2to128 scalar,
                                                       const gf2 *matrix2,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to128_vector_scalar_add_gf2_u256_v2(matrix1, scalar, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t col = 0; col < n_cols; col++) {
        gf2to128 *out = matrix1 + col*n_rows;
        const gf2 *in = matrix2 + col*gf2_col_bytes;
        gf2to128_vector_scalar_add_gf2_u256_v2(out, scalar, in, n_rows);
    }
}

/// \brief result += matrix1 * matrix2, computed column-wise:
///     result[:, j] += matrix2[k, j] * matrix1[:, k]
/// \param[out] result Matrix over gf2to128, must not alias the inputs
/// \param[in] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline
void gf2to128_matrix_add_mul_u256(gf2to128 *result,
                                  const gf2to128 *matrix1,
                                  const gf2to128 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf2to128 b = gf2to128_matrix_get(matrix2, n_cols1, k, j);
            gf2to128_vector_scalar_add_u256(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to128, must not alias the inputs
/// \param[in] matrix1 Matrix over gf2to128
/// \param[in] matrix2 Matrix over gf2to128
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline
void gf2to128_matrix_mul_u256(gf2to128 *result,
                              const gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    gf2to128_matrix_zero(result, n_rows1, n_cols2);
    gf2to128_matrix_add_mul_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif

#ifdef USE_AVX512
/// \brief matrix1 = matrix2 + matrix3
static inline
void gf2to128_matrix_add_u512(gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const gf2to128 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to128_vector_add_u512_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 += scalar * matrix2
static inline
void gf2to128_matrix_add_scalar_u512(gf2to128 *matrix1,
                                     const gf2to128 scalar,
                                     const gf2to128 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to128_vector_scalar_add_u512(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief result += matrix1 * matrix2 (result must not alias the inputs)
static inline
void gf2to128_matrix_add_mul_u512(gf2to128 *result,
                                  const gf2to128 *matrix1,
                                  const gf2to128 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf2to128 b = gf2to128_matrix_get(matrix2, n_cols1, k, j);
            gf2to128_vector_scalar_add_u512(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2 (result must not alias the inputs)
static inline
void gf2to128_matrix_mul_u512(gf2to128 *result,
                              const gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    gf2to128_matrix_zero(result, n_rows1, n_cols2);
    gf2to128_matrix_add_mul_u512(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif

#ifdef USE_NEON
/// \brief matrix1 = matrix2 + matrix3
static inline
void gf2to128_matrix_add_u128(gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const gf2to128 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to128_vector_add_u128_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 += scalar * matrix2
static inline
void gf2to128_matrix_add_scalar_u128(gf2to128 *matrix1,
                                     const gf2to128 scalar,
                                     const gf2to128 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to128_vector_scalar_add_u128(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief result += matrix1 * matrix2 (result must not alias the inputs)
static inline
void gf2to128_matrix_add_mul_u128(gf2to128 *result,
                                  const gf2to128 *matrix1,
                                  const gf2to128 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf2to128 b = gf2to128_matrix_get(matrix2, n_cols1, k, j);
            gf2to128_vector_scalar_add_u128(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2 (result must not alias the inputs)
static inline
void gf2to128_matrix_mul_u128(gf2to128 *result,
                              const gf2to128 *matrix1,
                              const gf2to128 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    gf2to128_matrix_zero(result, n_rows1, n_cols2);
    gf2to128_matrix_add_mul_u128(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif
