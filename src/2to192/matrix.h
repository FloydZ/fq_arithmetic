#pragma once

#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "../2/matrix.h"

/// NOTE: all matrices are stored column-major: entry (i, j) is at index
///       j*n_rows + i.
#define gf2to192_matrix_bytes_size(x, y)    ((x) * (y) * sizeof(gf2to192))

/// \param a[out]: = m[i, j]
/// \param m[in]: matrix
/// \param n_rows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
static inline
void gf2to192_matrix_get(gf2to192 a,
                         const gf2to192 *m,
                         const uint32_t n_rows,
                         const uint32_t i,
                         const uint32_t j) {
    gf2to192_copy(a, m[j*n_rows + i]);
}

/// \param m[out]: m[i, j] = a
/// \param n_rows[in]: number of rows
/// \param i[in]: row index
/// \param j[in]: column index
/// \param a[in]: value
static inline
void gf2to192_matrix_set(gf2to192 *m,
                         const uint32_t n_rows,
                         const uint32_t i,
                         const uint32_t j,
                         const gf2to192 a) {
    gf2to192_copy(m[j*n_rows + i], a);
}

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return pointer to a zero initialized n_rows*n_cols sized matrix
static inline
gf2to192* gf2to192_matrix_alloc(const uint32_t n_rows,
                                const uint32_t n_cols) {
    return (gf2to192 *) calloc(sizeof(gf2to192), n_rows * n_cols);
}

/// \param matrix[in]: matrix to print
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void
gf2to192_matrix_print(const gf2to192 *matrix,
                      const uint32_t n_rows,
                      const uint32_t n_cols) {
    gf2to192 a;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_matrix_get(a, matrix, n_rows, i, j);
            printf("%016llx%016llx%016llx ", (unsigned long long)a[2],
                   (unsigned long long)a[1], (unsigned long long)a[0]);
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix[out]: = random
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to192_matrix_random(gf2to192 *matrix,
                            const uint32_t n_rows,
                            const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows*n_cols; i++) {
        gf2to192_set_random(matrix[i]);
    }
}

/// \param matrix[out]: = 0
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to192_matrix_zero(gf2to192 *matrix,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    memset(matrix, 0, gf2to192_matrix_bytes_size(n_rows, n_cols));
}

/// \param matrix1[out] = matrix2
/// \param matrix2[in]: matrix over gf2to192
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to192_matrix_copy(gf2to192 *matrix1,
                          const gf2to192 *matrix2,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf2to192_matrix_bytes_size(n_rows, n_cols));
}

/// \param matrix[out]: = identity (1 on the diagonal, 0 elsewhere)
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to192_matrix_id(gf2to192 *matrix,
                        const uint32_t n_rows,
                        const uint32_t n_cols) {
    gf2to192_matrix_zero(matrix, n_rows, n_cols);
    const uint32_t m = n_rows < n_cols ? n_rows : n_cols;
    for (uint32_t i = 0; i < m; i++) {
        matrix[i*n_rows + i][0] = 1;
    }
}

/// \param out[out]: n_cols x n_rows matrix = in^T
/// \param in[in]: n_rows x n_cols matrix
/// \param n_rows[in]: number of rows of in
/// \param n_cols[in]: number of cols of in
static inline
void gf2to192_matrix_transpose(gf2to192 *out,
                               const gf2to192 *in,
                               const uint32_t n_rows,
                               const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_copy(out[i*n_cols + j], in[j*n_rows + i]);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add(gf2to192 *matrix1,
                                       const gf2to192 *matrix2,
                                       const gf2to192 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        gf2to192_add(matrix1[i], matrix2[i], matrix3[i]);
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to192_matrix_add_gf2(gf2to192 *matrix1,
                             const gf2 *matrix2,
                             const gf2to192 *matrix3,
                             const uint32_t n_rows,
                             const uint32_t n_cols) {
    gf2to192 entry1;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_matrix_get(entry1, matrix3, n_rows, i, j);
            const gf2 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            entry1[0] ^= entry2;
            gf2to192_matrix_set(matrix1, n_rows, i, j, entry1);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_gf2_v2(gf2to192 *matrix1,
                                              const gf2to192 *matrix2,
                                              const gf2 *matrix3,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    gf2to192_matrix_add_gf2(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] scalar scalar over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar(gf2to192 *matrix1,
                                              const gf2to192 scalar,
                                              const gf2to192 *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    gf2to192_vector_scalar_add(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] scalar scalar over gf2to192
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar_v2(gf2to192 *matrix1,
                                                 const gf2to192 *matrix2,
                                                 const gf2to192 scalar,
                                                 const gf2to192 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    gf2to192 entry1, entry2;
    for (uint32_t i = 0; i < n_rows*n_cols; i++) {
        gf2to192_copy(entry1, matrix2[i]);
        gf2to192_mul(entry2, scalar, matrix3[i]);
        gf2to192_add(matrix1[i], entry1, entry2);
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] scalar scalar over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar_gf2(gf2to192 *matrix1,
                                                  const gf2to192 scalar,
                                                  const gf2 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf2to192 entry1, entry3;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_matrix_get(entry1, matrix1, n_rows, i, j);
            const gf2 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            gf2to192_mul_gf2(entry3, scalar, entry2);
            gf2to192_add(entry3, entry3, entry1);
            gf2to192_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to192
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to192_matrix_mul_gf2(gf2to192 *result,
                                           const gf2 *matrix1,
                                           const gf2to192 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to192 entry_i_j, entry_k_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to192_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2 entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                gf2to192_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf2to192_mul_gf2(tmp, entry_k_j, entry_i_k);
                gf2to192_add(entry_i_j, entry_i_j, tmp);
            }
            gf2to192_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to192
/// \param[in] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                    respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to192_matrix_mul_gf2_2(gf2to192 *result,
                                             const gf2to192 *matrix1,
                                             const gf2 *matrix2,
                                             const uint32_t n_rows1,
                                             const uint32_t n_cols1,
                                             const uint32_t n_cols2) {
    gf2to192 entry_i_j, entry_i_k, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to192_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2to192_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                const gf2 entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                gf2to192_mul_gf2(tmp, entry_i_k, entry_k_j);
                gf2to192_add(entry_i_j, entry_i_j, tmp);
            }
            gf2to192_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to192
/// \param[in] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to192_matrix_mul(gf2to192 *result,
                                       const gf2to192 *matrix1,
                                       const gf2to192 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf2to192 entry_i_k, entry_k_j, entry_i_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to192_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2to192_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                gf2to192_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf2to192_mul(tmp, entry_i_k, entry_k_j);
                gf2to192_add(entry_i_j, entry_i_j, tmp);
            }
            gf2to192_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

///\brief result += matrix1 * matrix2
///\param[out] result Matrix over gf2to192
///\param[in] matrix1 Matrix over gf2to192
///\param[in] matrix2 Matrix over gf2to192
///\param[in] n_rows1 number of rows in matrix1
///\param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                   respectively
///\param[in] n_cols2 number of columns in matrix2
static inline void gf2to192_matrix_add_mul(gf2to192 *result,
                                           const gf2to192 *matrix1,
                                           const gf2to192 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf2to192 entry_i_k, entry_k_j, entry_i_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to192_matrix_get(entry_i_j, result, n_rows1, i, j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2to192_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                gf2to192_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf2to192_mul(tmp, entry_i_k, entry_k_j);
                gf2to192_add(entry_i_j, entry_i_j, tmp);
            }
            gf2to192_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param[out] out Matrix over gf2to192
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to192_matrix_map_gf2(gf2to192 *out,
                                           const gf2 *input,
                                           const uint32_t nrows,
                                           const uint32_t ncols) {
    memset(out, 0, sizeof(gf2to192)*nrows*ncols);
    for (uint32_t i = 0; i < ncols; ++i) {
        for (uint32_t j = 0; j < nrows; ++j) {
            const gf2 tmp = gf2_matrix_get(input, nrows, j, i);
            (*out)[0] = tmp;
            out += 1;
        }
    }
}

#ifdef USE_AVX2
/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to192_matrix_add_u256(gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const gf2to192 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to192_vector_add_u256_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to192_matrix_add_gf2_u256(gf2to192 *matrix1,
                                  const gf2 *matrix2,
                                  const gf2to192 *matrix3,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    const uint32_t gf2_col_bytes = (n_rows + 7) / 8;
    for (uint32_t col = 0; col < n_cols; ++col) {
        gf2to192_vector_add_gf2_u256_v2(matrix1 + col*n_rows,
                                        matrix3 + col*n_rows,
                                        matrix2 + col*gf2_col_bytes,
                                        n_rows);
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to192_matrix_add_gf2_u256_v2(gf2to192 *matrix1,
                                     const gf2to192 *matrix2,
                                     const gf2 *matrix3,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to192_matrix_add_gf2_u256(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] scalar scalar over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to192_matrix_add_scalar_u256(gf2to192 *matrix1,
                                     const gf2to192 scalar,
                                     const gf2to192 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to192_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] scalar scalar over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar_gf2_u256(gf2to192 *matrix1,
                                                       const gf2to192 scalar,
                                                       const gf2 *matrix2,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {
    const uint32_t gf2_col_bytes = (n_rows + 7) / 8;
    for (uint32_t col = 0; col < n_cols; col++) {
        gf2to192_vector_scalar_add_gf2_u256_v2(matrix1 + col*n_rows, scalar,
                                               matrix2 + col*gf2_col_bytes,
                                               n_rows);
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf2to192
/// \param[in] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline
void gf2to192_matrix_mul_u256(gf2to192 *result,
                              const gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        gf2to192 *r = result + j*n_rows1;
        gf2to192_vector_zero(r, n_rows1);
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u256(r, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result += matrix1 * matrix2
/// \param[out] result Matrix over gf2to192
/// \param[in] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline
void gf2to192_matrix_add_mul_u256(gf2to192 *result,
                                  const gf2to192 *matrix1,
                                  const gf2to192 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u256(result + j*n_rows1, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}
#endif // USE_AVX2

#ifdef USE_AVX512
/// \brief matrix1 = matrix2 + matrix3
static inline
void gf2to192_matrix_add_u512(gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const gf2to192 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to192_vector_add_u512_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 += scalar * matrix2
static inline
void gf2to192_matrix_add_scalar_u512(gf2to192 *matrix1,
                                     const gf2to192 scalar,
                                     const gf2to192 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to192_vector_scalar_add_u512(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief result = matrix1 * matrix2
static inline
void gf2to192_matrix_mul_u512(gf2to192 *result,
                              const gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        gf2to192 *r = result + j*n_rows1;
        gf2to192_vector_zero(r, n_rows1);
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u512(r, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result += matrix1 * matrix2
static inline
void gf2to192_matrix_add_mul_u512(gf2to192 *result,
                                  const gf2to192 *matrix1,
                                  const gf2to192 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u512(result + j*n_rows1, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}
#endif // USE_AVX512

#ifdef USE_NEON
/// \brief matrix1 = matrix2 + matrix3
static inline
void gf2to192_matrix_add_u128(gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const gf2to192 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf2to192_vector_add_u128_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 += scalar * matrix2
static inline
void gf2to192_matrix_add_scalar_u128(gf2to192 *matrix1,
                                     const gf2to192 scalar,
                                     const gf2to192 *matrix2,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to192_vector_scalar_add_u128(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief result = matrix1 * matrix2
static inline
void gf2to192_matrix_mul_u128(gf2to192 *result,
                              const gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        gf2to192 *r = result + j*n_rows1;
        gf2to192_vector_zero(r, n_rows1);
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u128(r, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result += matrix1 * matrix2
static inline
void gf2to192_matrix_add_mul_u128(gf2to192 *result,
                                  const gf2to192 *matrix1,
                                  const gf2to192 *matrix2,
                                  const uint32_t n_rows1,
                                  const uint32_t n_cols1,
                                  const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            gf2to192_vector_scalar_add_u128(result + j*n_rows1, matrix2[j*n_cols1 + k],
                                            matrix1 + k*n_rows1, n_rows1);
        }
    }
}
#endif // USE_NEON
