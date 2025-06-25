#pragma once

#include <stdint.h>

#include "arith.h"
#include "vector.h"
#include "../2/matrix.h"


#define gf2to32_matrix_get(m,n,i,j)        m[j*n + i]
#define gf2to32_matrix_set(m,n,i,j,v)      m[j*n + i] = v
#define gf2to32_matrix_bytes_size(x, y)    ((x) * (y) * sizeof(gf2to32))

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return pointer to an n_rows*n_cols sized matrix
static inline
gf2to32* gf2to32_matrix_alloc(const uint32_t n_rows,
                                const uint32_t n_cols) {
    return (gf2to32 *) calloc(sizeof(gf2to32), n_rows * n_cols);
}

/// \param matrix[in]: matrix to print
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void
gf2to32_matrix_print(const gf2to32 *matrix,
                      const uint32_t n_rows,
                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            printf("%llx ", (unsigned long long)gf2to32_matrix_get(matrix, n_rows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix[out]: = rand()
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to32_matrix_random(gf2to32 *matrix,
                            const uint32_t n_rows,
                            const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = rand() & 0xFFF;
        }
    }
}

/// \param matrix[out]: = 0
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to32_matrix_zero(gf2to32 *matrix,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = 0;
        }
    }
}

/// \param matrix1[out] = matrix2
/// \param matrix2[in]: matrix over gf2to32
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf2to32_matrix_copy(gf2to32 *matrix1,
                          const gf2to32 *matrix2,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf2to32_matrix_bytes_size(n_rows, n_cols));
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2to32
/// \param[in] matrix3 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to32_matrix_add(gf2to32 *matrix1,
                                      const gf2to32 *matrix2,
                                      const gf2to32 *matrix3,
                                      const uint32_t n_rows,
                                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to32_matrix_add_gf2(gf2to32 *matrix1,
                             const gf2 *matrix2,
                             const gf2to32 *matrix3,
                             const uint32_t n_rows,
                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to32 entry1 = gf2to32_matrix_get(matrix3, n_rows, i, j);
            gf2to32 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to32_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2to32
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to32_matrix_add_gf2_v2(gf2to32 *matrix1,
                                             const gf2to32 *matrix2,
                                             const gf2 *matrix3,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to32 entry2;
            const gf2to32 entry1 = gf2to32_matrix_get(matrix2, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix3, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to32_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf2to32_matrix_add_scalar(gf2to32 *matrix1,
                                       const gf2to32 scalar,
                                       const gf2to32 *matrix2,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to32 entry1 = gf2to32_matrix_get(matrix1, n_rows, i, j);
            const gf2to32 entry2 = gf2to32_matrix_get(matrix2, n_rows, i, j);
            const gf2to32 entry3 = entry1 ^ gf2to32_mul(scalar, entry2);

            gf2to32_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix3 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to32_matrix_add_scalar_v2(gf2to32 *matrix1,
                                                const gf2to32 *matrix2,
                                                const gf2to32 scalar,
                                                const gf2to32 *matrix3,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to32 entry1 = gf2to32_matrix_get(matrix2, n_rows, i, j);
            const gf2to32 entry2 = gf2to32_matrix_get(matrix3, n_rows, i, j);
            const gf2to32 entry3 = entry1 ^ gf2to32_mul(scalar, entry2);
            gf2to32_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}


/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to32_matrix_add_scalar_gf2(gf2to32 *matrix1,
                                                 gf2to32 scalar,
                                                 const gf2 *matrix2,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {

            const gf2to32 entry1 = gf2to32_matrix_get(matrix1, n_rows, i, j);
            const gf2 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            const gf2to32 entry3 = entry1 ^ gf2to32_mul_gf2(scalar, entry2);
            gf2to32_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to32_matrix_mul_gf2(gf2to32 *result,
                                          const gf2 *matrix1,
                                          const gf2to32 *matrix2,
                                          const uint32_t n_rows1,
                                          const uint32_t n_cols1,
                                          const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to32 entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2 entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                gf2to32 entry_k_j = gf2to32_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to32_mul_gf2(entry_k_j, entry_i_k);
            }

            gf2to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                    respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to32_matrix_mul_gf2_2(gf2to32 *result,
                                            const gf2to32 *matrix1,
                                            const gf2 *matrix2,
                                            const uint32_t n_rows1,
                                            const uint32_t n_cols1,
                                            const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to32 entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2to32 entry_i_k = gf2to32_matrix_get(matrix1, n_rows1, i, k);
                const gf2 entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to32_mul_gf2(entry_i_k, entry_k_j);
            }

            gf2to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to32_matrix_mul(gf2to32 *result,
                                      const gf2to32 *matrix1,
                                      const gf2to32 *matrix2,
                                      const uint32_t n_rows1,
                                      const uint32_t n_cols1,
                                      const uint32_t n_cols2) {
    gf2to32 entry_i_k, entry_k_j, entry_i_j;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf2to32_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2to32_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to32_mul(entry_i_k, entry_k_j);
            }

            gf2to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

///\brief result += matrix1 * matrix2
///\param[out] result Matrix over ff_mu
///\param[in] matrix1 Matrix over ff_mu
///\param[in] matrix2 Matrix over ff_mu
///\param[in] n_rows1 number of rows in matrix1
///\param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                   respectively
///\param[in] n_cols2 number of columns in matrix2
static inline void gf2to32_matrix_add_mul(gf2to32 *result,
                                          const gf2to32 *matrix1,
                                          const gf2to32 *matrix2,
                                          const uint32_t n_rows1,
                                          const uint32_t n_cols1,
                                          const uint32_t n_cols2) {
  gf2to32 entry_i_k, entry_k_j, entry_i_j;

    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = gf2to32_matrix_get(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
              entry_i_k = gf2to32_matrix_get(matrix1, n_rows1, i, k);
              entry_k_j = gf2to32_matrix_get(matrix2, n_cols1, k, j);
              entry_i_j ^= gf2to32_mul(entry_i_k, entry_k_j);
            }

            gf2to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param[out] out Matrix over gf2to32
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to32_matrix_map_gf2(gf2to32 *out,
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
// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2to32
/// \param[in] matrix3 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to12_matrix_add_u256(gf2to32 *matrix1,
                             const gf2to32 *matrix2,
                             const gf2to32 *matrix3,
                             const uint32_t n_rows,
                             const uint32_t n_cols) {
    gf2to32_vector_add_u256_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}


/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to32_matrix_add_gf2_u256(gf2to32 *matrix1,
                                  const gf2 *matrix2,
                                  const gf2to32 *matrix3,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to32_vector_add_gf2_u256_v2(matrix1, matrix3, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t col = 0; col < n_cols; ++col) {
        const gf2 *in2 = matrix2 + col*gf2_col_bytes;
        const gf2to32 *in1 = matrix3 + col*n_rows;
        gf2to32 *out = matrix1 + col*n_rows;
        gf2to32_vector_add_gf2_u256_v2(out, in1, in2, n_rows);
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to32_matrix_add_gf2_u256_v2(gf2to32 *matrix1,
                                     const gf2to32 *matrix2,
                                     const gf2 *matrix3,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf2to32_matrix_add_gf2_u256(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to32
/// \param[in] scalar scalar over gf2to32
/// \param[in] matrix2 Matrix over gf2to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf2to32_matrix_add_scalar_u256(gf2to32 *matrix1,
                                           const gf2to32 scalar,
                                           const gf2to32 *matrix2,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to32_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] scalar scalar over gf2to12
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to32_matrix_add_scalar_gf2_u256(gf2to32 *matrix1,
                                                      const gf2to32 scalar,
                                                      const gf2 *matrix2,
                                                      const uint32_t n_rows,
                                                      const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to32_vector_scalar_add_gf2_u256_v2(matrix1, scalar, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t col = 0; col < n_cols; col++) {
        gf2to32 *out = matrix1 + col*n_rows;
        const gf2 *in = matrix2 + col*gf2_col_bytes;
        gf2to32_vector_scalar_add_gf2_u256_v2(out, scalar, in, n_rows);
    }
}


/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline
void gf2to32_matrix_mul_u256(gf2to32 *result,
                              const gf2to32 *matrix1,
                              const gf2to32 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    // TODO, still need to know which size of the matrices
}
#endif