#pragma once

#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "../16/matrix.h"


#define gf16to32_matrix_get(a,m,n,i,j)                  \
    do {                                                \
        for (uint32_t tttt = 0; tttt < 16; tttt++) {    \
            a[tttt] = m[j*n + i][tttt];                 \
        }                                               \
    } while(0);

#define gf16to32_matrix_set(m,n,i,j,v)                  \
    do {                                                \
        for (uint32_t tttt = 0; tttt < 16; tttt++) {    \
            m[j*n + i][tttt] = v[tttt];                 \
        }                                               \
    } while(0);

#define gf16to32_matrix_bytes_size(x, y)    ((x) * (y) * sizeof(gf16to32))

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return pointer to an n_rows*n_cols sized matrix
static inline
gf16to32* gf16to32_matrix_alloc(const uint32_t n_rows,
                                const uint32_t n_cols) {
    return (gf16to32 *) calloc(sizeof(gf16to32), n_rows * n_cols);
}

/// \param matrix[in]: matrix to print
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void
gf16to32_matrix_print(const gf16to32 *matrix,
                      const uint32_t n_rows,
                      const uint32_t n_cols) {
    gf16to32 a;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to32_matrix_get(a, matrix, n_rows, i, j);
            for (uint32_t t = 0; t < 16; t++) {
                printf("%llx ", (unsigned long long)a[t]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix[out]: = rand()
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf16to32_matrix_random(gf16to32 *matrix,
                            const uint32_t n_rows,
                            const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            for (uint32_t t = 0; t < 4; t++) {
                matrix[j*n_rows + i][t] = rand() ^ ((uint64_t)rand());
            }
        }
    }
}

/// \param matrix[out]: = 0
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf16to32_matrix_zero(gf16to32 *matrix,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            for (uint32_t t = 0; t < 16; t++) {
                matrix[j*n_rows + i][t] = 0;
            }
        }
    }
}

/// \param matrix1[out] = matrix2
/// \param matrix2[in]: matrix over gf16to32
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
void gf16to32_matrix_copy(gf16to32 *matrix1,
                          const gf16to32 *matrix2,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf16to32_matrix_bytes_size(n_rows, n_cols));
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16to32
/// \param[in] matrix3 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to32_matrix_add(gf16to32 *matrix1,
                                       const gf16to32 *matrix2,
                                       const gf16to32 *matrix3,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        for (uint32_t t = 0; t < 16; t++) {
            matrix1[i][t] = matrix2[i][t] ^ matrix3[i][t];
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16
/// \param[in] matrix3 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf16to32_matrix_add_gf16(gf16to32 *matrix1,
                             const gf16 *matrix2,
                             const gf16to32 *matrix3,
                             const uint32_t n_rows,
                             const uint32_t n_cols) {
    gf16to32 entry1;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to32_matrix_get(entry1, matrix3, n_rows, i, j);
            const gf16 entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            entry1[0] ^= entry2;
            gf16to32_matrix_set(matrix1, n_rows, i, j, entry1);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16to32
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to32_matrix_add_gf16_v2(gf16to32 *matrix1,
                                             const gf16to32 *matrix2,
                                             const gf16 *matrix3,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    gf16to32_matrix_add_gf16(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf16to32_matrix_add_scalar(gf16to32 *matrix1,
                                       const gf16to32 scalar,
                                       const gf16to32 *matrix2,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    gf16to32 entry1, entry2;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to32_matrix_get(entry1, matrix1, n_rows, i, j);
            gf16to32_matrix_get(entry2, matrix2, n_rows, i, j);
            gf16to32_mul(entry2, scalar, entry2);
            gf16to32_add(entry1, entry1, entry2);
            gf16to32_matrix_set(matrix1, n_rows, i, j, entry1);
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
static inline void gf16to32_matrix_add_scalar_v2(gf16to32 *matrix1,
                                                 const gf16to32 *matrix2,
                                                 const gf16to32 scalar,
                                                 const gf16to32 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    gf16to32 entry1, entry2;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to32_matrix_get(entry1, matrix2, n_rows, i, j);
            gf16to32_matrix_get(entry2, matrix3, n_rows, i, j);
            gf16to32_mul(entry2, scalar, entry2);
            gf16to32_add(entry1, entry1, entry2);
            gf16to32_matrix_set(matrix1, n_rows, i, j, entry1);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to32_matrix_add_scalar_gf16(gf16to32 *matrix1,
                                                   gf16to32 scalar,
                                                   const gf16 *matrix2,
                                                   const uint32_t n_rows,
                                                   const uint32_t n_cols) {
    gf16to32 entry1, entry3;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to32_matrix_get(entry1, matrix1, n_rows, i, j);
            const gf16 entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            gf16to32_mul_gf16(entry3, scalar, entry2);
            gf16to32_add(entry3, entry3, entry1);
            gf16to32_matrix_set(matrix1, n_rows, i, j, entry3);
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
static inline void gf16to32_matrix_mul_gf16(gf16to32 *result,
                                            const gf16 *matrix1,
                                            const gf16to32 *matrix2,
                                            const uint32_t n_rows1,
                                            const uint32_t n_cols1,
                                            const uint32_t n_cols2) {
    gf16to32 entry_i_j, entry_k_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to32_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf16 entry_i_k = gf16_matrix_get(matrix1, n_rows1, i, k);
                gf16to32_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf16to32_mul_gf16(tmp, entry_k_j, entry_i_k);
                gf16to32_add(entry_i_j, entry_i_j, tmp);
            }
            gf16to32_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf16to32_matrix_mul_gf16_2(gf16to32 *result,
                                              const gf16to32 *matrix1,
                                              const gf16 *matrix2,
                                              const uint32_t n_rows1,
                                              const uint32_t n_cols1,
                                              const uint32_t n_cols2) {
    gf16to32 entry_i_j, entry_i_k, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to32_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf16to32_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                const gf16 entry_k_j = gf16_matrix_get(matrix2, n_cols1, k, j);
                gf16to32_mul_gf16(tmp, entry_i_k, entry_k_j);
                gf16to32_add(entry_i_j, entry_i_j, tmp);
            }
            gf16to32_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf16to32_matrix_mul(gf16to32 *result,
                                       const gf16to32 *matrix1,
                                       const gf16to32 *matrix2,
                                       const uint32_t n_rows1,
                                       const uint32_t n_cols1,
                                       const uint32_t n_cols2) {
    gf16to32 entry_i_k, entry_k_j, entry_i_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to32_set_zero(entry_i_j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf16to32_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                gf16to32_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf16to32_mul(tmp, entry_i_k, entry_k_j);
                gf16to32_add(entry_i_j,entry_i_j, tmp);
            }
            gf16to32_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf16to32_matrix_add_mul(gf16to32 *result,
                                           const gf16to32 *matrix1,
                                           const gf16to32 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf16to32 entry_i_k, entry_k_j, entry_i_j, tmp;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to32_matrix_get(entry_i_j, result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                gf16to32_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                gf16to32_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf16to32_mul(tmp, entry_i_k, entry_k_j);
                gf16to32_add(entry_i_j,entry_i_j, tmp);
            }
            gf16to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param[out] out Matrix over gf16to32
/// \param[in] input Matrix over gf16
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf16to32_matrix_map_gf16(gf16to32 *out,
                                            const gf16 *input,
                                            const uint32_t nrows,
                                            const uint32_t ncols) {
    memset(out, 0, sizeof(gf16to32)*nrows*ncols);
    for (uint32_t i = 0; i < ncols; ++i) {
        for (uint32_t j = 0; j < nrows; ++j) {
            const gf16 tmp = gf16_matrix_get(input, nrows, j, i);
            **out = tmp;
            out += 1;
        }
    }
}

#ifdef USE_AVX2
// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16to32
/// \param[in] matrix3 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf16to12_matrix_add_u256(gf16to32 *matrix1,
                              const gf16to32 *matrix2,
                              const gf16to32 *matrix3,
                              const uint32_t n_rows,
                              const uint32_t n_cols) {
    gf16to32_vector_add_u128_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}


/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16
/// \param[in] matrix3 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf16to32_matrix_add_gf16_u256(gf16to32 *matrix1,
                                  const gf16 *matrix2,
                                  const gf16to32 *matrix3,
                                  const uint32_t n_rows,
                                  const uint32_t n_cols) {
    if (n_rows % 2 == 0) {
        gf16to32_vector_add_gf16_u256_v2(matrix1, matrix3, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf16_col_bytes = gf16_matrix_bytes_per_column(n_rows);
    __uint128_t buffer[8];
    for (uint32_t col = 0; col < n_cols; ++col) {
        uint32_t i = n_rows;

        const gf16 *in2 = matrix2 + col*gf16_col_bytes;
        const gf16to32 *in1 = matrix3 + col*n_rows;
        gf16to32 *out = matrix1 + col*n_rows;
        gf16to32_vector_add_gf16_u256_v2(out, in1, in2, n_rows);

    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] matrix2 Matrix over gf16
/// \param[in] matrix3 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf16to32_matrix_add_gf16_u256_v2(gf16to32 *matrix1,
                                     const gf16to32 *matrix2,
                                     const gf16 *matrix3,
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    gf16to32_matrix_add_gf16_u256(matrix1, matrix3, matrix2, n_rows, n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf16to32
/// \param[in] scalar scalar over gf16to32
/// \param[in] matrix2 Matrix over gf16to32
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf16to32_matrix_add_scalar_u256(gf16to32 *matrix1,
                                           const gf16to32 scalar,
                                           const gf16to32 *matrix2,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf16to32_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf16to12
/// \param[in] scalar scalar over gf16to12
/// \param[in] matrix2 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to32_matrix_add_scalar_gf16_u256(gf16to32 *matrix1,
                                                      const gf16to32 scalar,
                                                      const gf16 *matrix2,
                                                      const uint32_t n_rows,
                                                      const uint32_t n_cols) {
    if (n_rows % 2 == 0) {
        gf16to32_vector_scalar_add_gf16_u256_v2(matrix1, scalar, matrix2, n_rows*n_cols);
        return;
    }

    __m256i buffer[4];
    const __m256i s = _mm256_loadu_si256((const __m256i *)scalar);
    const uint32_t gf16_col_bytes = gf16_matrix_bytes_per_column(n_rows);

    for (uint32_t col = 0; col < n_cols; col++) {
        uint32_t i = 0;

        gf16to32 *out = matrix1 + col*n_rows;
        const gf16 *in = matrix2 + col*gf16_col_bytes;
        gf16to32_vector_scalar_add_gf16_u256_v2(out, scalar, in, n_rows);
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
void gf16to32_matrix_mul_u256(gf16to32 *result,
                              const gf16to32 *matrix1,
                              const gf16to32 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    gf16to32 entry_i_k, entry_k_j, entry_i_j, tmp;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to32_set_zero(entry_i_j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                gf16to32_matrix_get(entry_i_k, matrix1, n_rows1, i, k);
                gf16to32_matrix_get(entry_k_j, matrix2, n_cols1, k, j);
                gf16to32_mul(tmp, entry_i_k, entry_k_j);
                gf16to32_add(entry_i_j, entry_i_j, tmp);
            }

            gf16to32_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }

}
#endif

