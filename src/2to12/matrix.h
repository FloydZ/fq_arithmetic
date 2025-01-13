#pragma once

#include <stdint.h>

#include "arith.h"
#include "vector.h"
#include "../2/matrix.h"

#define gf2to12_matrix_get(m,n,i,j) m[j*n + i]
#define gf2to12_matrix_set(m,n,i,j,v) m[j*n + i] = v
#define gf2to12_matrix_bytes_size(x, y) ((x) * (y) * sizeof(gf2to12))

static inline gf2to12* gf2to12_matrix_alloc(const uint32_t n_rows,
                                            const uint32_t n_cols) {
    return (gf2to12 *) calloc(2, n_rows * n_cols);
}


static inline void gf2to12_matrix_print(gf2to12 *matrix,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            printf("%4d ", gf2to12_matrix_get(matrix, n_rows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param[out] matrix = rand()
static inline void gf2to12_matrix_random(gf2to12 *matrix,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = rand() & 0xFFF;
        }
    }
}

/// \param[out] matrix = 0
static inline void gf2to12_matrix_zero(gf2to12 *matrix,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            matrix[j*n_rows + i] = 0;
        }
    }
}

/// \param[out] matrix1 = matrix2
static inline void gf2to12_matrix_copy(gf2to12 *matrix1,
                                       const gf2to12 *matrix2,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf2to12_matrix_bytes_size(n_rows, n_cols));
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] matrix3 Matrix over gf2to12
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add(gf2to12 *matrix1,
                                      const gf2to12 *matrix2,
                                      const gf2to12 *matrix3,
                                      const uint32_t n_rows,
                                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to12
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_gf2(gf2to12 *matrix1,
                                          const gf2 *matrix2,
                                          const gf2to12 *matrix3,
                                          const uint32_t n_rows,
                                          const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to12 entry2;
            const gf2to12 entry1 = gf2to12_matrix_get(matrix3, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to12_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_gf2_v2(gf2to12 *matrix1,
                                             const gf2to12 *matrix2,
                                             const gf2 *matrix3,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to12 entry2;
            const gf2to12 entry1 = gf2to12_matrix_get(matrix2, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix3, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf2to12_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf2to12_matrix_add_scalar(gf2to12 *matrix1, 
                                      const gf2to12 scalar,
                                      const gf2to12 *matrix2,
                                      const uint32_t n_rows,
                                      const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to12 entry1 = gf2to12_matrix_get(matrix1, n_rows, i, j);
            const gf2to12 entry2 = gf2to12_matrix_get(matrix2, n_rows, i, j);
            const gf2to12 entry3 = entry1 ^ gf2to12_mul(scalar, entry2);
            
            gf2to12_matrix_set(matrix1, n_rows, i, j, entry3);
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
static inline void gf2to12_matrix_add_scalar_v2(gf2to12 *matrix1, 
                                                const gf2to12 *matrix2, 
                                                const gf2to12 scalar,
                                                const gf2to12 *matrix3,
                                                const uint32_t n_rows, 
                                                const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf2to12 entry1 = gf2to12_matrix_get(matrix2, n_rows, i, j);
            const gf2to12 entry2 = gf2to12_matrix_get(matrix3, n_rows, i, j);
            const gf2to12 entry3 = entry1 ^ gf2to12_mul(scalar, entry2);
            gf2to12_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}


/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_scalar_gf2(gf2to12 *matrix1,
                                                 gf2to12 scalar,
                                                 const gf2 *matrix2,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            
            const gf2to12 entry1 = gf2to12_matrix_get(matrix1, n_rows, i, j);
            const gf2 entry2 = gf2_matrix_get(matrix2, n_rows, i, j);
            const gf2to12 entry3 = entry1 ^ gf2to12_mul_gf2(scalar, entry2);
            gf2to12_matrix_set(matrix1, n_rows, i, j, entry3);
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
static inline void gf2to12_matrix_mul_gf2(gf2to12 *result, 
                                          const gf2 *matrix1, 
                                          const gf2to12 *matrix2,
                                          const uint32_t n_rows1, 
                                          const uint32_t n_cols1, 
                                          const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to12 entry_i_j = 0;
            
            for (uint32_t k = 0; k < n_cols1; k++) {
                gf2 entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                gf2to12 entry_k_j = gf2to12_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to12_mul_gf2(entry_k_j, entry_i_k);
            }
            
            gf2to12_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf2to12_matrix_mul_gf2_2(gf2to12 *result,
                                            const gf2to12 *matrix1,
                                            const gf2 *matrix2,
                                            const uint32_t n_rows1, 
                                            const uint32_t n_cols1,
                                            const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf2to12 entry_i_j = 0;
            
            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf2to12 entry_i_k = gf2to12_matrix_get(matrix1, n_rows1, i, k);
                const gf2 entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to12_mul_gf2(entry_i_k, entry_k_j);
            }
            
            gf2to12_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf2to12_matrix_mul(gf2to12 *result, 
                                      const gf2to12 *matrix1,
                                      const gf2to12 *matrix2,
                                      const uint32_t n_rows1,
                                      const uint32_t n_cols1, 
                                      const uint32_t n_cols2) {
    gf2to12 entry_i_k, entry_k_j, entry_i_j;
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;
            
            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf2to12_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2to12_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2to12_mul(entry_i_k, entry_k_j);
            }
            
            gf2to12_matrix_set(result, n_rows1, i, j, entry_i_j);
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
static inline void gf2to12_matrix_add_mul(gf2to12 *result, 
                                          const gf2to12 *matrix1,
                                          const gf2to12 *matrix2,
                                          const uint32_t n_rows1,
                                          const uint32_t n_cols1, 
                                          const uint32_t n_cols2) {
  gf2to12 entry_i_k, entry_k_j, entry_i_j;

    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = gf2to12_matrix_get(result, n_rows1, i, j);
            
            for (uint32_t k = 0; k < n_cols1; k++) {
              entry_i_k = gf2to12_matrix_get(matrix1, n_rows1, i, k);
              entry_k_j = gf2to12_matrix_get(matrix2, n_cols1, k, j);
              entry_i_j ^= gf2to12_mul(entry_i_k, entry_k_j);
            }
            
            gf2to12_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param[out] out Matrix over gf2to12
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to12_matrix_map_gf2(gf2to12 *out,
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
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] matrix3 Matrix over gf2to12
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_u256(gf2to12 *matrix1,
                                           const gf2to12 *matrix2,
                                           const gf2to12 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to12_vector_add_u256_v2(matrix1, matrix2, matrix3, n_cols * n_rows);
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to12
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_gf2_u256(gf2to12 *matrix1,
                                               const gf2 *matrix2,
                                               const gf2to12 *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to12_vector_add_gf2_u256_v2(matrix1, matrix3, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    uint16_t tmp[8] __attribute__((aligned(16))) = {0};
    for (uint32_t col = 0; col < n_cols; ++col) {
        uint32_t i = n_rows;

        const gf2 *in2 = matrix2 + col*gf2_col_bytes;
        const gf2to12 *in1 = matrix3 + col*n_rows;
        gf2to12 *out = matrix1 + col*n_rows;

        while (i >= 8u) {
            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);
            const __m128i m1 = _mm_loadu_si128((__m128i *)in1);

            _mm_storeu_si128((__m128i *)out, m1 ^ m2);
            i   -= 8u;
            in2 += 1u;
            in1 += 8u;
            out += 8u;
        }

        if (i) {
            for (uint32_t j = 0; j < i; j++) { tmp[j] = in1[j];}
            const __m128i m1 = _mm_loadu_si128((__m128i *)tmp);

            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);

            _mm_storeu_si128((__m128i *)tmp, m2 ^ m1);

            for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] matrix3 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_gf2_u256_v2(gf2to12 *matrix1,
                                                  const gf2to12 *matrix2,
                                                  const gf2 *matrix3,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to12_vector_add_gf2_u256_v2(matrix1, matrix2, matrix3, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    uint16_t tmp[8] __attribute__((aligned(16))) = {0};
    for (uint32_t col = 0; col < n_cols; ++col) {
        uint32_t i = n_rows;

        const gf2 *in2 = matrix3 + col*gf2_col_bytes;
        const gf2to12 *in1 = matrix2 + col*n_rows;
        gf2to12 *out = matrix1 + col*n_rows;

        while (i >= 8u) {
            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);
            const __m128i m1 = _mm_loadu_si128((__m128i *)in1);

            _mm_storeu_si128((__m128i *)out, m1 ^ m2);
            i   -= 8u;
            in2 += 1u;
            in1 += 8u;
            out += 8u;
        }

        if (i) {
            for (uint32_t j = 0; j < i; j++) { tmp[j] = in1[j];}
            const __m128i m1 = _mm_loadu_si128((__m128i *)tmp);
            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);

            _mm_storeu_si128((__m128i *)tmp, m2 ^ m1);

            for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf2to12_matrix_add_scalar_u256(gf2to12 *matrix1, 
                                           const gf2to12 scalar,
                                           const gf2to12 *matrix2,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to12_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}


/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] scalar scalar over gf2to12
/// \param[in] matrix3 Matrix over gf2to12
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_scalar_u256_v2(gf2to12 *matrix1, 
                                                     const gf2to12 *matrix2, 
                                                     const gf2to12 scalar,
                                                     const gf2to12 *matrix3,
                                                     const uint32_t n_rows, 
                                                     const uint32_t n_cols) {
    gf2to12_vector_scalar_add_u256_v2(matrix1, matrix2, scalar, matrix3, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] scalar scalar over gf2to12
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to12_matrix_add_scalar_gf2_u256(gf2to12 *matrix1,
                                                      gf2to12 scalar,
                                                      const gf2 *matrix2,
                                                      const uint32_t n_rows,
                                                      const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to12_vector_scalar_add_gf2_u256_v3(matrix1, scalar, matrix2, n_rows*n_cols);
        return;
    }

    __m128i s = _mm_set1_epi16(scalar);
    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    uint16_t tmp[8] __attribute__((aligned(16))) = {0};
    for (uint32_t col = 0; col < n_cols; ++col) {
        uint32_t i = n_rows;

        const gf2 *in2 = matrix2 + col*gf2_col_bytes;
        gf2to12 *out = matrix1 + col*n_rows;

        while (i >= 8u) {
            const __m128i m1 = _mm_loadu_si128((__m128i *)out);
            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);

            const __m128i t = gf2to12v_mul_gf2_u128(s, m2);

            _mm_storeu_si128((__m128i *)out, m1 ^ t);
            i   -= 8u;
            in2 += 1u;
            out += 8u;
        }

        if (i) {
            for (uint32_t j = 0; j < i; j++) { tmp[j] = out[j];}
            const __m128i m1 = _mm_loadu_si128((__m128i *)tmp);
            const __m128i m2 = gf2to12v_expand_gf2_x8_u256(in2);

            const __m128i t = gf2to12v_mul_gf2_u128(s, m2);
            _mm_storeu_si128((__m128i *)tmp, m1 ^ t);

            for (uint32_t j = 0; j < i; j++) { out[j] = tmp[j];}
        }
    }
}


/// \brief result = matrix1 * matrix2
/// ASSUMES n_rows1 % 8 == 0
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to12_matrix_mul_gf2_full_v1_u256(gf2to12 *result,
                                                        const gf2 *matrix1,
                                                        const gf2to12 *matrix2,
                                                        const uint32_t n_rows1,
                                                        const uint32_t n_cols1,
                                                        const uint32_t n_cols2) {
    // NOTE: set this value to n_rows/8
    __m128i A[4];
    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows1);

    // for each col in A
    for (uint32_t i = 0; i < n_cols1; ++i) {
        const gf2 *m1 = matrix1 + i*gf2_col_bytes;
        // load the full column
        for (uint32_t j = 0; j < n_rows1 / 8; ++j) {
            A[j] = gf2to12v_expand_gf2_x8_u256(m1 + j);
        }

        // for each col in B
        for (uint32_t j = 0; j < n_cols2; ++j) {
            const __m128i b = _mm_set1_epi16(*(matrix2 + j*n_cols1 + i));
            for (uint32_t k = 0; k < n_rows1; k += 8) {
                const __m128i t = gf2to12v_mul_gf2_u128(b, A[k/8]);
                const __m128i c = _mm_loadu_si128((const __m128i *)(result + j*n_rows1 + k));
                const __m128i d = t^c;
                _mm_storeu_si128((__m128i *)(result + j*n_rows1 + k), d);
            }
        }
    }
}

/// \brief result = matrix1 * matrix2
/// NOTE: assumes ncols2 == 1
/// \param[out] result Matrix over gf2to12
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf2to12
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively 
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf2to12_matrix_mul_gf2_vector_u256(gf2to12 *result, 
                                                      const gf2 *matrix1, 
                                                      const gf2to12 *matrix2,
                                                      const uint32_t n_rows1, 
                                                      const uint32_t n_cols1) {
    const __m256i mask = _mm256_setzero_si256();
    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows1);
    const uint32_t limit = n_rows1 % 32;

    uint8_t tmp[64] __attribute__((aligned(32)));
    uint16_t *tmp1 = (uint16_t *)tmp;

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const uint8_t *m1 = matrix1 + col*gf2_col_bytes;
        const __m256i b = _mm256_set1_epi16(*(matrix2 + col));
        gf2to12 *r = result;
        while ((i + 32) <= n_rows1) {
            const __m256i a1 = gf2to12v_expand_gf2_x16_u256(m1 + 0);
            const __m256i a2 = gf2to12v_expand_gf2_x16_u256(m1 + 2);

            const __m256i t1 = _mm256_cmpgt_epi16(a1, mask);
            const __m256i t2 = _mm256_cmpgt_epi16(a2, mask);

            const __m256i c1 = _mm256_loadu_si256((const __m256i *)(r +  0));
            const __m256i c2 = _mm256_loadu_si256((const __m256i *)(r + 16));

            const __m256i d1 = c1 ^ (b & t1);
            const __m256i d2 = c2 ^ (b & t2);

            _mm256_storeu_si256((__m256i *)(r +  0), d1);
            _mm256_storeu_si256((__m256i *)(r + 16), d2);
            m1 += 4;
            r  += 32;
            i  += 32;
        }

        if (limit) {
            for (uint32_t j = 0; j < (limit+7)/8; j++) { tmp[j] = m1[j]; }
            const __m256i a1 = gf2to12v_expand_gf2_x16_u256(tmp + 0);
            const __m256i a2 = gf2to12v_expand_gf2_x16_u256(tmp + 2);

            const __m256i t1 = _mm256_cmpgt_epi16(a1, mask);
            const __m256i t2 = _mm256_cmpgt_epi16(a2, mask);

            for (uint32_t j = 0; j < limit; j++) { tmp[j] = r[j]; }

            const __m256i d1 = b & t1;
            const __m256i d2 = b & t2;

            _mm256_storeu_si256((__m256i *)(tmp1 +  0), d1);
            _mm256_storeu_si256((__m256i *)(tmp1 + 16), d2);
            for (uint32_t j = 0; j < limit; j++) { r[j] ^= tmp1[j]; }

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
static inline void gf2to12_matrix_mul_gf2_u256(gf2to12 *result, 
                                               const gf2 *matrix1, 
                                               const gf2to12 *matrix2,
                                               const uint32_t n_rows1, 
                                               const uint32_t n_cols1, 
                                               const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        gf2to12_matrix_mul_gf2_vector_u256(result, matrix1, matrix2, n_rows1, n_cols1);
        return;
    }

    if (n_rows1 % 8 == 0) {
        gf2to12_matrix_mul_gf2_full_v1_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
        return;
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
static inline void gf2to12_matrix_mul_gf2_u256_v2(gf2to12 *result,
                                                  const gf2to12 *matrix1,
                                                  const gf2 *matrix2,
                                                  const uint32_t n_rows1, 
                                                  const uint32_t n_cols1,
                                                  const uint32_t n_cols2) {
    /// TODO
}

/// \brief result = matrix1 * matrix2
/// NOTE: assumes `n_cols2` = 1
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2
///                     respectively
static inline void gf2to12_matrix_mul_u256_slim(gf2to12 *result,
                                                const gf2to12 *matrix1,
                                                const gf2to12 *matrix2,
                                                const uint32_t n_rows1,
                                                const uint32_t n_cols1) {
    uint16_t tmp[16] __attribute__((aligned(32)));

    const uint32_t limit = n_rows1 % 16;

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const uint16_t *m1 = matrix1 + col*n_rows1;
        const __m256i b = _mm256_set1_epi16(*(matrix2 + col));
        gf2to12 *r = result;
        while ((i + 16) <= n_rows1) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)m1);
            __m256i t = gf2to12v_mul_u256(a, b);
            t ^= _mm256_loadu_si256((__m256i *)(r));
            _mm256_storeu_si256((__m256i *)(r), t);

            m1 += 16;
            r  += 16;
            i  += 16;
        }

        // tail mngt
        if (limit) {
            for (uint32_t j = 0; j < limit; ++j) { tmp[j] = m1[j]; }
            const __m256i a = _mm256_loadu_si256((const __m256i *)tmp);
            __m256i t = gf2to12v_mul_u256(a, b);

            _mm256_storeu_si256((__m256i *)tmp, t);

            for (uint32_t j = 0; j < limit; ++j) { r[j] ^= tmp[j]; }
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
static inline void gf2to12_matrix_mul_u256(gf2to12 *result, 
                                           const gf2to12 *matrix1,
                                           const gf2to12 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1, 
                                           const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        gf2to12_matrix_mul_u256_slim(result, matrix1, matrix2, n_rows1, n_cols1);
        return;
    }

    // set this to the number of element in each column*16 you want
    // to be able to load
    __m256i A[4];

    // number of avc register in each column in A
    const uint32_t nr256= n_rows1 >> 4u;
    const uint32_t tb   = n_rows1 - (nr256 * 16);
    const uint8_t tail  = (nr256 == 0) || (tb != 0);
    uint16_t tmp[16] __attribute__((aligned(32)));

    for (uint32_t j = 0; j < n_cols1; ++j) {
        const uint16_t *m1 = matrix1 + j*n_rows1;
        for (uint32_t i = 0; i < nr256; i++) {
            A[i] = _mm256_loadu_si256((const __m256i *)(m1 + i*16));
        }

        if (tail) {
            for (uint32_t k = 0; k < tb; ++k) {
                tmp[k] = m1[nr256*16 + k];
            }
            A[nr256] = _mm256_load_si256((const __m256i *)tmp);
        }


        // iterate over one row in matrix2
        for (uint32_t i = 0; i < n_cols2; ++i) {
            const __m256i b = _mm256_set1_epi16(matrix2[i*n_cols1 + j]);

            for (uint32_t s = 0; s < nr256; ++s) {
                __m256i r = gf2to12v_mul_u256(A[s], b);
                r ^= _mm256_loadu_si256((__m256i *)(result + i*n_rows1 + s*16));
                _mm256_storeu_si256((__m256i *)(result + i*n_rows1 + s*16), r);
            }

            // tail mngt
            if (tail) {
                const __m256i r = gf2to12v_mul_u256(A[nr256], b);
                _mm256_store_si256((__m256i *)tmp, r);
                for (uint32_t k = 0; k < tb; ++k) {
                    result[i*n_rows1 + nr256*16 + k] ^= tmp[k];
                }
            }
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
static inline void gf2to12_matrix_add_mul_u256(gf2to12 *result, 
                                               const gf2to12 *matrix1,
                                               const gf2to12 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1, 
                                               const uint32_t n_cols2) {
}


/// \param[out] out Matrix over gf2to12
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to12_matrix_map_gf2_u256(gf2to12 *out,
                                               const gf2 *input,
                                               const uint32_t nrows,
                                               const uint32_t ncols) {
    if (nrows % 8 == 0) {
        gf2to12_vector_set_to_gf2_u256(out, input, nrows*ncols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t col = 0; col < ncols; ++col) {
        const gf2 *in2 = input + col*gf2_col_bytes;
        gf2to12 *o = out + col*nrows;

        gf2to12_vector_set_to_gf2_u256(o, in2, nrows);
    }
}
#endif
