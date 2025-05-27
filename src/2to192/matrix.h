#pragma once

#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "../2/matrix.h"


#define gf2to192_matrix_get(a, m,n,i,j)                 \
    do {                                                \
        for (uint32_t tttt = 0; tttt < 3; tttt++) {     \
            a[tttt] = m[j*n + i][tttt];                 \
        }                                               \
    } while(0);


#define gf2to192_matrix_set(m,n,i,j,v)                  \
    do {                                                \
        for (uint32_t tttt = 0; tttt < 3; tttt++) {     \
            m[j*n + i][tttt] = v[tttt];                 \
        }                                               \
    } while(0);

#define gf2to192_matrix_bytes_size(x, y)    ((x) * (y) * sizeof(gf2to192))

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
/// \return pointer to an n_rows*n_cols sized matrix
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
            for (uint32_t t = 0; t < 4; t++) {
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
void gf2to192_matrix_random(gf2to192 *matrix,
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
void gf2to192_matrix_zero(gf2to192 *matrix,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        for (uint32_t i = 0; i < n_rows; i++) {
            for (uint32_t t = 0; t < 3; t++) {
                matrix[j*n_rows + i][t] = 0;
            }
        }
    }
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
        for (uint32_t t = 0; t < 3; t++) {
            matrix1[i][t] = matrix2[i][t] ^ matrix3[i][t];
        }
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
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf2to192_matrix_add_scalar(gf2to192 *matrix1,
                                       const gf2to192 scalar,
                                       const gf2to192 *matrix2,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    gf2to192 entry1, entry2;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_matrix_get(entry1, matrix1, n_rows, i, j);
            gf2to192_matrix_get(entry2, matrix2, n_rows, i, j);
            gf2to192_mul(entry2, scalar, entry2);
            gf2to192_add(entry1, entry1, entry2);
            gf2to192_matrix_set(matrix1, n_rows, i, j, entry1);
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
static inline void gf2to192_matrix_add_scalar_v2(gf2to192 *matrix1,
                                                const gf2to192 *matrix2,
                                                const gf2to192 scalar,
                                                const gf2to192 *matrix3,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    gf2to192 entry1, entry2;
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf2to192_matrix_get(entry1, matrix2, n_rows, i, j);
            gf2to192_matrix_get(entry2, matrix3, n_rows, i, j);
            gf2to192_mul(entry2, scalar, entry2);
            gf2to192_add(entry1, entry1, entry2);
            gf2to192_matrix_set(matrix1, n_rows, i, j, entry1);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar_gf2(gf2to192 *matrix1,
                                                  gf2to192 scalar,
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
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff
/// \param[in] matrix2 Matrix over ff_mu
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
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff
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
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
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
                gf2to192_add(entry_i_j,entry_i_j, tmp);
            }
            gf2to192_matrix_set(result, n_rows1, i, j, entry_i_j);
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
                gf2to192_add(entry_i_j,entry_i_j, tmp);
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
            **out = tmp;
            out += 1;
        }
    }
}

#ifdef USE_AVX2
// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2to192
/// \param[in] matrix3 Matrix over gf2to192
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline
void gf2to12_matrix_add_u256(gf2to192 *matrix1,
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
    if (n_rows % 8 == 0) {
        gf2to192_vector_add_gf2_u256_v2(matrix1, matrix3, matrix2, n_rows*n_cols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);
    __m256i buffer[8];
    gf2to192 *buffer2 = (gf2to192 *)buffer;
    for (uint32_t col = 0; col < n_cols; ++col) {
        uint32_t i = n_rows;

        const gf2 *in2 = matrix2 + col*gf2_col_bytes;
        const gf2to192 *in1 = matrix3 + col*n_rows;
        gf2to192 *out = matrix1 + col*n_rows;

        while (i >= 8u) {
            // NOTE: only correct if n_rows % 8 > 0
            gf2to192v_expand_gf2_x8_u256(buffer, *in2, 8);
            for (uint32_t t = 0; t < 8; t++) {
                const __m256i tmp = _mm256_loadu_si256((const __m256i *)(in1+t));
                buffer[t] = _mm256_xor_si256(buffer[t], tmp);
                _mm256_storeu_si256((__m256i *)(out + t), buffer[t]);
            }

            i   -= 8u;
            in2 += 1u;
            in1 += 8u;
            out += 8u;
        }

        if (i) {
            uint64_t tmp[4] __attribute__((aligned(32)))= {0};
            gf2to192v_expand_gf2_x8_u256(buffer, *in2, i);
            for (uint32_t t = 0; t < i; t++) {
                for (uint32_t tt = 0; tt < 3; tt++) { tmp[tt] = in1[t][tt]; }
                const __m256i tmp2 = _mm256_load_si256((const __m256i *)(tmp));
                buffer[t] = _mm256_xor_si256(buffer[t], tmp2);
                _mm256_store_si256((__m256i *)tmp, buffer[t]);
                for (uint32_t tt = 0; tt < 3; tt++) { out[t][tt] = tmp[tt]; }
            }
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf2to192
/// \param[in] matrix2 Matrix over gf2
/// \param[in] matrix3 Matrix over gf2to192
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
static void gf2to192_matrix_add_scalar_u256(gf2to192 *matrix1,
                                           const gf2to192 scalar,
                                           const gf2to192 *matrix2,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf2to192_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf2to12
/// \param[in] scalar scalar over gf2to12
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf2to192_matrix_add_scalar_gf2_u256(gf2to192 *matrix1,
                                                      const gf2to192 scalar,
                                                      const gf2 *matrix2,
                                                      const uint32_t n_rows,
                                                      const uint32_t n_cols) {
    if (n_rows % 8 == 0) {
        gf2to192_vector_scalar_add_gf2_u256_v2(matrix1, scalar, matrix2, n_rows*n_cols);
        return;
    }

    __m256i buffer[8];
    // TODO research: maybe its faster to replace all the {tmp[i] = XXX} code with this?
    const __m256i mask = _mm256_setr_epi64x(1, 1, 1, 0);
    const __m256i s = _mm256_maskload_epi64((long long *)scalar, mask);
    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(n_rows);

    for (uint32_t col = 0; col < n_cols; col++) {
        uint32_t i = 0;

        gf2to192 *out = matrix1 + col*n_rows;
        const gf2 *in = matrix2 + col*gf2_col_bytes;

        for (; (i+8) <= n_rows; i+=8) {
            // NOTE: only correct if n_rows % 8 > 0
            gf2to192v_expand_gf2_x8_u256(buffer, *in, 8);
            for (uint32_t t = 0; t < 8; t++) {
                const __m256i tmp = gf2to192v_mul_gf2_u256(buffer[t], s);
                _mm256_storeu_si256((__m256i *)(out + t), tmp);
            }

            out += 8;
            in += 1;
        }

        const uint32_t limit = n_rows % 8;
        if (limit) {
            uint64_t tmp[4] __attribute__((aligned(32)))= {0};
            gf2to192v_expand_gf2_x8_u256(buffer, *in, i);
            for (uint32_t t = 0; t < i; t++) {
                const __m256i tmp2 = gf2to192v_mul_gf2_u256(buffer[t], s);
                _mm256_store_si256((__m256i *)tmp, tmp2);
                for (uint32_t tt = 0; tt < 3; tt++) { out[t][tt] = tmp[tt]; }
            }
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
static inline
void gf2to192_matrix_mul_u256(gf2to192 *result,
                              const gf2to192 *matrix1,
                              const gf2to192 *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    // TODO, still need to know which size of the matrices
}
#endif
