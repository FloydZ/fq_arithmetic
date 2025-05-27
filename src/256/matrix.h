#pragma once

#include <stdint.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "../16/matrix.h"
#include "../2/matrix.h"

#define gf256_matrix_get_entry(m,n,i,j) m[j*n + i]
#define gf256_matrix_set_entry(m,n,i,j,v) m[j*n + i] = v
#define gf256_matrix_bytes_size(x, y) ((x) * (y) * sizeof(gf256))

/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline
gf256* gf256_matrix_alloc(const uint32_t n_rows,
                          const uint32_t n_cols) {
    return (gf256 *)calloc(sizeof(gf256), gf256_matrix_bytes_size(n_rows, n_cols));
}

/// \param matrix1[in]: pointer to the matrix
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline 
void gf256_matrix_print(const gf256 *matrix1,
                        const uint32_t n_rows,
                        const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf256 t = gf256_matrix_get_entry(matrix1, n_rows, i, j);
            printf("%3d ", t);
        }
        printf("\n");
    }
    printf("\n");
}

/// \param matrix1[in]: pointer to the matrix
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf256_matrix_random(gf256 *matrix,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < (n_rows * n_cols); i++) {
        matrix[i] = rand();
    }
}

/// \param matrix1[in]: pointer to the matrix
/// \param matrix2[in]: pointer to the matrix
/// \param n_rows[in]: number of rows
/// \param n_cols[in]: number of cols
static inline void gf256_matrix_copy(gf256 *matrix1,
                                     const gf256 *matrix2, 
                                     const uint32_t n_rows,
                                     const uint32_t n_cols) {
    memcpy(matrix1, matrix2, gf256_matrix_bytes_size(n_rows, n_cols));
}

/// \brief matrix1 = matrix2 + matrix3
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf256
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add(gf256 *matrix1, 
                                    const gf256 *matrix2,
                                    const gf256 *matrix3, 
                                    const uint32_t n_rows,
                                    const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows * n_cols; i++) {
        matrix1[i] = gf256_add(matrix2[i], matrix3[i]);
    }
}

/// \brief matrix1 = matrix2 + matrix3
///
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_gf2(gf256 *matrix1,
                                         const gf256 *matrix2,
                                         const gf2 *matrix3,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf256 entry2;

            const gf256 entry1 = gf256_matrix_get_entry(matrix2, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix3, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry2);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_gf16(gf256 *matrix1,
                                         const gf256 *matrix2,
                                         const ff_t *matrix3,
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf256 entry2;

            const gf256 entry1 = gf256_matrix_get_entry(matrix2, n_rows, i, j);
            entry2 = gf16_matrix_get(matrix3, n_rows, i, j);
            entry2 = gf256_expand_tab[entry2];
            entry2 = entry1 ^ entry2;
            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry2);
        }
    }
}

/**
 * \fn static inline void mirath_matrix_ff_mu_add_multiple_ff(ff_mu_t *matrix1, ff_mu_t scalar, const ff_t *matrix2,
 *                                       const uint32_t n_rows, const uint32_t n_cols)
 * \brief matrix1 += scalar * matrix2
 *
 * \param[out] matrix1 Matrix over ff_mu
 * \param[in] scalar scalar over ff_mu
 * \param[in] matrix2 Matrix over ff
 * \param[in] n_rows number of rows
 * \param[in] n_cols number of columns
 */
static inline void gf256_matrix_add_multiple_gf2(gf256 *matrix1,
                                                 const gf256 scalar, 
                                                 const gf2 *matrix2,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf256 entry2, entry3;

            const gf256 entry1 = gf256_matrix_get_entry(matrix1, n_rows, i, j);
            entry2 = gf2_matrix_get(matrix2, n_rows, i, j) & 0x01;
            entry3 = entry1 ^ gf256_mul(scalar, entry2);
            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix2 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_gf16(gf256 *matrix1,
                                                  const gf256 scalar,
                                                  const ff_t *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf256 entry2, entry3;

            const gf256 entry1 = gf256_matrix_get_entry(matrix1, n_rows, i, j);
            entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            entry2 = gf256_expand_tab[entry2];
            entry3 = entry1 ^ gf256_mul(scalar, entry2);
            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf256_matrix_add_multiple_2(gf256 *matrix1,
                                        const gf256 scalar,
                                        const gf256 *matrix2,
                                        const uint32_t n_rows,
                                        const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf256 entry1 = gf256_matrix_get_entry(matrix1, n_rows, i, j);
            const gf256 entry2 = gf256_matrix_get_entry(matrix2, n_rows, i, j);
            const gf256 entry3 = entry1 ^ gf256_mul(scalar, entry2);

            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry3);
        }
    }
}

///\brief matrix1 = matrix2 + scalar * matrix3
///
///\param[out] matrix1 Matrix over gf256
///\param[in] matrix2 Matrix over gf256
///\param[in] scalar scalar over gf256
///\param[in] matrix3 Matrix over gf256
///\param[in] n_rows number of rows
///\param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple(gf256 *matrix1,
                                             const gf256 *matrix2,
                                             const gf256 scalar,
                                             const gf256 *matrix3,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf256 entry1 = gf256_matrix_get_entry(matrix2, n_rows, i, j);
            const gf256 entry2 = gf256_matrix_get_entry(matrix3, n_rows, i, j);
            const gf256 entry3 = entry1 ^ gf256_mul(scalar, entry2);

            gf256_matrix_set_entry(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief result = matrix1 * matrix2
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf2_1(gf256 *result,
                                              const gf2 *matrix1,
                                              const gf256 *matrix2,
                                              const uint32_t n_rows1,
                                              const uint32_t n_cols1,
                                              const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k) & 0x01;
                entry_k_j = gf256_matrix_get_entry(matrix2, n_cols1, k, j);
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}


/// \brief result = matrix1 * matrix2
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf2_2(gf256 *result,
                                              const gf256 *matrix1,
                                              const gf2 *matrix2,
                                              const uint32_t n_rows1,
                                              const uint32_t n_cols1,
                                              const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;
    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf256_matrix_get_entry(matrix1, n_rows1, i, k);
                entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j) & 0x01;
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// 
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf16_1(gf256 *result,
                                               const ff_t *matrix1,
                                               const gf256 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf16_matrix_get(matrix1, n_rows1, i, k);
                entry_i_k = gf256_expand_tab[entry_i_k];
                entry_k_j = gf256_matrix_get_entry(matrix2, n_cols1, k, j);
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// 
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf16_2(gf256 *result,
                                               const gf256 *matrix1,
                                               const ff_t *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf256_matrix_get_entry(matrix1, n_rows1, i, k);
                entry_k_j = gf16_matrix_get(matrix2, n_cols1, k, j);
                entry_k_j = gf256_expand_tab[entry_k_j];
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// 
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product(gf256 *result,
                                        const gf256 *matrix1,
                                        const gf256 *matrix2,
                                        const uint32_t n_rows1,
                                        const uint32_t n_cols1,
                                        const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf256_matrix_get_entry(matrix1, n_rows1, i, k);
                entry_k_j = gf256_matrix_get_entry(matrix2, n_cols1, k, j);
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief result += matrix1 * matrix2
/// 
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_add_product(gf256 *result,
                                            const gf256 *matrix1,
                                            const gf256 *matrix2,
                                            const uint32_t n_rows1,
                                            const uint32_t n_cols1,
                                            const uint32_t n_cols2) {
    gf256 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = gf256_matrix_get_entry(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf256_matrix_get_entry(matrix1, n_rows1, i, k);
                entry_k_j = gf256_matrix_get_entry(matrix2, n_cols1, k, j);
                entry_i_j ^= gf256_mul(entry_i_k, entry_k_j);
            }

            gf256_matrix_set_entry(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief mapping from gf16 to gf256
/// 
/// \param[out] out Matrix over gf256
/// \param[in] input Matrix over gf16
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf256_matrix_map_gf16_to_gf256(gf256 *out,
                                                  const uint8_t *input,
                                                  const uint32_t nrows,
                                                  const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; ++i) {
        for (uint32_t j = 0; j < nrows; ++j) {
            const uint8_t tmp = gf16_matrix_get(input, nrows, j, i);
            *out = gf256_expand_tab[tmp];
            out += 1;
        }
    }
}

#ifdef USE_AVX2 

/// reads up to 16 bytes from a single columns
static inline __m128i read16column(const gf256 *col,
                                   const uint32_t bytes) {
    if (bytes == 32) { return _mm_loadu_si128((const __m128i *)col); }

    uint8_t tmp[16] __attribute__((aligned(16)));
    for (uint32_t i = 0; i < bytes; ++i) {
        tmp[i] = col[i];
    }

    return _mm_load_si128((const __m128i *)tmp);
}

/// reads up to 32 bytes from a single columns
static inline __m256i read32column(const gf256 *col,
                                   const uint32_t bytes) {
    if (bytes == 32) { return _mm256_loadu_si256((const __m256i *)col); }

    uint8_t tmp[32] __attribute__((aligned(32)));
    for (uint32_t i = 0; i < bytes; ++i) {
        tmp[i] = col[i];
    }

    return _mm256_load_si256((const __m256i *)tmp);
}

/// writes up to 32 bytes into a single colum
static inline void write32column(gf256 *col,
                                 const __m256i data,
                                 const uint32_t bytes) {
    uint8_t tmp[32] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, data);
    for (uint32_t i = 0; i < bytes; ++i) {
        col[i] = tmp[i];
    }
}


/// \brief mapping from gf16 to gf256
/// 
/// \param[out] out Matrix over gf256
/// \param[in] input Matrix over gf16
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf256_matrix_map_gf16_to_gf256_u256(gf256 *out,
                                                       const uint8_t *input,
                                                       const uint32_t nrows,
                                                       const uint32_t ncols) {
    if ((nrows & 1u) == 0) {
        gf256_vector_set_to_gf16_u256(out, input, nrows * ncols);
        return;
    }

    for (uint32_t j = 0; j < ncols; j++) {
        gf16  *i = (gf16  *)(input + j*((nrows + 1)/2));
        gf256 *o = (gf256 *)(out + j*nrows);
        gf256_vector_set_to_gf16_u256(o, i, nrows);
    }
}


/// NOTE: assumes no stride
/// \brief matrix1 = matrix2 + matrix3
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf256
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_u256(gf256 *matrix1, 
                                         const gf256 *matrix2,
                                         const gf256 *matrix3, 
                                         const uint32_t n_rows,
                                         const uint32_t n_cols) {
    gf256_vector_add_u256(matrix1, matrix2, matrix3, n_rows*n_cols);
}

/// \brief matrix1 = matrix2 + matrix3
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_gf16_u256(gf256 *matrix1,
                                              const gf256 *matrix2,
                                              const gf16 *matrix3,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m128i mask  = _mm_set1_epi8(0x0F);

    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = n_rows;
        gf16 *in   = (gf16  *)matrix3 + j*((n_rows + 1)/2);
        gf256 *in2 = (gf256 *)matrix2 + j*n_rows;
        gf256 *out = matrix1 + j*n_rows;
        while (i >= 32u) {
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)in2);

            // faster
            const __m128i load = _mm_loadu_si128((__m128i *)in);
            const __m128i ll = _mm_shuffle_epi8(perm128, load & mask);
            const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask);
            const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
            const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
            const __m256i t4 = _mm256_unpacklo_epi8(tl, th);

            const __m256i t5 = t4 ^ m2;
            _mm256_storeu_si256((__m256i *)out, t5);
            in  += 16u;
            out += 32u;
            in2 += 32u;
            i   -= 32u;
        }

        while (i >= 16u) {
            const __m128i m2 = _mm_loadu_si128((const __m128i *)in2);
            const uint32_t t11 = *(uint32_t *)(in +  0);
            const uint32_t t12 = *(uint32_t *)(in +  4);

            const __m64 t21 = (__m64)_pdep_u64(t11, 0x0f0f0f0f0f0f0f0f);
            const __m64 t22 = (__m64)_pdep_u64(t12, 0x0f0f0f0f0f0f0f0f);

            const __m128i t3 = _mm_setr_epi64(t21, t22);
            const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
            const __m128i t5 = t4 ^ m2;
            _mm_storeu_si128((__m128i *)out, t5);
            in  += 8u;
            out += 16u;
            in2 += 16u;
            i   -= 16u;
        }

        if (i) {
            uint8_t tmp[16] __attribute__((aligned(32)));
            for (uint32_t k = 0; k < i; k++) { tmp[k] = in2[k]; }
            const __m128i m2 = _mm_load_si128((const __m128i *) tmp);

            for (uint32_t k = 0; k < (i + 1) / 2; k++) { tmp[k] = in[k]; }
            const __m64 t21 = (__m64) _pdep_u64(*(uint32_t *) (tmp + 0), 0x0f0f0f0f0f0f0f0f);
            const __m64 t22 = (__m64) _pdep_u64(*(uint32_t *) (tmp + 4), 0x0f0f0f0f0f0f0f0f);

            const __m128i t3 = _mm_setr_epi64(t21, t22);
            const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
            const __m128i t5 = t4 ^ m2;
            _mm_store_si128((__m128i *) tmp, t5);
            for (uint32_t k = 0; k < i; k++) { out[k] = tmp[k]; }
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
///
/// \param[out] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_gf2_u256(gf256 *matrix1,
                                              const gf256 *matrix2,
                                              const gf16 *matrix3,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    const uint32_t bytes_per_column = gf2_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = n_rows;
        gf16 *in   = (gf16  *)matrix3 + j*bytes_per_column;
        gf256 *in2 = (gf256 *)matrix2 + j*n_rows;
        gf256 *out = matrix1 + j*n_rows;
        while (i >= 32u) {
            const __m256i m1 = gf256v_expand_gf2_x32_u256(in);
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)in2);
            const __m256i t1 = m1 ^ m2;
            _mm256_storeu_si256((__m256i *)out, t1);
            in  += 4u;
            in2 += 32u;
            out += 32u;
            i   -= 32u;
        }

        while (i >= 16u) {
            const __m128i m1 = gf256_expand_gf2_x16_u256(in);
            const __m128i m2 = _mm_loadu_si128((const __m128i *)in2);
            const __m128i t1 = m1 ^ m2;
            _mm_storeu_si128((__m128i *)out, t1);
            in  += 2u;
            out += 16u;
            in2 += 16u;
            i   -= 16u;
        }

        while (i >= 8u) {
            const uint64_t m1 = gf2to12v_expand_gf2_x8_u256(in);
            const uint64_t m2 = *(uint64_t *)(in2);
            const uint64_t t1 = m1 ^ m2;
            *(uint64_t *)out = t1;
            in  += 1u;
            out += 8u;
            in2 += 8u;
            i   -= 8u;
        }

        if (i) {
            uint8_t tmp[8] __attribute__((aligned(32)));
            const uint64_t m1 = gf2to12v_expand_gf2_x8_u256(in);

            for (uint32_t k = 0; k < i; k++) { tmp[k] = in2[k]; }
            const uint64_t m2 = *(uint64_t *)(tmp);
            const uint64_t t1 = m1 ^ m2;
            *(uint64_t *)tmp = t1;
            for (uint32_t k = 0; k < i; k++) { out[k] = tmp[k]; }
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_gf2_u256(gf256 *matrix1,
                                                       const gf256 scalar,
                                                       const gf2 *matrix2,
                                                       const uint32_t nrows,
                                                       const uint32_t ncols) {
    if ((nrows % 8 == 0) || (ncols == 1)) {
        gf256_vector_add_scalar_gf2_u256(matrix1, scalar, matrix2, nrows*ncols);
        return;
    }
   
    const __m128i s128 = _mm_set1_epi8(scalar);
    if (nrows == 4) {
        const uint64_t m = 0x01010101;
        const uint32_t limit = ncols % 4;

        const gf2 *mm2 = matrix2;
        gf256 *out = matrix1;

        /// NOTE: asumes n_cols %2 == 2
        uint32_t col = 0;
        for (; (col+4) <= ncols; col+=4) {
            const uint32_t a = *(uint32_t *)(matrix2 + col);

            const uint64_t t41 = _pdep_u64(a>> 0u, m);
            const uint64_t t42 = _pdep_u64(a>> 8u, m);
            const uint64_t t43 = _pdep_u64(a>>16u, m);
            const uint64_t t44 = _pdep_u64(a>>24u, m);
            const uint64_t t31 =  t41 ^ (t42 << 32);
            const uint64_t t32 =  t43 ^ (t44 << 32);
            const __m128i t  = _mm_set_epi64x(t32, t31);

            const __m128i m1 = _mm_loadu_si128((__m128i *)out);
            const __m128i c1 = gf256v_mul_gf2_u128(s128, t);
            _mm_storeu_si128((__m128i *)(out +  0), m1 ^ c1);

            out += 4*nrows;
            mm2 += 4;
        }

        if (limit) {
            gf256 tmp[16] __attribute__((aligned(32)));

            const uint64_t t41 = _pdep_u64(mm2[0], m);
            const uint64_t t42 = _pdep_u64(mm2[1], m);
            const uint64_t t31 = t41 ^ (t42 << 32);
            const __m128i t  = _mm_set_epi64x(0, t31);
            const __m128i c1 = gf256v_mul_gf2_u128(s128, t);
            _mm_store_si128((__m128i *)tmp, c1);

            for (uint32_t i = 0; i < 2*nrows; i++) {
                out[i] ^= tmp[i];
            }
        }
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t col = 0; col < ncols; ++col) {
        gf256 *o      = matrix1 + col*nrows;
        const gf2 *in = matrix2 + col*gf2_col_bytes;

        gf256_vector_add_scalar_gf2_u256(o, scalar, in, nrows);
    }
}

/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_gf16_u256(gf256 *matrix1,
                                                       const gf256 scalar,
                                                       const gf16 *matrix3,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {

    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    const __m128i ml128 = _mm256_extracti128_si256(ml, 0);
    const __m128i mh128 = _mm256_extracti128_si256(mh, 0);
    const __m128i mask128 = _mm_set1_epi8(0xF);

    const uint32_t limit = n_rows % 16;

    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = n_rows;
        gf16 *in   = (gf16  *)matrix3 + j*((n_rows + 1)/2);
        gf256 *out = matrix1 + j*n_rows;
        // while (i >= 32u) {
        //     const __m256i m2 = _mm256_loadu_si256((const __m256i *)in2);

        //     // slower
        //     // const uint32_t t11 = *(uint32_t *)(in +  0);
        //     // const uint32_t t12 = *(uint32_t *)(in +  4);
        //     // const uint32_t t13 = *(uint32_t *)(in +  8);
        //     // const uint32_t t14 = *(uint32_t *)(in + 12);

        //     // const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
        //     // const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
        //     // const uint64_t t23 = _pdep_u64(t13, 0x0F0F0F0F0F0F0F0F);
        //     // const uint64_t t24 = _pdep_u64(t14, 0x0F0F0F0F0F0F0F0F);

        //     // const __m256i t3 = _mm256_setr_epi64x(t21, t22, t23, t24);
        //     // const __m256i t4 = _mm256_shuffle_epi8(perm256, t3);

        //     // faster
        //     const __m128i load = _mm_loadu_si128((__m128i *)in);
        //     const __m128i ll = _mm_shuffle_epi8(perm128, load & mask128);
        //     const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask128);
        //     const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
        //     const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
        //     const __m256i t4 = _mm256_unpacklo_epi8(tl, th);
        //     const __m256i t5 = gf256_linear_transform_8x8_256b(ml, mh, t4, mask);
        //     const __m256i t6 = t5 ^ m2;
        //     _mm256_storeu_si256((__m256i *)out, t6);
        //     in  += 16u;
        //     out += 32u;
        //     i   -= 32u;
        // }

        while (i >= 16u) {
            const __m128i m2 = _mm_loadu_si128((const __m128i *)out);
            const uint32_t t11 = *(uint32_t *)(in +  0);
            const uint32_t t12 = *(uint32_t *)(in +  4);

            const __m64 t21 = (__m64)_pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m64 t22 = (__m64)_pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);

            const __m128i t3 = _mm_setr_epi64(t21, t22);
            const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
            const __m128i t5 = gf256_linear_transform_8x8_128b(ml128, mh128, t4, mask128);
            const __m128i t6 = t5 ^ m2;
            _mm_storeu_si128((__m128i *)out, t6);
            in  += 8u;
            out += 16u;
            i   -= 16u;
        }

        if (limit) {
            uint8_t tmp1[16] __attribute__((aligned(32)));
            uint8_t tmp2[16] __attribute__((aligned(32)));
            for (uint32_t k = 0; k < limit; k++) { tmp1[k] = out[k]; }
            const __m128i m2 = _mm_load_si128((const __m128i *) tmp1);

            for (uint32_t k = 0; k < (limit + 1) / 2; k++) { tmp2[k] = in[k]; }
            const __m64 t21 = (__m64) _pdep_u64(*(uint32_t *) (tmp2 + 0), 0x0F0F0F0F0F0F0F0F);
            const __m64 t22 = (__m64) _pdep_u64(*(uint32_t *) (tmp2 + 4), 0x0F0F0F0F0F0F0F0F);

            const __m128i t3 = _mm_setr_epi64(t21, t22);
            const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
            const __m128i t5 = gf256_linear_transform_8x8_128b(ml128, mh128, t4, mask128);
            const __m128i t6 = t5 ^ m2;
            _mm_store_si128((__m128i *) tmp1, t6);
            for (uint32_t k = 0; k < limit; k++) { out[k] = tmp2[k]; }
        }
    }
}


/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf256_matrix_add_multiple_2_u256(gf256 *matrix1,
                                             const gf256 scalar,
                                             const gf256 *matrix2,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    gf256_vector_add_scalar_2_u256(matrix1, scalar, matrix2, n_rows * n_cols);
}


///\brief matrix1 = matrix2 + scalar * matrix3
///
///\param[out] matrix1 Matrix over gf256
///\param[in] matrix2 Matrix over gf256
///\param[in] scalar scalar over gf256
///\param[in] matrix3 Matrix over gf256
///\param[in] n_rows number of rows
///\param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_u256(gf256 *matrix1,
                                                    const gf256 *matrix2,
                                                    const gf256 scalar,
                                                    const gf256 *matrix3,
                                                    const uint32_t n_rows,
                                                    const uint32_t n_cols) {

    gf256_vector_add_scalar_3_u256(matrix1, matrix2, scalar, matrix3, n_cols*n_rows);
}

/// \brief result = matrix1 * matrix2
/// `n_cols2` == 1
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
static inline void gf256_matrix_product_gf2_1_slim_u256(gf256 *result,
                                                        const gf2 *matrix1,
                                                        const gf256 *matrix2,
                                                        const uint32_t n_rows1,
                                                        const uint32_t n_cols1) {
    uint8_t tmp[32] __attribute__((aligned(32)));

    const uint32_t limit = n_rows1 % 32;
    const uint32_t bytes_per_col = gf2_matrix_bytes_per_column(n_rows1);

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const uint8_t *m1 = matrix1 + col * bytes_per_col;
        const __m256i b = _mm256_set1_epi8(*(matrix2 + col));
        gf256 *r = result;
        while ((i + 32) <= n_rows1) {
            const __m256i a = gf256v_expand_gf2_x32_u256(m1);
            __m256i t = gf256v_mul_gf2_u256(b, a);
            t ^= _mm256_loadu_si256((__m256i *)(r));
            _mm256_storeu_si256((__m256i *)(r), t);

            m1 += 4;
            r  += 32;
            i  += 32;
        }

        if (limit) {
            for (uint32_t j = 0; j < (limit+7)/8; ++j) { tmp[j] = m1[j]; }
            const __m256i a = gf256v_expand_gf2_x32_u256(m1);
            __m256i t = gf256v_mul_gf2_u256(b, a);
            _mm256_store_si256((__m256i *)tmp, t);
            for (uint32_t j = 0; j < limit; ++j) { r[j] ^= tmp[j]; }
        }
    }
}

/// \brief result = matrix1 * matrix2
/// `n_cols2` == 1
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf2
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
static inline void gf256_matrix_product_gf2_2_slim_u256(gf256 *result,
                                                         const gf256 *matrix1,
                                                         const gf2 *matrix2,
                                                         const uint32_t n_rows1,
                                                         const uint32_t n_cols1) {
    uint16_t tmp[16] __attribute__((aligned(32)));

    const uint32_t limit = n_rows1 % 32;
    const uint32_t bytes_per_col = gf2_matrix_bytes_per_column(n_rows1);

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const uint8_t *m1 = matrix1 + col * bytes_per_col;
        const __m256i b = _mm256_set1_epi16(*(matrix2 + col));
        gf256 *r = result;
        while ((i + 32) <= n_rows1) {
            const __m256i a = gf256_vector_extend_gf16_x32(m1);
            __m256i t = gf256v_mul_gf2_u256(a, b);
            t ^= _mm256_loadu_si256((__m256i *)(r));
            _mm256_storeu_si256((__m256i *)(r), t);

            m1 += 4;
            r  += 32;
            i  += 32;
        }

        if (limit) {

        }
    }
}

/// \brief result = matrix1 * matrix2
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf2_1_u256(gf256 *result,
                                                    const gf2 *matrix1,
                                                    const gf256 *matrix2,
                                                    const uint32_t n_rows1,
                                                    const uint32_t n_cols1,
                                                    const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        gf256_matrix_product_gf2_1_slim_u256(result, matrix1, matrix2, n_rows1, n_cols1);
        return;
    }

    gf256_matrix_product_gf2_1(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

/// \brief result = matrix1 * matrix2
///
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf2_2_u256(gf256 *result,
                                                    const gf256 *matrix1,
                                                    const gf2 *matrix2,
                                                    const uint32_t n_rows1,
                                                    const uint32_t n_cols1,
                                                    const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        gf256_matrix_product_gf2_2_slim_u256(result, matrix1, matrix2, n_rows1, n_cols1);
        return;
    }
}

/// \brief result = matrix1 * matrix2
/// `n_cols2` == 1
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
static inline void gf256_matrix_product_gf16_1_vector_u256(gf256 *result,
                                                        const gf16 *matrix1,
                                                        const gf256 *matrix2,
                                                        const uint32_t n_rows1,
                                                        const uint32_t n_cols1) {
    uint8_t tmp[32] __attribute__((aligned(32)));

    const uint32_t limit = n_rows1 % 32;
    const uint32_t bytes_per_col = gf16_matrix_bytes_per_column(n_rows1);

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const uint8_t *m1 = matrix1 + col * bytes_per_col;
        const __m256i b = _mm256_set1_epi8(*(matrix2 + col));
        gf256 *r = result;

        while ((i + 32) <= n_rows1) {
            const __m256i a = gf256_vector_extend_gf16_x32(m1);
            __m256i t = gf256v_mul_u256(b, a);
            t ^= _mm256_loadu_si256((__m256i *)(r));
            _mm256_storeu_si256((__m256i *)(r), t);

            m1 += 16;
            r  += 32;
            i  += 32;
        }

        if (limit) {
            for (uint32_t j = 0; j < (limit+1)/2; ++j) { tmp[j] = m1[j]; }
            const __m256i a = gf256_vector_extend_gf16_x32(tmp);
            __m256i t = gf256v_mul_u256(b, a);
            _mm256_store_si256((__m256i *)tmp, t);
            for (uint32_t j = 0; j < limit; ++j) { r[j] ^= tmp[j]; }
        }
    }
}


/// \brief result = matrix1 * matrix2
/// 
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf16_1_u256(gf256 *result,
                                                    const gf16 *matrix1,
                                                    const gf256 *matrix2,
                                                    const uint32_t n_rows1,
                                                    const uint32_t n_cols1,
                                                    const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        return gf256_matrix_product_gf16_1_vector_u256(result, matrix1, matrix2, n_rows1, n_cols1);
    }

    const uint32_t nr256= n_rows1 >> 5u;
    const uint32_t tb   = n_rows1 - (nr256 * 32);
    const uint8_t tail  = (nr256 == 0) || (tb != n_rows1);
    const __m256i mask2 = _mm256_set1_epi8(0x0F);

    // NOTE: set this to the maximal number of gf16 elements you want to support divided by 32
    __m256i A[3];
    uint8_t tmp[32] __attribute__((aligned(32)));

    // for each row in matrix2
    for (uint32_t j = 0; j < n_cols1; ++j) {
        // load A
        for (uint32_t i = 0; i < nr256; i++) {
            A[i] = gf256_vector_extend_gf16_x32(matrix1 + j*((n_rows1+1)/2) + i*16);
        }

        if (tail) {
            for (uint32_t k = 0; k < (tb+1)/2; ++k) {
                tmp[k] = matrix1[j*((n_rows1+1)/2) + nr256*16 + k];
            }
            A[nr256] = gf256_vector_extend_gf16_x32(tmp);
        }

        // iterate over one row in matrix2
        for (uint32_t i = 0; i < n_cols2; ++i) {
            const gf256 b = *(matrix2 + i*n_cols1 + j);
            const __m256i m_tab = gf256v_generate_multab_16_single_element_u256(b);
            const __m256i ml = _mm256_permute2x128_si256(m_tab, m_tab, 0);
            const __m256i mh = _mm256_permute2x128_si256(m_tab, m_tab, 0x11);

            for (uint32_t s = 0; s < nr256; ++s) {
                __m256i r = gf256_linear_transform_8x8_256b(ml, mh, A[s], mask2);
                r ^= _mm256_loadu_si256((__m256i *)(result + i*n_rows1 + s*32));
                _mm256_storeu_si256((__m256i *)(result + i*n_rows1 + s*32), r);
            }

            // tail mngt
            if (tail) {
                const __m256i r = gf256_linear_transform_8x8_256b(ml, mh, A[nr256], mask2);
                _mm256_store_si256((__m256i *)tmp, r);
                for (uint32_t k = 0; k < tb; ++k) {
                    result[i*n_rows1 + nr256*32 + k] ^= tmp[k];
                }
            }
        }
    }
}

/// \brief result = matrix1 * matrix2
///
/// \param[out] result Matrix over ff_mu
/// \param[in] matrix1 Matrix over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_gf16_2_u256(gf256 *result,
                                                    const gf256 *matrix1,
                                                    const gf16 *matrix2,
                                                    const uint32_t n_rows1,
                                                    const uint32_t n_cols1,
                                                    const uint32_t n_cols2) {

    const uint32_t nr256= n_rows1 >> 5u;
    const uint32_t tb   = n_rows1 - (nr256 * 32);
    const uint8_t tail  = (nr256 == 0) || (tb != n_rows1);
    const __m256i mask2 = _mm256_set1_epi8(0x0F);

    // NOTE: set this to the maximal number of gf256 elements you want to support divided by 32
    __m256i A[2];
    uint8_t tmp[32] __attribute__((aligned(32)));

    // for each row in matrix2
    for (uint32_t j = 0; j < n_cols1; ++j) {
        for (uint32_t i = 0; i < nr256; i++) {
            A[i] = _mm256_loadu_si256((const __m256i *)(matrix1 + j*n_rows1 + i*32));
        }

        if (tail) {
            for (uint32_t k = 0; k < tb; ++k) {
                tmp[k] = matrix1[j*n_rows1 + nr256*32 + k];
            }
            A[nr256] = _mm256_load_si256((const __m256i *)tmp);
        }

        // iterate over one row in matrix2
        for (uint32_t i = 0; i < n_cols2; ++i) {
            const gf256 b1 = gf16_matrix_get(matrix2, n_cols1, j, i);
            const gf256 b2 = gf256_expand_tab[b1];
            const __m256i m_tab = gf256v_generate_multab_16_single_element_u256(b2);
            const __m256i ml = _mm256_permute2x128_si256(m_tab, m_tab, 0);
            const __m256i mh = _mm256_permute2x128_si256(m_tab, m_tab, 0x11);

            for (uint32_t s = 0; s < nr256; ++s) {
                __m256i r = gf256_linear_transform_8x8_256b(ml, mh, A[s], mask2);
                r ^= _mm256_loadu_si256((__m256i *)(result + i*n_rows1 + s*32));
                _mm256_storeu_si256((__m256i *)(result + i*n_rows1 + s*32), r);
            }

            // tail mngt
            if (tail) {
                const __m256i r = gf256_linear_transform_8x8_256b(ml, mh, A[nr256], mask2);
                _mm256_store_si256((__m256i *)tmp, r);
                for (uint32_t k = 0; k < tb; ++k) {
                    result[i*n_rows1 + nr256*32 + k] ^= tmp[k];
                }
            }
        }
    }
}

/// special matrix multiplication
///
/// \param[out] C Matrix over gf256 8 by c matrix
/// \param[in]  A Matrix over gf256 8 by 8 matrix
/// \param[in]  B Matrix over gf256 8 by c matrix
/// \param[in] n_colsB number of columns in matrix2
static inline void gf256_matrix_product_8x8xC_u256(gf256 *C,
                                                   const gf256 *A,
                                                   const gf256 *B,
                                                   const uint32_t c) {
    // A memory layout
    //     col0  col7
    // row0  | A11
    // row4  | A21

    // some variables
    uint32_t *c32 = (uint32_t *)C;
    uint64_t *b64 = (uint64_t *)B;
    __m256i acc;
    const __m256i vindex = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);

    // load A into registers
    const __m256i A11 = _mm256_i32gather_epi32((int *)(A + 0), vindex, 1u);
    const __m256i A21 = _mm256_i32gather_epi32((int *)(A + 4), vindex, 1u);

    // for each col in B
    for (uint32_t col = 0; col < c; ++col) {
        const __m128i tb1 = _mm_set1_epi64x(b64[col]);
        const __m256i B1 = _mm256_cvtepu8_epi32(tb1);

        // this computes 4 rows at once
        acc = gf256v_mul_u32_u256(A11, B1);
        c32[0] = gf256v_hadd_u32_u256(acc);

        acc = gf256v_mul_u32_u256(A21, B1);
        c32[1] = gf256v_hadd_u32_u256(acc);
        c32 += 2;
    }
}

/// special matrix multiplication
///
/// \param[out] C Matrix over gf256 16 by c matrix
/// \param[in]  A Matrix over gf256 16 by 16 matrix
/// \param[in]  B Matrix over gf256 16 by c matrix
/// \param[in] n_colsB number of columns in matrix2
static inline void gf256_matrix_product_16x16xC_u256(gf256 *C,
                                                     const gf256 *A,
                                                     const gf256 *B,
                                                     const uint32_t c) {
    // A memory layout
    //     col0  col8  col15
    // row0  | A11  A12
    // row4  | A21  A22
    // row8  | A31  A32
    // row12 | A41  A42

    // some variables
    uint32_t *c32 = (uint32_t *)C;
    const uint64_t *b64 = (uint64_t *)B;
    __m256i acc;
    const __m256i vindex = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

    // load A into registers
    const __m256i A11 = _mm256_i32gather_epi32((int *)(A +  0 +   0), vindex, 1u);
    const __m256i A21 = _mm256_i32gather_epi32((int *)(A +  4 +   0), vindex, 1u);
    const __m256i A31 = _mm256_i32gather_epi32((int *)(A +  8 +   0), vindex, 1u);
    const __m256i A41 = _mm256_i32gather_epi32((int *)(A + 12 +   0), vindex, 1u);

    const __m256i A12 = _mm256_i32gather_epi32((int *)(A +  0 + 128), vindex, 1u);
    const __m256i A22 = _mm256_i32gather_epi32((int *)(A +  4 + 128), vindex, 1u);
    const __m256i A32 = _mm256_i32gather_epi32((int *)(A +  8 + 128), vindex, 1u);
    const __m256i A42 = _mm256_i32gather_epi32((int *)(A + 12 + 128), vindex, 1u);

    // for each col in B
    for (uint32_t col = 0; col < c; ++col) {
        // TODO, not really optimial
        const __m128i tb1 = _mm_set1_epi64x(b64[col*2 + 0]);
        const __m128i tb2 = _mm_set1_epi64x(b64[col*2 + 1]);
        const __m256i B1 = _mm256_cvtepu8_epi32(tb1);
        const __m256i B2 = _mm256_cvtepu8_epi32(tb2);

        // this computes 4 rows at once
        acc  = gf256v_mul_u32_u256(A11, B1);
        acc ^= gf256v_mul_u32_u256(A12, B2);
        c32[0] = gf256v_hadd_u32_u256(acc);

        acc  = gf256v_mul_u32_u256(A21, B1);
        acc ^= gf256v_mul_u32_u256(A22, B2);
        c32[1] = gf256v_hadd_u32_u256(acc);

        acc  = gf256v_mul_u32_u256(A31, B1);
        acc ^= gf256v_mul_u32_u256(A32, B2);
        c32[2] = gf256v_hadd_u32_u256(acc);

        acc  = gf256v_mul_u32_u256(A41, B1);
        acc ^= gf256v_mul_u32_u256(A42, B2);
        c32[3] = gf256v_hadd_u32_u256(acc);

        c32 += 4;
    }
}


/// special matrix multiplication
/// B <= 16
/// C <= 16
/// \param[out] C Matrix over gf256 a by c matrix
/// \param[in]  A Matrix over gf256 a by b matrix
/// \param[in]  B Matrix over gf256 b by c matrix
/// \param[in] n_colsB number of columns in matrix2
static inline void gf256_matrix_product_le32xBxle16_u256(gf256 *C,
                                                       const gf256 *A,
                                                       const gf256 *B,
                                                       const uint32_t a,
                                                       const uint32_t b,
                                                       const uint32_t c) {
    const __m256i m = _mm256_set1_epi8(0x0F);
    __m256i tmp_C[16] = {0};

    // for each col in A
    for (uint32_t i = 0; i < b; ++i) {
        // load full col in A
        const __m256i col_A = read32column(A + i*a, a);

        // for each element in row i in B
        for (uint32_t j = 0; j < c; ++j) {
            // load element in B
            const gf256 elm_B = B[j*b + i];
            const __m256i m_tab = gf256v_generate_multab_16_single_element_u256(elm_B);
            const __m256i ml = _mm256_permute2x128_si256(m_tab, m_tab, 0);
            const __m256i mh = _mm256_permute2x128_si256(m_tab, m_tab, 0x11);

            __m256i r = gf256_linear_transform_8x8_256b(ml, mh, col_A, m);
            tmp_C[j] ^= r;
        }
    }

    for (uint32_t i = 0; i < c; ++i) {
        write32column(C +i*a, tmp_C[i], a);
    }
}

// TODO
/// generic matrix multiplication
/// \param C output
/// \param A input
/// \param B input
/// \param nr_rows_A <= 16
/// \param nr_cols_A <= 16
/// \param nr_cols_B can be arbitrary
//static inline void __ext_matrix_product_le16xle16xt(gf256 *C, const gf256 *A, const gf256 *B,
//                                                    const uint32_t nr_rows_A,
//                                                    const uint32_t nr_cols_A,
//                                                    const uint32_t nr_cols_B) {
//    // A memory layout
//    //     col0  col8  col15
//    // row0  | A11  A12
//    // row4  | A21  A22  if rows(A) >= 4
//    // row8  | A31  A32  if rows(A) >= 8
//    // row12 | A41  A42  if rows(A) >= 12
//    __m256i A11,A12,A21,A22,A31,A32,A41,A42;
//
//    // some variables
//    __m256i acc;
//    const __m256i vindex = _mm256_setr_epi32(0, nr_rows_A, 2*nr_rows_A, 3*nr_rows_A, 4*nr_rows_A, 5*nr_rows_A, 6*nr_rows_A, 7*nr_rows_A);
//
//    // load A into registers
//    A11 = _mm256_i32gather_epi32((int *)(A +  0 + 0), vindex, 1u);
//    if (nr_rows_A >  4) A21 = _mm256_i32gather_epi32((int *)(A +  4 + 0), vindex, 1u);
//    if (nr_rows_A >  8) A31 = _mm256_i32gather_epi32((int *)(A +  8 + 0), vindex, 1u);
//    if (nr_rows_A > 12) A41 = _mm256_i32gather_epi32((int *)(A + 12 + 0), vindex, 1u);
//
//    if (nr_cols_A > 8u) {
//        A12 = _mm256_i32gather_epi32((int *) (A +  0 + 8 * nr_rows_A), vindex, 1u);
//        if (nr_rows_A >  4) A22 = _mm256_i32gather_epi32((int *) (A +  4 + 8 * nr_rows_A), vindex, 1u);
//        if (nr_rows_A >  8) A32 = _mm256_i32gather_epi32((int *) (A +  8 + 8 * nr_rows_A), vindex, 1u);
//        if (nr_rows_A > 12) A42 = _mm256_i32gather_epi32((int *) (A + 12 + 8 * nr_rows_A), vindex, 1u);
//    }
//
//    // intermediate values for the B matrix
//    __m256i B1, B2;
//    const uint32_t bctr1 = MIN(8, nr_cols_A);
//    const uint32_t bctr2 = MIN(8, nr_cols_A-8);
//
//    // intermediate values for the C matrix
//    const uint32_t cctr1 = MIN(4, nr_rows_A -  0);
//    const uint32_t cctr2 = MIN(4, nr_rows_A -  4);
//    const uint32_t cctr3 = MIN(4, nr_rows_A -  8);
//    const uint32_t cctr4 = MIN(4, nr_rows_A - 12);
//
//    // for each col in B
//    for (uint32_t col = 0; col < nr_cols_B; ++col) {
//        B1 = _mm256_cvtepu8_epi32(read8column(B, bctr1));
//        B += bctr1;
//
//        if (nr_cols_A > 8) {
//            B2 = _mm256_cvtepu8_epi32(read8column(B, bctr2));
//            B += bctr2;
//        }
//
//        // this computes 4 rows at once
//        acc = gf256_mul_simd(A11, B1);
//        if (nr_cols_A > 8) { acc ^= gf256_mul_simd(A12, B2); }
//        write4column(C, _mm256_extract_epi32(gf256_hadd_avx2_32(acc), 0), cctr1);
//        C += cctr1;
//
//        if (nr_rows_A > 4) {
//            acc = gf256_mul_simd(A21, B1);
//            if (nr_cols_A > 8) { acc ^= gf256_mul_simd(A22, B2); }
//            write4column(C, _mm256_extract_epi32(gf256_hadd_avx2_32(acc), 0), cctr2);
//            C += cctr2;
//        }
//
//        if (nr_rows_A > 8) {
//            acc = gf256_mul_simd(A31, B1);
//            if (nr_cols_A > 8) { acc ^= gf256_mul_simd(A32, B2); }
//            write4column(C, _mm256_extract_epi32(gf256_hadd_avx2_32(acc), 0), cctr3);
//            C += cctr3;
//        }
//
//        if (nr_rows_A > 12) {
//            acc = gf256_mul_simd(A41, B1);
//            if (nr_cols_A > 8) { acc ^= gf256_mul_simd(A42, B2); }
//            write4column(C, _mm256_extract_epi32(gf256_hadd_avx2_32(acc), 0), cctr4);
//            C += cctr4;
//        }
//    }
//}


/// special matrix multiplication
/// B <= 8
/// 32 < a <= 64
/// \param[out] C Matrix over gf256 a by c matrix
/// \param[in]  A Matrix over gf256 a by b matrix
/// \param[in]  B Matrix over gf256 b by c matrix
/// \param[in] n_colsB number of columns in matrix2
static inline void gf256_matrix_product_le64xBxle48_u256(gf256 *C,
                                                         const gf256 *A,
                                                         const gf256 *B,
                                                         const uint32_t a,
                                                         const uint32_t b,
                                                         const uint32_t c) {
    const __m256i m = _mm256_set1_epi8(0x0F);
    const uint32_t la = a % 32;

    // for each col in A
    for (uint32_t i = 0; i < b; ++i) {
        // load full col in A
        const __m256i col_A1 = _mm256_loadu_si256((const __m256i *)(A + i*a));
        const __m256i col_A2 = read32column(A + i*a + 32, la);

        // for each element in row i in B
        for (uint32_t j = 0; j < c; ++j) {
            // load element in B
            const gf256 elm_B = B[j*b + i];
            const __m256i m_tab = gf256v_generate_multab_16_single_element_u256(elm_B);
            const __m256i ml = _mm256_permute2x128_si256(m_tab, m_tab, 0);
            const __m256i mh = _mm256_permute2x128_si256(m_tab, m_tab, 0x11);

            const __m256i r1 = gf256_linear_transform_8x8_256b(ml, mh, col_A1, m);
            const __m256i r2 = gf256_linear_transform_8x8_256b(ml, mh, col_A2, m);


            const __m256i c1 = _mm256_loadu_si256((const __m256i *)(C + j*a));
            const __m256i c2 = read32column(C + j*a + 32, la);
            _mm256_storeu_si256((__m256i *)(C + j*a), c1^r1);
            write32column(C + j*a + 32, c2^r2, la);
        }
    }
}



/// \brief result = matrix1 * matrix2
/// `n_cols2` == 1
///
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf2
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
static inline void gf256_matrix_product_slim_u256(gf256 *result,
                                                  const gf256 *matrix1,
                                                  const gf256 *matrix2,
                                                  const uint32_t n_rows1,
                                                  const uint32_t n_cols1) {
    uint8_t tmp[32] __attribute__((aligned(32)));

    const uint32_t limit = n_rows1 % 32;

    for (uint32_t col = 0; col < n_cols1; ++col) {
        uint32_t i = 0;
        const gf256 *m1 = matrix1 + col*n_rows1;
        const __m256i b = _mm256_set1_epi8(*(matrix2 + col));
        gf256 *r = result;
        while ((i + 32) <= n_rows1) {
            const __m256i a = _mm256_loadu_si256((const __m256i *)m1);
            __m256i t = gf256v_mul_u256(a, b);
            t ^= _mm256_loadu_si256((__m256i *)(r));
            _mm256_storeu_si256((__m256i *)(r), t);

            m1 += 32;
            r  += 32;
            i  += 32;
        }

        if (limit) {
            for (uint32_t j = 0; j < limit; ++j) { tmp[j] = m1[j]; }
            const __m256i a = _mm256_loadu_si256((const __m256i *)tmp);
            __m256i t = gf256v_mul_u256(a, b);
            _mm256_store_si256((__m256i *)tmp, t);
            for (uint32_t j = 0; j < limit; ++j) { r[j] ^= tmp[j]; }
        }
    }
}

/// \brief result = matrix1 * matrix2
/// 
/// \param[out] result Matrix over gf256
/// \param[in] matrix1 Matrix over gf256
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf256_matrix_product_u256(gf256 *result,
                                             const gf256 *matrix1,
                                             const gf256 *matrix2,
                                             const uint32_t n_rows1,
                                             const uint32_t n_cols1,
                                             const uint32_t n_cols2) {
    if (n_cols2 == 1) {
        gf256_matrix_product_slim_u256(result, matrix1, matrix2, n_rows1, n_cols1);
        return;
    }

    gf256_matrix_product_le64xBxle48_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

/// \param[out] out Matrix over gf2to12
/// \param[in] input Matrix over gf2
/// \param[in] nrows number of rows
/// \param[in] ncols number of columns
static inline void gf2to12_matrix_map_gf2_u256(gf256 *out,
                                               const gf2 *input,
                                               const uint32_t nrows,
                                               const uint32_t ncols) {
    if (nrows % 8 == 0) {
        gf256_vector_set_to_gf2_u256(out, input, nrows*ncols);
        return;
    }

    const uint32_t gf2_col_bytes = gf2_matrix_bytes_per_column(nrows);
    for (uint32_t col = 0; col < ncols; ++col) {
        const gf2 *in2 = input + col*gf2_col_bytes;
        gf256 *o = out + col*nrows;

        gf256_vector_set_to_gf2_u256(o, in2, nrows);
    }
}

#elif defined(USE_NEON)
#include <arm_neon.h>

/// source original from: https://github.com/pqov/pqov-paper
/// \param mat[out]
/// \param src_mat[in]
/// \param vec_len[in]
/// \param src_vec_len[in]
static  void gf256_matrix_transpose_32x32(uint8_t *mat,
                                           const uint8_t * src_mat,
                                           unsigned vec_len,
                                           unsigned src_vec_len ) {
    byte_transpose_16x16_neon(mat , vec_len , src_mat , src_vec_len );
    byte_transpose_16x16_neon(mat+16 , vec_len , src_mat+16*src_vec_len, src_vec_len );
    byte_transpose_16x16_neon(mat+16*vec_len , vec_len , src_mat+16, src_vec_len );
    byte_transpose_16x16_neon(mat+16*vec_len+16 , vec_len , src_mat+16*src_vec_len+16, src_vec_len );

}

/// orifinal code from: https://github.com/less-sig/LESS
/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param src_stride[in]: number of bytes (including alignment) in each row for source matrix
/// \param dst_stride[in]: number of bytes (including alignment) in each row for destination matrix
void gf256_matrix_transpose_32x32_v2(uint8_t* dst_origin,
                                    const uint8_t* src_origin,
                                    const size_t src_stride,
                                    const size_t dst_stride) {
    const vec256_t rnd_0_0 = *((vec256_t *)(src_origin + 0*src_stride));
    const vec256_t rnd_0_1 = *((vec256_t *)(src_origin + 1*src_stride));
    vec256_t rnd_1_0; rnd_1_0.v[0] = vtrn1q_u8(rnd_0_0.v[0], rnd_0_1.v[0]); rnd_1_0.v[1] = vtrn1q_u8(rnd_0_0.v[1], rnd_0_1.v[1]);
    vec256_t rnd_1_1; rnd_1_1.v[0] = vtrn2q_u8(rnd_0_0.v[0], rnd_0_1.v[0]); rnd_1_1.v[1] = vtrn2q_u8(rnd_0_0.v[1], rnd_0_1.v[1]);
    const vec256_t rnd_0_2 = *((vec256_t *)(src_origin + 2*src_stride));
    const vec256_t rnd_0_3 = *((vec256_t *)(src_origin + 3*src_stride));
    vec256_t rnd_1_2; rnd_1_2.v[0] = vtrn1q_u8(rnd_0_2.v[0], rnd_0_3.v[0]); rnd_1_2.v[1] = vtrn1q_u8(rnd_0_2.v[1], rnd_0_3.v[1]);
    vec256_t rnd_1_3; rnd_1_3.v[0] = vtrn2q_u8(rnd_0_2.v[0], rnd_0_3.v[0]); rnd_1_3.v[1] = vtrn2q_u8(rnd_0_2.v[1], rnd_0_3.v[1]);
    const vec256_t rnd_0_4 = *((vec256_t *)(src_origin + 4*src_stride));
    const vec256_t rnd_0_5 = *((vec256_t *)(src_origin + 5*src_stride));
    vec256_t rnd_1_4; rnd_1_4.v[0] = vtrn1q_u8(rnd_0_4.v[0], rnd_0_5.v[0]); rnd_1_4.v[1] = vtrn1q_u8(rnd_0_4.v[1], rnd_0_5.v[1]);
    vec256_t rnd_1_5; rnd_1_5.v[0] = vtrn2q_u8(rnd_0_4.v[0], rnd_0_5.v[0]); rnd_1_5.v[1] = vtrn2q_u8(rnd_0_4.v[1], rnd_0_5.v[1]);
    const vec256_t rnd_0_6 = *((vec256_t *)(src_origin + 6*src_stride));
    const vec256_t rnd_0_7 = *((vec256_t *)(src_origin + 7*src_stride));
    vec256_t rnd_1_6; rnd_1_6.v[0] = vtrn1q_u8(rnd_0_6.v[0], rnd_0_7.v[0]); rnd_1_6.v[1] = vtrn1q_u8(rnd_0_6.v[1], rnd_0_7.v[1]);
    vec256_t rnd_1_7; rnd_1_7.v[0] = vtrn2q_u8(rnd_0_6.v[0], rnd_0_7.v[0]); rnd_1_7.v[1] = vtrn2q_u8(rnd_0_6.v[1], rnd_0_7.v[1]);
    const vec256_t rnd_0_8 = *((vec256_t *)(src_origin + 8*src_stride));
    const vec256_t rnd_0_9 = *((vec256_t *)(src_origin + 9*src_stride));
    vec256_t rnd_1_8; rnd_1_8.v[0] = vtrn1q_u8(rnd_0_8.v[0], rnd_0_9.v[0]); rnd_1_8.v[1] = vtrn1q_u8(rnd_0_8.v[1], rnd_0_9.v[1]);
    vec256_t rnd_1_9; rnd_1_9.v[0] = vtrn2q_u8(rnd_0_8.v[0], rnd_0_9.v[0]); rnd_1_9.v[1] = vtrn2q_u8(rnd_0_8.v[1], rnd_0_9.v[1]);
    const vec256_t rnd_0_10 = *((vec256_t *)(src_origin + 10*src_stride));
    const vec256_t rnd_0_11 = *((vec256_t *)(src_origin + 11*src_stride));
    vec256_t rnd_1_10; rnd_1_10.v[0] = vtrn1q_u8(rnd_0_10.v[0], rnd_0_11.v[0]); rnd_1_10.v[1] = vtrn1q_u8(rnd_0_10.v[1], rnd_0_11.v[1]);
    vec256_t rnd_1_11; rnd_1_11.v[0] = vtrn2q_u8(rnd_0_10.v[0], rnd_0_11.v[0]); rnd_1_11.v[1] = vtrn2q_u8(rnd_0_10.v[1], rnd_0_11.v[1]);
    const vec256_t rnd_0_12 = *((vec256_t *)(src_origin + 12*src_stride));
    const vec256_t rnd_0_13 = *((vec256_t *)(src_origin + 13*src_stride));
    vec256_t rnd_1_12; rnd_1_12.v[0] = vtrn1q_u8(rnd_0_12.v[0], rnd_0_13.v[0]); rnd_1_12.v[1] = vtrn1q_u8(rnd_0_12.v[1], rnd_0_13.v[1]);
    vec256_t rnd_1_13; rnd_1_13.v[0] = vtrn2q_u8(rnd_0_12.v[0], rnd_0_13.v[0]); rnd_1_13.v[1] = vtrn2q_u8(rnd_0_12.v[1], rnd_0_13.v[1]);
    const vec256_t rnd_0_14 = *((vec256_t *)(src_origin + 14*src_stride));
    const vec256_t rnd_0_15 = *((vec256_t *)(src_origin + 15*src_stride));
    vec256_t rnd_1_14; rnd_1_14.v[0] = vtrn1q_u8(rnd_0_14.v[0], rnd_0_15.v[0]); rnd_1_14.v[1] = vtrn1q_u8(rnd_0_14.v[1], rnd_0_15.v[1]);
    vec256_t rnd_1_15; rnd_1_15.v[0] = vtrn2q_u8(rnd_0_14.v[0], rnd_0_15.v[0]); rnd_1_15.v[1] = vtrn2q_u8(rnd_0_14.v[1], rnd_0_15.v[1]);
    const vec256_t rnd_0_16 = *((vec256_t *)(src_origin + 16*src_stride));
    const vec256_t rnd_0_17 = *((vec256_t *)(src_origin + 17*src_stride));
    vec256_t rnd_1_16; rnd_1_16.v[0] = vtrn1q_u8(rnd_0_16.v[0], rnd_0_17.v[0]); rnd_1_16.v[1] = vtrn1q_u8(rnd_0_16.v[1], rnd_0_17.v[1]);
    vec256_t rnd_1_17; rnd_1_17.v[0] = vtrn2q_u8(rnd_0_16.v[0], rnd_0_17.v[0]); rnd_1_17.v[1] = vtrn2q_u8(rnd_0_16.v[1], rnd_0_17.v[1]);
    const vec256_t rnd_0_18 = *((vec256_t *)(src_origin + 18*src_stride));
    const vec256_t rnd_0_19 = *((vec256_t *)(src_origin + 19*src_stride));
    vec256_t rnd_1_18; rnd_1_18.v[0] = vtrn1q_u8(rnd_0_18.v[0], rnd_0_19.v[0]); rnd_1_18.v[1] = vtrn1q_u8(rnd_0_18.v[1], rnd_0_19.v[1]);
    vec256_t rnd_1_19; rnd_1_19.v[0] = vtrn2q_u8(rnd_0_18.v[0], rnd_0_19.v[0]); rnd_1_19.v[1] = vtrn2q_u8(rnd_0_18.v[1], rnd_0_19.v[1]);
    const vec256_t rnd_0_20 = *((vec256_t *)(src_origin + 20*src_stride));
    const vec256_t rnd_0_21 = *((vec256_t *)(src_origin + 21*src_stride));
    vec256_t rnd_1_20; rnd_1_20.v[0] = vtrn1q_u8(rnd_0_20.v[0], rnd_0_21.v[0]); rnd_1_20.v[1] = vtrn1q_u8(rnd_0_20.v[1], rnd_0_21.v[1]);
    vec256_t rnd_1_21; rnd_1_21.v[0] = vtrn2q_u8(rnd_0_20.v[0], rnd_0_21.v[0]); rnd_1_21.v[1] = vtrn2q_u8(rnd_0_20.v[1], rnd_0_21.v[1]);
    const vec256_t rnd_0_22 = *((vec256_t *)(src_origin + 22*src_stride));
    const vec256_t rnd_0_23 = *((vec256_t *)(src_origin + 23*src_stride));
    vec256_t rnd_1_22; rnd_1_22.v[0] = vtrn1q_u8(rnd_0_22.v[0], rnd_0_23.v[0]); rnd_1_22.v[1] = vtrn1q_u8(rnd_0_22.v[1], rnd_0_23.v[1]);
    vec256_t rnd_1_23; rnd_1_23.v[0] = vtrn2q_u8(rnd_0_22.v[0], rnd_0_23.v[0]); rnd_1_23.v[1] = vtrn2q_u8(rnd_0_22.v[1], rnd_0_23.v[1]);
    const vec256_t rnd_0_24 = *((vec256_t *)(src_origin + 24*src_stride));
    const vec256_t rnd_0_25 = *((vec256_t *)(src_origin + 25*src_stride));
    vec256_t rnd_1_24; rnd_1_24.v[0] = vtrn1q_u8(rnd_0_24.v[0], rnd_0_25.v[0]); rnd_1_24.v[1] = vtrn1q_u8(rnd_0_24.v[1], rnd_0_25.v[1]);
    vec256_t rnd_1_25; rnd_1_25.v[0] = vtrn2q_u8(rnd_0_24.v[0], rnd_0_25.v[0]); rnd_1_25.v[1] = vtrn2q_u8(rnd_0_24.v[1], rnd_0_25.v[1]);
    const vec256_t rnd_0_26 = *((vec256_t *)(src_origin + 26*src_stride));
    const vec256_t rnd_0_27 = *((vec256_t *)(src_origin + 27*src_stride));
    vec256_t rnd_1_26; rnd_1_26.v[0] = vtrn1q_u8(rnd_0_26.v[0], rnd_0_27.v[0]); rnd_1_26.v[1] = vtrn1q_u8(rnd_0_26.v[1], rnd_0_27.v[1]);
    vec256_t rnd_1_27; rnd_1_27.v[0] = vtrn2q_u8(rnd_0_26.v[0], rnd_0_27.v[0]); rnd_1_27.v[1] = vtrn2q_u8(rnd_0_26.v[1], rnd_0_27.v[1]);
    const vec256_t rnd_0_28 = *((vec256_t *)(src_origin + 28*src_stride));
    const vec256_t rnd_0_29 = *((vec256_t *)(src_origin + 29*src_stride));
    vec256_t rnd_1_28; rnd_1_28.v[0] = vtrn1q_u8(rnd_0_28.v[0], rnd_0_29.v[0]); rnd_1_28.v[1] = vtrn1q_u8(rnd_0_28.v[1], rnd_0_29.v[1]);
    vec256_t rnd_1_29; rnd_1_29.v[0] = vtrn2q_u8(rnd_0_28.v[0], rnd_0_29.v[0]); rnd_1_29.v[1] = vtrn2q_u8(rnd_0_28.v[1], rnd_0_29.v[1]);
    const vec256_t rnd_0_30 = *((vec256_t *)(src_origin + 30*src_stride));
    const vec256_t rnd_0_31 = *((vec256_t *)(src_origin + 31*src_stride));
    vec256_t rnd_1_30; rnd_1_30.v[0] = vtrn1q_u8(rnd_0_30.v[0], rnd_0_31.v[0]); rnd_1_30.v[1] = vtrn1q_u8(rnd_0_30.v[1], rnd_0_31.v[1]);
    vec256_t rnd_1_31; rnd_1_31.v[0] = vtrn2q_u8(rnd_0_30.v[0], rnd_0_31.v[0]); rnd_1_31.v[1] = vtrn2q_u8(rnd_0_30.v[1], rnd_0_31.v[1]);

    vec256_t rnd_2_0; rnd_2_0.v[0] = vtrn1q_u16(rnd_1_0.v[0], rnd_1_2.v[0]); rnd_2_0.v[1] = vtrn1q_u16(rnd_1_0.v[1], rnd_1_2.v[1]);
    vec256_t rnd_2_2; rnd_2_2.v[0] = vtrn2q_u16(rnd_1_0.v[0], rnd_1_2.v[0]); rnd_2_2.v[1] = vtrn2q_u16(rnd_1_0.v[1], rnd_1_2.v[1]);
    vec256_t rnd_2_1; rnd_2_1.v[0] = vtrn1q_u16(rnd_1_1.v[0], rnd_1_3.v[0]); rnd_2_1.v[1] = vtrn1q_u16(rnd_1_1.v[1], rnd_1_3.v[1]);
    vec256_t rnd_2_3; rnd_2_3.v[0] = vtrn2q_u16(rnd_1_1.v[0], rnd_1_3.v[0]); rnd_2_3.v[1] = vtrn2q_u16(rnd_1_1.v[1], rnd_1_3.v[1]);
    vec256_t rnd_2_4; rnd_2_4.v[0] = vtrn1q_u16(rnd_1_4.v[0], rnd_1_6.v[0]); rnd_2_4.v[1] = vtrn1q_u16(rnd_1_4.v[1], rnd_1_6.v[1]);
    vec256_t rnd_2_6; rnd_2_6.v[0] = vtrn2q_u16(rnd_1_4.v[0], rnd_1_6.v[0]); rnd_2_6.v[1] = vtrn2q_u16(rnd_1_4.v[1], rnd_1_6.v[1]);
    vec256_t rnd_2_5; rnd_2_5.v[0] = vtrn1q_u16(rnd_1_5.v[0], rnd_1_7.v[0]); rnd_2_5.v[1] = vtrn1q_u16(rnd_1_5.v[1], rnd_1_7.v[1]);
    vec256_t rnd_2_7; rnd_2_7.v[0] = vtrn2q_u16(rnd_1_5.v[0], rnd_1_7.v[0]); rnd_2_7.v[1] = vtrn2q_u16(rnd_1_5.v[1], rnd_1_7.v[1]);
    vec256_t rnd_2_8 ; rnd_2_8.v[0]  = vtrn1q_u16(rnd_1_8.v[0], rnd_1_10.v[0]); rnd_2_8.v[1]  = vtrn1q_u16(rnd_1_8.v[1], rnd_1_10.v[1]);
    vec256_t rnd_2_10; rnd_2_10.v[0] = vtrn2q_u16(rnd_1_8.v[0], rnd_1_10.v[0]); rnd_2_10.v[1] = vtrn2q_u16(rnd_1_8.v[1], rnd_1_10.v[1]);
    vec256_t rnd_2_9 ; rnd_2_9.v[0]  = vtrn1q_u16(rnd_1_9.v[0], rnd_1_11.v[0]); rnd_2_9.v[1]  = vtrn1q_u16(rnd_1_9.v[1], rnd_1_11.v[1]);
    vec256_t rnd_2_11; rnd_2_11.v[0] = vtrn2q_u16(rnd_1_9.v[0], rnd_1_11.v[0]); rnd_2_11.v[1] = vtrn2q_u16(rnd_1_9.v[1], rnd_1_11.v[1]);
    vec256_t rnd_2_12; rnd_2_12.v[0] = vtrn1q_u16(rnd_1_12.v[0], rnd_1_14.v[0]); rnd_2_12.v[1] = vtrn1q_u16(rnd_1_12.v[1], rnd_1_14.v[1]);
    vec256_t rnd_2_14; rnd_2_14.v[0] = vtrn2q_u16(rnd_1_12.v[0], rnd_1_14.v[0]); rnd_2_14.v[1] = vtrn2q_u16(rnd_1_12.v[1], rnd_1_14.v[1]);
    vec256_t rnd_2_13; rnd_2_13.v[0] = vtrn1q_u16(rnd_1_13.v[0], rnd_1_15.v[0]); rnd_2_13.v[1] = vtrn1q_u16(rnd_1_13.v[1], rnd_1_15.v[1]);
    vec256_t rnd_2_15; rnd_2_15.v[0] = vtrn2q_u16(rnd_1_13.v[0], rnd_1_15.v[0]); rnd_2_15.v[1] = vtrn2q_u16(rnd_1_13.v[1], rnd_1_15.v[1]);
    vec256_t rnd_2_16; rnd_2_16.v[0] = vtrn1q_u16(rnd_1_16.v[0], rnd_1_18.v[0]); rnd_2_16.v[1] = vtrn1q_u16(rnd_1_16.v[1], rnd_1_18.v[1]);
    vec256_t rnd_2_18; rnd_2_18.v[0] = vtrn2q_u16(rnd_1_16.v[0], rnd_1_18.v[0]); rnd_2_18.v[1] = vtrn2q_u16(rnd_1_16.v[1], rnd_1_18.v[1]);
    vec256_t rnd_2_17; rnd_2_17.v[0] = vtrn1q_u16(rnd_1_17.v[0], rnd_1_19.v[0]); rnd_2_17.v[1] = vtrn1q_u16(rnd_1_17.v[1], rnd_1_19.v[1]);
    vec256_t rnd_2_19; rnd_2_19.v[0] = vtrn2q_u16(rnd_1_17.v[0], rnd_1_19.v[0]); rnd_2_19.v[1] = vtrn2q_u16(rnd_1_17.v[1], rnd_1_19.v[1]);
    vec256_t rnd_2_20; rnd_2_20.v[0] = vtrn1q_u16(rnd_1_20.v[0], rnd_1_22.v[0]); rnd_2_20.v[1] = vtrn1q_u16(rnd_1_20.v[1], rnd_1_22.v[1]);
    vec256_t rnd_2_22; rnd_2_22.v[0] = vtrn2q_u16(rnd_1_20.v[0], rnd_1_22.v[0]); rnd_2_22.v[1] = vtrn2q_u16(rnd_1_20.v[1], rnd_1_22.v[1]);
    vec256_t rnd_2_21; rnd_2_21.v[0] = vtrn1q_u16(rnd_1_21.v[0], rnd_1_23.v[0]); rnd_2_21.v[1] = vtrn1q_u16(rnd_1_21.v[1], rnd_1_23.v[1]);
    vec256_t rnd_2_23; rnd_2_23.v[0] = vtrn2q_u16(rnd_1_21.v[0], rnd_1_23.v[0]); rnd_2_23.v[1] = vtrn2q_u16(rnd_1_21.v[1], rnd_1_23.v[1]);
    vec256_t rnd_2_24; rnd_2_24.v[0] = vtrn1q_u16(rnd_1_24.v[0], rnd_1_26.v[0]); rnd_2_24.v[1] = vtrn1q_u16(rnd_1_24.v[1], rnd_1_26.v[1]);
    vec256_t rnd_2_26; rnd_2_26.v[0] = vtrn2q_u16(rnd_1_24.v[0], rnd_1_26.v[0]); rnd_2_26.v[1] = vtrn2q_u16(rnd_1_24.v[1], rnd_1_26.v[1]);
    vec256_t rnd_2_25; rnd_2_25.v[0] = vtrn1q_u16(rnd_1_25.v[0], rnd_1_27.v[0]); rnd_2_25.v[1] = vtrn1q_u16(rnd_1_25.v[1], rnd_1_27.v[1]);
    vec256_t rnd_2_27; rnd_2_27.v[0] = vtrn2q_u16(rnd_1_25.v[0], rnd_1_27.v[0]); rnd_2_27.v[1] = vtrn2q_u16(rnd_1_25.v[1], rnd_1_27.v[1]);
    vec256_t rnd_2_28; rnd_2_28.v[0] = vtrn1q_u16(rnd_1_28.v[0], rnd_1_30.v[0]); rnd_2_28.v[1] = vtrn1q_u16(rnd_1_28.v[1], rnd_1_30.v[1]);
    vec256_t rnd_2_30; rnd_2_30.v[0] = vtrn2q_u16(rnd_1_28.v[0], rnd_1_30.v[0]); rnd_2_30.v[1] = vtrn2q_u16(rnd_1_28.v[1], rnd_1_30.v[1]);
    vec256_t rnd_2_29; rnd_2_29.v[0] = vtrn1q_u16(rnd_1_29.v[0], rnd_1_31.v[0]); rnd_2_29.v[1] = vtrn1q_u16(rnd_1_29.v[1], rnd_1_31.v[1]);
    vec256_t rnd_2_31; rnd_2_31.v[0] = vtrn2q_u16(rnd_1_29.v[0], rnd_1_31.v[0]); rnd_2_31.v[1] = vtrn2q_u16(rnd_1_29.v[1], rnd_1_31.v[1]);

    vec256_t rnd_3_0; rnd_3_0.v[0] = vtrn1q_u32(rnd_2_0.v[0], rnd_2_4.v[0]); rnd_3_0.v[1] = vtrn1q_u32(rnd_2_0.v[1], rnd_2_4.v[1]);
    vec256_t rnd_3_4; rnd_3_4.v[0] = vtrn2q_u32(rnd_2_0.v[0], rnd_2_4.v[0]); rnd_3_4.v[1] = vtrn2q_u32(rnd_2_0.v[1], rnd_2_4.v[1]);
    vec256_t rnd_3_1; rnd_3_1.v[0] = vtrn1q_u32(rnd_2_1.v[0], rnd_2_5.v[0]); rnd_3_1.v[1] = vtrn1q_u32(rnd_2_1.v[1], rnd_2_5.v[1]);
    vec256_t rnd_3_5; rnd_3_5.v[0] = vtrn2q_u32(rnd_2_1.v[0], rnd_2_5.v[0]); rnd_3_5.v[1] = vtrn2q_u32(rnd_2_1.v[1], rnd_2_5.v[1]);
    vec256_t rnd_3_2; rnd_3_2.v[0] = vtrn1q_u32(rnd_2_2.v[0], rnd_2_6.v[0]); rnd_3_2.v[1] = vtrn1q_u32(rnd_2_2.v[1], rnd_2_6.v[1]);
    vec256_t rnd_3_6; rnd_3_6.v[0] = vtrn2q_u32(rnd_2_2.v[0], rnd_2_6.v[0]); rnd_3_6.v[1] = vtrn2q_u32(rnd_2_2.v[1], rnd_2_6.v[1]);
    vec256_t rnd_3_3; rnd_3_3.v[0] = vtrn1q_u32(rnd_2_3.v[0], rnd_2_7.v[0]); rnd_3_3.v[1] = vtrn1q_u32(rnd_2_3.v[1], rnd_2_7.v[1]);
    vec256_t rnd_3_7; rnd_3_7.v[0] = vtrn2q_u32(rnd_2_3.v[0], rnd_2_7.v[0]); rnd_3_7.v[1] = vtrn2q_u32(rnd_2_3.v[1], rnd_2_7.v[1]);
    vec256_t rnd_3_8 ; rnd_3_8.v[0]  = vtrn1q_u32(rnd_2_8.v[0], rnd_2_12.v[0]); rnd_3_8.v[1]  = vtrn1q_u32(rnd_2_8.v[1], rnd_2_12.v[1]);
    vec256_t rnd_3_12; rnd_3_12.v[0] = vtrn2q_u32(rnd_2_8.v[0], rnd_2_12.v[0]); rnd_3_12.v[1] = vtrn2q_u32(rnd_2_8.v[1], rnd_2_12.v[1]);
    vec256_t rnd_3_9;  rnd_3_9.v[0] =  vtrn1q_u32(rnd_2_9.v[0], rnd_2_13.v[0]); rnd_3_9.v[1] =  vtrn1q_u32(rnd_2_9.v[1], rnd_2_13.v[1]);
    vec256_t rnd_3_13; rnd_3_13.v[0] = vtrn2q_u32(rnd_2_9.v[0], rnd_2_13.v[0]); rnd_3_13.v[1] = vtrn2q_u32(rnd_2_9.v[1], rnd_2_13.v[1]);
    vec256_t rnd_3_10; rnd_3_10.v[0] = vtrn1q_u32(rnd_2_10.v[0], rnd_2_14.v[0]); rnd_3_10.v[1] = vtrn1q_u32(rnd_2_10.v[1], rnd_2_14.v[1]);
    vec256_t rnd_3_14; rnd_3_14.v[0] = vtrn2q_u32(rnd_2_10.v[0], rnd_2_14.v[0]); rnd_3_14.v[1] = vtrn2q_u32(rnd_2_10.v[1], rnd_2_14.v[1]);
    vec256_t rnd_3_11; rnd_3_11.v[0] = vtrn1q_u32(rnd_2_11.v[0], rnd_2_15.v[0]); rnd_3_11.v[1] = vtrn1q_u32(rnd_2_11.v[1], rnd_2_15.v[1]);
    vec256_t rnd_3_15; rnd_3_15.v[0] = vtrn2q_u32(rnd_2_11.v[0], rnd_2_15.v[0]); rnd_3_15.v[1] = vtrn2q_u32(rnd_2_11.v[1], rnd_2_15.v[1]);
    vec256_t rnd_3_16; rnd_3_16.v[0] = vtrn1q_u32(rnd_2_16.v[0], rnd_2_20.v[0]); rnd_3_16.v[1] = vtrn1q_u32(rnd_2_16.v[1], rnd_2_20.v[1]);
    vec256_t rnd_3_20; rnd_3_20.v[0] = vtrn2q_u32(rnd_2_16.v[0], rnd_2_20.v[0]); rnd_3_20.v[1] = vtrn2q_u32(rnd_2_16.v[1], rnd_2_20.v[1]);
    vec256_t rnd_3_17; rnd_3_17.v[0] = vtrn1q_u32(rnd_2_17.v[0], rnd_2_21.v[0]); rnd_3_17.v[1] = vtrn1q_u32(rnd_2_17.v[1], rnd_2_21.v[1]);
    vec256_t rnd_3_21; rnd_3_21.v[0] = vtrn2q_u32(rnd_2_17.v[0], rnd_2_21.v[0]); rnd_3_21.v[1] = vtrn2q_u32(rnd_2_17.v[1], rnd_2_21.v[1]);
    vec256_t rnd_3_18; rnd_3_18.v[0] = vtrn1q_u32(rnd_2_18.v[0], rnd_2_22.v[0]); rnd_3_18.v[1] = vtrn1q_u32(rnd_2_18.v[1], rnd_2_22.v[1]);
    vec256_t rnd_3_22; rnd_3_22.v[0] = vtrn2q_u32(rnd_2_18.v[0], rnd_2_22.v[0]); rnd_3_22.v[1] = vtrn2q_u32(rnd_2_18.v[1], rnd_2_22.v[1]);
    vec256_t rnd_3_19; rnd_3_19.v[0] = vtrn1q_u32(rnd_2_19.v[0], rnd_2_23.v[0]); rnd_3_19.v[1] = vtrn1q_u32(rnd_2_19.v[1], rnd_2_23.v[1]);
    vec256_t rnd_3_23; rnd_3_23.v[0] = vtrn2q_u32(rnd_2_19.v[0], rnd_2_23.v[0]); rnd_3_23.v[1] = vtrn2q_u32(rnd_2_19.v[1], rnd_2_23.v[1]);
    vec256_t rnd_3_24; rnd_3_24.v[0] = vtrn1q_u32(rnd_2_24.v[0], rnd_2_28.v[0]); rnd_3_24.v[1] = vtrn1q_u32(rnd_2_24.v[1], rnd_2_28.v[1]);
    vec256_t rnd_3_28; rnd_3_28.v[0] = vtrn2q_u32(rnd_2_24.v[0], rnd_2_28.v[0]); rnd_3_28.v[1] = vtrn2q_u32(rnd_2_24.v[1], rnd_2_28.v[1]);
    vec256_t rnd_3_25; rnd_3_25.v[0] = vtrn1q_u32(rnd_2_25.v[0], rnd_2_29.v[0]); rnd_3_25.v[1] = vtrn1q_u32(rnd_2_25.v[1], rnd_2_29.v[1]);
    vec256_t rnd_3_29; rnd_3_29.v[0] = vtrn2q_u32(rnd_2_25.v[0], rnd_2_29.v[0]); rnd_3_29.v[1] = vtrn2q_u32(rnd_2_25.v[1], rnd_2_29.v[1]);
    vec256_t rnd_3_26; rnd_3_26.v[0] = vtrn1q_u32(rnd_2_26.v[0], rnd_2_30.v[0]); rnd_3_26.v[1] = vtrn1q_u32(rnd_2_26.v[1], rnd_2_30.v[1]);
    vec256_t rnd_3_30; rnd_3_30.v[0] = vtrn2q_u32(rnd_2_26.v[0], rnd_2_30.v[0]); rnd_3_30.v[1] = vtrn2q_u32(rnd_2_26.v[1], rnd_2_30.v[1]);
    vec256_t rnd_3_27; rnd_3_27.v[0] = vtrn1q_u32(rnd_2_27.v[0], rnd_2_31.v[0]); rnd_3_27.v[1] = vtrn1q_u32(rnd_2_27.v[1], rnd_2_31.v[1]);
    vec256_t rnd_3_31; rnd_3_31.v[0] = vtrn2q_u32(rnd_2_27.v[0], rnd_2_31.v[0]); rnd_3_31.v[1] = vtrn2q_u32(rnd_2_27.v[1], rnd_2_31.v[1]);

    vec256_t rnd_4_0; rnd_4_0.v[0] = vtrn1q_u64(rnd_3_0.v[0], rnd_3_8.v[0]); rnd_4_0.v[1] = vtrn1q_u64(rnd_3_0.v[1], rnd_3_8.v[1]);
    vec256_t rnd_4_8; rnd_4_8.v[0] = vtrn2q_u64(rnd_3_0.v[0], rnd_3_8.v[0]); rnd_4_8.v[1] = vtrn2q_u64(rnd_3_0.v[1], rnd_3_8.v[1]);
    vec256_t rnd_4_1; rnd_4_1.v[0] = vtrn1q_u64(rnd_3_1.v[0], rnd_3_9.v[0]); rnd_4_1.v[1] = vtrn1q_u64(rnd_3_1.v[1], rnd_3_9.v[1]);
    vec256_t rnd_4_9; rnd_4_9.v[0] = vtrn2q_u64(rnd_3_1.v[0], rnd_3_9.v[0]); rnd_4_9.v[1] = vtrn2q_u64(rnd_3_1.v[1], rnd_3_9.v[1]);
    vec256_t rnd_4_2 ; rnd_4_2.v[0]  = vtrn1q_u64(rnd_3_2.v[0], rnd_3_10.v[0]); rnd_4_2.v[1]  = vtrn1q_u64(rnd_3_2.v[1], rnd_3_10.v[1]);
    vec256_t rnd_4_10; rnd_4_10.v[0] = vtrn2q_u64(rnd_3_2.v[0], rnd_3_10.v[0]); rnd_4_10.v[1] = vtrn2q_u64(rnd_3_2.v[1], rnd_3_10.v[1]);
    vec256_t rnd_4_3;  rnd_4_3.v[0]  = vtrn1q_u64(rnd_3_3.v[0], rnd_3_11.v[0]); rnd_4_3.v[1]  = vtrn1q_u64(rnd_3_3.v[1], rnd_3_11.v[1]);
    vec256_t rnd_4_11; rnd_4_11.v[0] = vtrn2q_u64(rnd_3_3.v[0], rnd_3_11.v[0]); rnd_4_11.v[1] = vtrn2q_u64(rnd_3_3.v[1], rnd_3_11.v[1]);
    vec256_t rnd_4_4;  rnd_4_4.v[0]  = vtrn1q_u64(rnd_3_4.v[0], rnd_3_12.v[0]); rnd_4_4.v[1]  = vtrn1q_u64(rnd_3_4.v[1], rnd_3_12.v[1]);
    vec256_t rnd_4_12; rnd_4_12.v[0] = vtrn2q_u64(rnd_3_4.v[0], rnd_3_12.v[0]); rnd_4_12.v[1] = vtrn2q_u64(rnd_3_4.v[1], rnd_3_12.v[1]);
    vec256_t rnd_4_5;  rnd_4_5.v[0]  = vtrn1q_u64(rnd_3_5.v[0], rnd_3_13.v[0]); rnd_4_5.v[1]  = vtrn1q_u64(rnd_3_5.v[1], rnd_3_13.v[1]);
    vec256_t rnd_4_13; rnd_4_13.v[0] = vtrn2q_u64(rnd_3_5.v[0], rnd_3_13.v[0]); rnd_4_13.v[1] = vtrn2q_u64(rnd_3_5.v[1], rnd_3_13.v[1]);
    vec256_t rnd_4_6;  rnd_4_6.v[0]  = vtrn1q_u64(rnd_3_6.v[0], rnd_3_14.v[0]); rnd_4_6.v[1]  = vtrn1q_u64(rnd_3_6.v[1], rnd_3_14.v[1]);
    vec256_t rnd_4_14; rnd_4_14.v[0] = vtrn2q_u64(rnd_3_6.v[0], rnd_3_14.v[0]); rnd_4_14.v[1] = vtrn2q_u64(rnd_3_6.v[1], rnd_3_14.v[1]);
    vec256_t rnd_4_7;  rnd_4_7.v[0]  = vtrn1q_u64(rnd_3_7.v[0], rnd_3_15.v[0]); rnd_4_7.v[1]  = vtrn1q_u64(rnd_3_7.v[1], rnd_3_15.v[1]);
    vec256_t rnd_4_15; rnd_4_15.v[0] = vtrn2q_u64(rnd_3_7.v[0], rnd_3_15.v[0]); rnd_4_15.v[1] = vtrn2q_u64(rnd_3_7.v[1], rnd_3_15.v[1]);
    vec256_t rnd_4_16; rnd_4_16.v[0] = vtrn1q_u64(rnd_3_16.v[0], rnd_3_24.v[0]); rnd_4_16.v[1] = vtrn1q_u64(rnd_3_16.v[1], rnd_3_24.v[1]);
    vec256_t rnd_4_24; rnd_4_24.v[0] = vtrn2q_u64(rnd_3_16.v[0], rnd_3_24.v[0]); rnd_4_24.v[1] = vtrn2q_u64(rnd_3_16.v[1], rnd_3_24.v[1]);
    vec256_t rnd_4_17; rnd_4_17.v[0] = vtrn1q_u64(rnd_3_17.v[0], rnd_3_25.v[0]); rnd_4_17.v[1] = vtrn1q_u64(rnd_3_17.v[1], rnd_3_25.v[1]);
    vec256_t rnd_4_25; rnd_4_25.v[0] = vtrn2q_u64(rnd_3_17.v[0], rnd_3_25.v[0]); rnd_4_25.v[1] = vtrn2q_u64(rnd_3_17.v[1], rnd_3_25.v[1]);
    vec256_t rnd_4_18; rnd_4_18.v[0] = vtrn1q_u64(rnd_3_18.v[0], rnd_3_26.v[0]); rnd_4_18.v[1] = vtrn1q_u64(rnd_3_18.v[1], rnd_3_26.v[1]);
    vec256_t rnd_4_26; rnd_4_26.v[0] = vtrn2q_u64(rnd_3_18.v[0], rnd_3_26.v[0]); rnd_4_26.v[1] = vtrn2q_u64(rnd_3_18.v[1], rnd_3_26.v[1]);
    vec256_t rnd_4_19; rnd_4_19.v[0] = vtrn1q_u64(rnd_3_19.v[0], rnd_3_27.v[0]); rnd_4_19.v[1] = vtrn1q_u64(rnd_3_19.v[1], rnd_3_27.v[1]);
    vec256_t rnd_4_27; rnd_4_27.v[0] = vtrn2q_u64(rnd_3_19.v[0], rnd_3_27.v[0]); rnd_4_27.v[1] = vtrn2q_u64(rnd_3_19.v[1], rnd_3_27.v[1]);
    vec256_t rnd_4_20; rnd_4_20.v[0] = vtrn1q_u64(rnd_3_20.v[0], rnd_3_28.v[0]); rnd_4_20.v[1] = vtrn1q_u64(rnd_3_20.v[1], rnd_3_28.v[1]);
    vec256_t rnd_4_28; rnd_4_28.v[0] = vtrn2q_u64(rnd_3_20.v[0], rnd_3_28.v[0]); rnd_4_28.v[1] = vtrn2q_u64(rnd_3_20.v[1], rnd_3_28.v[1]);
    vec256_t rnd_4_21; rnd_4_21.v[0] = vtrn1q_u64(rnd_3_21.v[0], rnd_3_29.v[0]); rnd_4_21.v[1] = vtrn1q_u64(rnd_3_21.v[1], rnd_3_29.v[1]);
    vec256_t rnd_4_29; rnd_4_29.v[0] = vtrn2q_u64(rnd_3_21.v[0], rnd_3_29.v[0]); rnd_4_29.v[1] = vtrn2q_u64(rnd_3_21.v[1], rnd_3_29.v[1]);
    vec256_t rnd_4_22; rnd_4_22.v[0] = vtrn1q_u64(rnd_3_22.v[0], rnd_3_30.v[0]); rnd_4_22.v[1] = vtrn1q_u64(rnd_3_22.v[1], rnd_3_30.v[1]);
    vec256_t rnd_4_30; rnd_4_30.v[0] = vtrn2q_u64(rnd_3_22.v[0], rnd_3_30.v[0]); rnd_4_30.v[1] = vtrn2q_u64(rnd_3_22.v[1], rnd_3_30.v[1]);
    vec256_t rnd_4_23; rnd_4_23.v[0] = vtrn1q_u64(rnd_3_23.v[0], rnd_3_31.v[0]); rnd_4_23.v[1] = vtrn1q_u64(rnd_3_23.v[1], rnd_3_31.v[1]);
    vec256_t rnd_4_31; rnd_4_31.v[0] = vtrn2q_u64(rnd_3_23.v[0], rnd_3_31.v[0]); rnd_4_31.v[1] = vtrn2q_u64(rnd_3_23.v[1], rnd_3_31.v[1]);

    // NOTE: maybe It's useful to reoptimize the following code.
    // As we are just shuffling some register we could do
    // already do this in round 4, just above.
    vec256_t rnd_5_0;  rnd_5_0.v[0]  = rnd_4_0.v[0]; rnd_5_0.v[1]  = rnd_4_16.v[0];
    vec256_t rnd_5_16; rnd_5_16.v[0] = rnd_4_0.v[1]; rnd_5_16.v[1] = rnd_4_16.v[1];
    *((vec256_t *)(dst_origin +  0*dst_stride)) = rnd_5_0;
    *((vec256_t *)(dst_origin + 16*dst_stride)) = rnd_5_16;
    vec256_t rnd_5_1;  rnd_5_1.v[0]  = rnd_4_1.v[0]; rnd_5_1.v[1]  = rnd_4_17.v[0];
    vec256_t rnd_5_17; rnd_5_17.v[0] = rnd_4_1.v[1]; rnd_5_17.v[1] = rnd_4_17.v[1];
    *((vec256_t *)(dst_origin +  1*dst_stride)) = rnd_5_1;
    *((vec256_t *)(dst_origin + 17*dst_stride)) = rnd_5_17;
    vec256_t rnd_5_2;  rnd_5_2.v[0]  = rnd_4_2.v[0];  rnd_5_2.v[1] = rnd_4_18.v[0];
    vec256_t rnd_5_18; rnd_5_18.v[0] = rnd_4_2.v[1]; rnd_5_18.v[1] = rnd_4_18.v[1];
    *((vec256_t *)(dst_origin +  2*dst_stride)) = rnd_5_2;
    *((vec256_t *)(dst_origin + 18*dst_stride)) = rnd_5_18;
    vec256_t rnd_5_3;  rnd_5_3.v[0]  = rnd_4_3.v[0];  rnd_5_3.v[1] = rnd_4_19.v[0];
    vec256_t rnd_5_19; rnd_5_19.v[0] = rnd_4_3.v[1]; rnd_5_19.v[1] = rnd_4_19.v[1];
    *((vec256_t *)(dst_origin +  3*dst_stride)) = rnd_5_3;
    *((vec256_t *)(dst_origin + 19*dst_stride)) = rnd_5_19;
    vec256_t rnd_5_4;  rnd_5_4.v[0]  = rnd_4_4.v[0];  rnd_5_4.v[1] = rnd_4_20.v[0];
    vec256_t rnd_5_20; rnd_5_20.v[0] = rnd_4_4.v[1]; rnd_5_20.v[1] = rnd_4_20.v[1];
    *((vec256_t *)(dst_origin +  4*dst_stride)) = rnd_5_4;
    *((vec256_t *)(dst_origin + 20*dst_stride)) = rnd_5_20;
    vec256_t rnd_5_5;  rnd_5_5.v[0]  = rnd_4_5.v[0];  rnd_5_5.v[1] = rnd_4_21.v[0];
    vec256_t rnd_5_21; rnd_5_21.v[0] = rnd_4_5.v[1]; rnd_5_21.v[1] = rnd_4_21.v[1];
    *((vec256_t *)(dst_origin +  5*dst_stride)) = rnd_5_5;
    *((vec256_t *)(dst_origin + 21*dst_stride)) = rnd_5_21;
    vec256_t rnd_5_6;  rnd_5_6.v[0]  = rnd_4_6.v[0];  rnd_5_6.v[1] = rnd_4_22.v[0];
    vec256_t rnd_5_22; rnd_5_22.v[0] = rnd_4_6.v[1]; rnd_5_22.v[1] = rnd_4_22.v[1];
    *((vec256_t *)(dst_origin +  6*dst_stride)) = rnd_5_6;
    *((vec256_t *)(dst_origin + 22*dst_stride)) = rnd_5_22;
    vec256_t rnd_5_7;  rnd_5_7.v[0]  = rnd_4_7.v[0];  rnd_5_7.v[1] = rnd_4_23.v[0];
    vec256_t rnd_5_23; rnd_5_23.v[0] = rnd_4_7.v[1]; rnd_5_23.v[1] = rnd_4_23.v[1];
    *((vec256_t *)(dst_origin +  7*dst_stride)) = rnd_5_7;
    *((vec256_t *)(dst_origin + 23*dst_stride)) = rnd_5_23;
    vec256_t rnd_5_8;  rnd_5_8.v[0]  = rnd_4_8.v[0];  rnd_5_8.v[1] = rnd_4_24.v[0];
    vec256_t rnd_5_24; rnd_5_24.v[0] = rnd_4_8.v[1]; rnd_5_24.v[1] = rnd_4_24.v[1];
    *((vec256_t *)(dst_origin +  8*dst_stride)) = rnd_5_8;
    *((vec256_t *)(dst_origin + 24*dst_stride)) = rnd_5_24;
    vec256_t rnd_5_9;  rnd_5_9.v[0]  = rnd_4_9.v[0];  rnd_5_9.v[1] = rnd_4_25.v[0];
    vec256_t rnd_5_25; rnd_5_25.v[0] = rnd_4_9.v[1]; rnd_5_25.v[1] = rnd_4_25.v[1];
    *((vec256_t *)(dst_origin +  9*dst_stride)) = rnd_5_9;
    *((vec256_t *)(dst_origin + 25*dst_stride)) = rnd_5_25;
    vec256_t rnd_5_10; rnd_5_10.v[0] = rnd_4_10.v[0]; rnd_5_10.v[1] = rnd_4_26.v[0];
    vec256_t rnd_5_26; rnd_5_26.v[0] = rnd_4_10.v[1]; rnd_5_26.v[1] = rnd_4_26.v[1];
    *((vec256_t *)(dst_origin + 10*dst_stride)) = rnd_5_10;
    *((vec256_t *)(dst_origin + 26*dst_stride)) = rnd_5_26;
    vec256_t rnd_5_11; rnd_5_11.v[0] = rnd_4_11.v[0]; rnd_5_11.v[1] = rnd_4_27.v[0];
    vec256_t rnd_5_27; rnd_5_27.v[0] = rnd_4_11.v[1]; rnd_5_27.v[1] = rnd_4_27.v[1];
    *((vec256_t *)(dst_origin + 11*dst_stride)) = rnd_5_11;
    *((vec256_t *)(dst_origin + 27*dst_stride)) = rnd_5_27;
    vec256_t rnd_5_12; rnd_5_12.v[0] = rnd_4_12.v[0]; rnd_5_12.v[1] = rnd_4_28.v[0];
    vec256_t rnd_5_28; rnd_5_28.v[0] = rnd_4_12.v[1]; rnd_5_28.v[1] = rnd_4_28.v[1];
    *((vec256_t *)(dst_origin + 12*dst_stride)) = rnd_5_12;
    *((vec256_t *)(dst_origin + 28*dst_stride)) = rnd_5_28;
    vec256_t rnd_5_13; rnd_5_13.v[0] = rnd_4_13.v[0]; rnd_5_13.v[1] = rnd_4_29.v[0];
    vec256_t rnd_5_29; rnd_5_29.v[0] = rnd_4_13.v[1]; rnd_5_29.v[1] = rnd_4_29.v[1];
    *((vec256_t *)(dst_origin + 13*dst_stride)) = rnd_5_13;
    *((vec256_t *)(dst_origin + 29*dst_stride)) = rnd_5_29;
    vec256_t rnd_5_14; rnd_5_14.v[0] = rnd_4_14.v[0]; rnd_5_14.v[1] = rnd_4_30.v[0];
    vec256_t rnd_5_30; rnd_5_30.v[0] = rnd_4_14.v[1]; rnd_5_30.v[1] = rnd_4_30.v[1];
    *((vec256_t *)(dst_origin + 14*dst_stride)) = rnd_5_14;
    *((vec256_t *)(dst_origin + 30*dst_stride)) = rnd_5_30;
    vec256_t rnd_5_15; rnd_5_15.v[0] = rnd_4_15.v[0]; rnd_5_15.v[1] = rnd_4_31.v[0];
    vec256_t rnd_5_31; rnd_5_31.v[0] = rnd_4_15.v[1]; rnd_5_31.v[1] = rnd_4_31.v[1];
    *((vec256_t *)(dst_origin + 15*dst_stride)) = rnd_5_15;
    *((vec256_t *)(dst_origin + 31*dst_stride)) = rnd_5_31;
}

/// source original from: https://github.com/pqov/pqov-paper
/// \param dest_mat
/// \param dest_vec_len
/// \param src_sqmat
static  void gf16mat_64x64_sqmat_transpose_neon(uint8_t *dest_mat, unsigned dest_vec_len , const uint8_t *src_sqmat ) {
    gf256_matrix_transpose_32x32(dest_mat             ,  src_sqmat ,   dest_vec_len*2 ,64 );     // transpose even rows
    gf256_matrix_transpose_32x32(dest_mat+dest_vec_len,  src_sqmat+32, dest_vec_len*2,64 );  // transpose odd rows
    // transpose 2x2 4-bit blocks
    uint8x16_t mask_0f = vdupq_n_u8(0x0f);
    uint8x16_t mask_f0 = vdupq_n_u8(0xf0);
    for(int i=0;i<64;i+=2) {
        uint8x16_t row1_0 = vld1q_u8( dest_mat+i*dest_vec_len );
        uint8x16_t row1_1 = vld1q_u8( dest_mat+i*dest_vec_len +16);
        uint8x16_t row2_0 = vld1q_u8( dest_mat+(i+1)*dest_vec_len );
        uint8x16_t row2_1 = vld1q_u8( dest_mat+(i+1)*dest_vec_len +16);

        uint8x16_t out1_0 = (row1_0&mask_0f)^vshlq_n_u8(row2_0&mask_0f,4);
        uint8x16_t out1_1 = (row1_1&mask_0f)^vshlq_n_u8(row2_1&mask_0f,4);
        uint8x16_t out2_0 = (row2_0&mask_f0)^vshrq_n_u8(row1_0&mask_f0,4);
        uint8x16_t out2_1 = (row2_1&mask_f0)^vshrq_n_u8(row1_1&mask_f0,4);

        vst1q_u8( dest_mat+i*dest_vec_len        , out1_0 );
        vst1q_u8( dest_mat+i*dest_vec_len+16     , out1_1 );
        vst1q_u8( dest_mat+(i+1)*dest_vec_len    , out2_0 );
        vst1q_u8( dest_mat+(i+1)*dest_vec_len+16 , out2_1 );
    }
}
#endif

