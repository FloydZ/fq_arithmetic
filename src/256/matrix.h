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

/// \param n_rows
/// \param n_cols
gf256* gf256_matrix_alloc(const uint32_t n_rows,
                          const uint32_t n_cols) {
    return (gf256 *)calloc(sizeof(gf256), gf256_matrix_bytes_size(n_rows, n_cols));
}

static inline void gf256_matrix_print(const gf256 *matrix1,
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

static inline void gf256_matrix_random(gf256 *matrix,
                                       const uint32_t n_rows,
                                       const uint32_t n_cols) {
    for (uint32_t i = 0; i < (n_rows * n_cols); i++) {
        matrix[i] = rand();
    }
}

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

/// \brief matrix1 += scalar * matrix2
/// 
/// \param[out] matrix1 Matrix over gf256
/// \param[in] scalar scalar over gf256
/// \param[in] matrix2 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_gf16(gf256 *matrix1,
                                                  gf256 scalar,
                                                  const ff_t *matrix2,
                                                  uint32_t n_rows,
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
                                        gf256 scalar,
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
            const __m256i m1 = gf2to12v_expand_gf2_x32_u256(in);
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)in2);
            const __m256i t1 = m1 ^ m2;
            _mm256_storeu_si256((__m256i *)out, t1);
            in  += 4u;
            in2 += 32u;
            out += 32u;
            i   -= 32u;
        }

        while (i >= 16u) {
            const __m128i m1 = gf2to12v_expand_gf2_x16_u256(in);
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
/// \param[in] matrix2 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf256_matrix_add_multiple_gf16_u256(gf256 *matrix1,
                                                       const gf256 scalar,
                                                       const gf16 *matrix3,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {

    const __m256i perm256 = _mm256_load_si256((const __m256i *)gf256_expand_tab);
    const __m128i perm128 = _mm_load_si128((const __m128i *)gf256_expand_tab);
    const __m256i tab = gf256v_generate_multab_16_single_element_u256(scalar);
    const __m256i ml = _mm256_permute2x128_si256(tab, tab, 0);
    const __m256i mh = _mm256_permute2x128_si256(tab, tab, 0x11);
    const __m256i mask = _mm256_set1_epi8(0xf);

    const __m128i ml128 = _mm256_extracti128_si256(ml, 0);
    const __m128i mh128 = _mm256_extracti128_si256(mh, 0);
    const __m128i mask128 = _mm_set1_epi8(0xF);


    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = n_rows;
        gf16 *in   = (gf16  *)matrix3 + (j*n_rows + 1)/2;
        gf256 *in2 = (gf256 *)matrix1 + j*n_rows;
        gf256 *out = matrix1 + j*n_rows;
        while (i >= 32u) {
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)in2);

            // slower
            // const uint32_t t11 = *(uint32_t *)(in +  0);
            // const uint32_t t12 = *(uint32_t *)(in +  4);
            // const uint32_t t13 = *(uint32_t *)(in +  8);
            // const uint32_t t14 = *(uint32_t *)(in + 12);

            // const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            // const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
            // const uint64_t t23 = _pdep_u64(t13, 0x0F0F0F0F0F0F0F0F);
            // const uint64_t t24 = _pdep_u64(t14, 0x0F0F0F0F0F0F0F0F);

            // const __m256i t3 = _mm256_setr_epi64x(t21, t22, t23, t24);
            // const __m256i t4 = _mm256_shuffle_epi8(perm256, t3);

            // faster
            const __m128i load = _mm_loadu_si128((__m128i *)in);
            const __m128i ll = _mm_shuffle_epi8(perm128, load & mask128);
            const __m128i lh = _mm_shuffle_epi8(perm128, _mm_srli_epi16(load, 4) & mask128);
            const __m256i tl = _mm256_setr_m128i(ll, _mm_bsrli_si128(ll, 8));
            const __m256i th = _mm256_setr_m128i(lh, _mm_bsrli_si128(lh, 8));
            const __m256i t4 = _mm256_unpacklo_epi8(tl, th);
            const __m256i t5 = gf256_linear_transform_8x8_256b(ml, mh, t4, mask);
            const __m256i t6 = t5 ^ m2;
            _mm256_storeu_si256((__m256i *)out, t6);
            in  += 16u;
            out += 32u;
            i   -= 32u;
        }

        while (i >= 16u) {
            const __m128i m2 = _mm_loadu_si128((const __m128i *)in2);
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

        if (i) {
            uint8_t tmp[16] __attribute__((aligned(32)));
            for (uint32_t k = 0; k < i; k++) { tmp[k] = in2[k]; }
            const __m128i m2 = _mm_load_si128((const __m128i *) tmp);

            for (uint32_t k = 0; k < (i + 1) / 2; k++) { tmp[k] = in[k]; }
            const __m64 t21 = (__m64) _pdep_u64(*(uint32_t *) (tmp + 0), 0x0F0F0F0F0F0F0F0F);
            const __m64 t22 = (__m64) _pdep_u64(*(uint32_t *) (tmp + 4), 0x0F0F0F0F0F0F0F0F);

            const __m128i t3 = _mm_setr_epi64(t21, t22);
            const __m128i t4 = _mm_shuffle_epi8(perm128, t3);
            const __m128i t5 = gf256_linear_transform_8x8_128b(ml128, mh128, t4, mask128);
            const __m128i t6 = t5 ^ m2;
            _mm_store_si128((__m128i *) tmp, t6);
            for (uint32_t k = 0; k < i; k++) { out[k] = tmp[k]; }
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
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf256
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
static inline void gf256_matrix_product_gf16_2_slim_u256(gf256 *result,
                                                         const gf256 *matrix1,
                                                         const gf16 *matrix2,
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
    }

    // TODO
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
                                                    const gf2 *matrix2,
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
    // TODO
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
#endif

