#pragma once
#include "arith.h"
#include "vector.h"

#include "../helper.h"


// NOTE all matrices are column major: m[j*nrows + i] = m(i, j)
/// \param nrows
/// \param ncols
/// \return zero initialized matrix
static inline
gf16to3* gf16to3_matrix_alloc(const uint32_t nrows,
                              const uint32_t ncols) {
    return (gf16to3 *)calloc(sizeof(gf16to3), nrows * ncols);
}

/// loads min(bytes, 8) bytes starting at the i-th row and j-th col
/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param bytes
/// \return
static inline
uint64_t gf16to3_matrix_load4(const gf16to3 *m,
                              const uint32_t nrows,
                              const uint32_t i,
                              const uint32_t j,
                              const uint32_t bytes) {
    uint8_t tmp[8] = {0};
    const uint8_t *p = (const uint8_t *)(m + j*nrows + i);
    for (uint32_t k = 0; k < MIN(bytes, 8); k++) {
        tmp[k] = p[k];
    }

    uint64_t ret;
    memcpy(&ret, tmp, 8);
    return ret;
}

/// \return m(i, j)
static inline
gf16to3 gf16to3_matrix_get(const gf16to3 *m,
                           const uint32_t nrows,
                           const uint32_t i,
                           const uint32_t j) {
    return m[j*nrows + i];
}

/// m(i, j) = a
static inline
void gf16to3_matrix_set(gf16to3 *m,
                        const uint32_t nrows,
                        const uint32_t i,
                        const uint32_t j,
                        const gf16to3 a) {
    m[j*nrows + i] = a;
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf16to3_matrix_print(const gf16to3 *m,
                          const uint32_t nrows,
                          const uint32_t ncols) {
    for(uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            printf("%4d ", gf16to3_matrix_get(m, nrows, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf16to3_matrix_zero(gf16to3 *m,
                         const uint32_t nrows,
                         const uint32_t ncols) {
    memset(m, 0, sizeof(gf16to3)*nrows*ncols);
}

/// \param m[out]: base pointer to the matrix
/// \param nrows
/// \param ncols
static inline
void gf16to3_matrix_random(gf16to3 *m,
                           const uint32_t nrows,
                           const uint32_t ncols) {
    for(uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf16to3_matrix_set(m, nrows, i, j, rand() & 0xFFF);
        }
    }
}

/// \param out[out]: = in
/// \param in[in]:
/// \param nrows
/// \param ncols
static inline
void gf16to3_matrix_copy(gf16to3 *out,
                         const gf16to3 *in,
                         const uint32_t nrows,
                         const uint32_t ncols) {
    memcpy(out, in, sizeof(gf16to3)*nrows*ncols);
}

/// \param m
/// \param nrows
/// \param ncols
static inline
void gf16to3_matrix_id(gf16to3 *m,
                       const uint32_t nrows,
                       const uint32_t ncols) {
    gf16to3_matrix_zero(m, nrows, ncols);
    for (uint32_t i = 0; i < MIN(nrows, ncols); ++i) {
        gf16to3_matrix_set(m, nrows, i, i, 1);
    }
}

/// \brief matrix1 = matrix2 + matrix3
static inline
void gf16to3_matrix_add(gf16to3 *matrix1,
                        const gf16to3 *matrix2,
                        const gf16to3 *matrix3,
                        const uint32_t n_rows,
                        const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows*n_cols; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// \brief out = in^T
/// \param out[out]: ncols x nrows
/// \param in[in]: nrows x ncols
static inline
void gf16to3_matrix_transpose(gf16to3 *out,
                              const gf16to3 *in,
                              const uint32_t nrows,
                              const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf16to3_matrix_set(out, ncols, j, i, gf16to3_matrix_get(in, nrows, i, j));
        }
    }
}

/// \param result[out]: matrix1*matrix2: n_rows1*n_cols2
/// \param matrix1[in]: n_rows1*n_cols1
/// \param matrix2[in]: n_cols1*n_cols2
/// \param n_rows1[in]: number of rows in m1
/// \param n_cols1[in]: number of cols in m1
/// \param n_cols2[in]: number of cols in m2
static inline void gf16to3_matrix_mul(gf16to3 *result,
                                      const gf16to3 *matrix1,
                                      const gf16to3 *matrix2,
                                      const uint32_t n_rows1,
                                      const uint32_t n_cols1,
                                      const uint32_t n_cols2) {
    gf16to3 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf16to3_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf16to3_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16to3_mul(entry_i_k, entry_k_j);
            }

            gf16to3_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \param result[in/out]: += matrix1*matrix2: n_rows1*n_cols2
/// \param matrix1[in]: n_rows1*n_cols1
/// \param matrix2[in]: n_cols1*n_cols2
/// \param n_rows1[in]: number of rows in m1
/// \param n_cols1[in]: number of cols in m1
/// \param n_cols2[in]: number of cols in m2
static inline void gf16to3_matrix_add_mul(gf16to3 *result,
                                          const gf16to3 *matrix1,
                                          const gf16to3 *matrix2,
                                          const uint32_t n_rows1,
                                          const uint32_t n_cols1,
                                          const uint32_t n_cols2) {
    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            gf16to3 entry_i_j = gf16to3_matrix_get(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                const gf16to3 entry_i_k = gf16to3_matrix_get(matrix1, n_rows1, i, k);
                const gf16to3 entry_k_j = gf16to3_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16to3_mul(entry_i_k, entry_k_j);
            }

            gf16to3_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_scalar(gf16to3 *matrix1,
                                             const gf16to3 scalar,
                                             const gf16to3 *matrix2,
                                             const uint32_t n_rows,
                                             const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf16to3 entry1 = gf16to3_matrix_get(matrix1, n_rows, i, j);
            const gf16to3 entry2 = gf16to3_matrix_get(matrix2, n_rows, i, j);
            const gf16to3 entry3 = entry1 ^ gf16to3_mul(scalar, entry2);
            gf16to3_matrix_set(matrix1, n_rows, i, j, entry3);
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
static inline void gf16to3_matrix_add_multiple_3(gf16to3 *matrix1,
                                                 const gf16to3 *matrix2,
                                                 const gf16to3 scalar,
                                                 const gf16to3 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf16to3 entry1 = gf16to3_matrix_get(matrix2, n_rows, i, j);
            const gf16to3 entry2 = gf16to3_matrix_get(matrix3, n_rows, i, j);
            const gf16to3 entry3 = entry1 ^ gf16to3_mul(scalar, entry2);

            gf16to3_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_scalar_gf16(gf16to3 *matrix1,
                                                  gf16to3 scalar,
                                                  const ff_t *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to3 entry2, entry3;

            const gf16to3 entry1 = gf16to3_matrix_get(matrix1, n_rows, i, j);
            entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            entry3 = entry1 ^ gf16to3_mul(scalar, entry2);
            gf16to3_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 = scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_scalar_gf16(gf16to3 *matrix1,
                                              gf16to3 scalar,
                                              const ff_t *matrix2,
                                              const uint32_t n_rows,
                                              const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            const gf16to3 entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            const gf16to3 entry3 = gf16to3_mul(scalar, entry2);
            gf16to3_matrix_set(matrix1, n_rows, i, j, entry3);
        }
    }
}

/// \brief matrix1 = matrix2 + matrix3
/// \param[out] matrix1 Matrix over gf16^3
/// \param[in] matrix2 Matrix over gf16^3
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_gf16(gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const ff_t *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to3 entry2;

            const gf16to3 entry1 = gf16to3_matrix_get(matrix2, n_rows, i, j);
            entry2 = gf16_matrix_get(matrix3, n_rows, i, j);
            entry2 = entry1 ^ entry2;
            gf16to3_matrix_set(matrix1, n_rows, i, j, entry2);
        }
    }
}

///\param[out] out Matrix over ff_mu
///\param[in] input Matrix over ff
///\param[in] nrows number of rows
///\param[in] ncols number of columns
static inline void gf16to3_matrix_map_gf16(gf16to3 *out,
                                           const uint8_t *input,
                                           const uint32_t nrows,
                                           const uint32_t ncols) {
    for (uint32_t i = 0; i < ncols; ++i) {
        for (uint32_t j = 0; j < nrows; ++j) {
            const uint8_t tmp = gf16_matrix_get(input, nrows, j, i);
            *out = gf16_to_gf16to3[tmp];
            out += 1;
        }
    }
}


/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16to3
/// \param[in] matrix2 Matrix over gf16
/// \param[in] n_rows1 number of rows in matrix1
/// \param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf16to3_matrix_mul_gf16(gf16to3 *result,
                                           const gf16to3 *matrix1,
                                           const ff_t *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf16to3 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf16to3_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf16_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16to3_mul_gf16(entry_i_k, entry_k_j);
            }

            gf16to3_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

///\brief result = matrix1 * matrix2
///\param[out] result Matrix over gf16to3
///\param[in] matrix1 Matrix over gf16
///\param[in] matrix2 Matrix over gf16to3
///\param[in] n_rows1 number of rows in matrix1
///\param[in] n_cols1 number of columns and rows in matrix1 and matrix2 respectively
///\param[in] n_cols2 number of columns in matrix2
static inline void gf16to3_matrix_product_gf16_v2(gf16to3 *result,
                                                  const gf16 *matrix1,
                                                  const gf16to3 *matrix2,
                                                  const uint32_t n_rows1,
                                                  const uint32_t n_cols1,
                                                  const uint32_t n_cols2) {
    gf16to3 entry_i_k, entry_k_j, entry_i_j;

    for(uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf16_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf16to3_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16to3_mul(entry_i_k, entry_k_j);
            }

            gf16to3_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// internal helper: loads the column `j` (`n` <= 16 elements) of the gf16
/// matrix `m` with `nrows` rows into a zero padded byte buffer.
static inline void gf16to3_matrix_load_gf16_col16(uint8_t tmp[8],
                                                  const gf16 *m,
                                                  const uint32_t nrows,
                                                  const uint32_t i,
                                                  const uint32_t j,
                                                  const uint32_t n) {
    memset(tmp, 0, 8);
    for (uint32_t k = 0; k < n; k++) {
        const uint8_t t = gf16_matrix_get(m, nrows, i + k, j);
        tmp[k/2] ^= t << (4*(k & 1u));
    }
}

#ifdef USE_AVX2
#include <immintrin.h>

/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param bytes
/// \return
static inline
__m128i gf16to3_matrix_load8(const gf16to3 *m,
                              const uint32_t nrows,
                              const uint32_t i,
                              const uint32_t j,
                              const uint32_t bytes) {
    uint8_t tmp[16] __attribute__((aligned(32))) = {0};

    const uint32_t pos = j*nrows + i;
    const uint8_t *p = (const uint8_t *)(m + pos);
    for (uint32_t k = 0; k < MIN(bytes, 16); k++) {
        tmp[k] = p[k];
    }

    return _mm_load_si128((__m128i *)tmp);
}

/// loads min(16, a) uint16_t limbs
static inline
__m256i gf16to3_matrix_load16(const gf16to3 *m,
                              const uint32_t a) {

    uint16_t tmp[16] __attribute__((aligned(32))) = {0};

    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        tmp[k] = m[k];
    }

    return _mm256_load_si256((__m256i *)tmp);
}

/// stores min(16, a) uint16_t limbs
static inline
void gf16to3_matrix_write16(gf16to3 *m,
                            const __m256i t,
                            const uint32_t a) {
    uint16_t tmp[16] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, t);
    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        m[k] = tmp[k];
    }
}

/// \param m base pointer to the matrix
/// \param nrows number of rows in m
/// \param i: row to access
/// \param j: column to access
/// \param bytes number of bytes to write <= 16
/// \param a sse register to write
static inline
void gf16to3_matrix_store16(gf16to3 *m,
                            const uint32_t nrows,
                            const uint32_t i,
                            const uint32_t j,
                            const uint32_t bytes,
                            const __m128i a) {
    uint8_t tmp[16] __attribute__((aligned(32)));
    _mm_store_si128((__m128i *)tmp, a);

    const uint32_t pos = j*nrows + i;
    uint8_t *p = (uint8_t *)(m + pos);
    for (uint32_t k = 0; k < MIN(bytes, 16); k++) {
        p[k] = tmp[k];
    }
}

/// this implementation iterates over the rows of B
/// \param result[out]: A*B
/// \param A[in]: 16*16
/// \param B[in]: 16*16
static inline void gf16to3_matrix_mul_16x16(gf16to3 *result,
                                            const gf16to3 *A,
                                            const gf16to3 *B) {
    gf16to3_matrix_zero(result, 16, 16);
    const uint32_t nrows_B = 16;
    const uint32_t ncols_B = 16;

    for (uint32_t j = 0; j < nrows_B; ++j) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(A + j*ncols_B));
        for (uint32_t i = 0; i < ncols_B; ++i) {
            const __m256i b = _mm256_set1_epi16(*(B + i*nrows_B + j));
            const __m256i c = gf16to3v_mul_u256(a, b);

            const __m256i r = _mm256_loadu_si256((const __m256i *)(result + i*ncols_B));
            const __m256i t = c ^ r;

            _mm256_storeu_si256((__m256i *)(result + i*ncols_B), t);
        }
    }
}

/// Assumes 16 rows in A, the rest does not matter
/// col major
static inline void gf16to3_matrix_mul_16xX(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    const uint32_t nrows = 16;

    for (uint32_t j = 0; j < nrows_B; ++j) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(A + j*nrows));
        for (uint32_t i = 0; i < ncols_B; ++i) {
            const __m256i b = _mm256_set1_epi16(*(B + i*nrows_B + j));
            const __m256i c = gf16to3v_mul_u256(a, b);

            const __m256i r = _mm256_loadu_si256((const __m256i *)(result + i*nrows));
            const __m256i t = c ^ r;

            _mm256_storeu_si256((__m256i *)(result + i*nrows), t);
        }
    }
}

/// Assumes 16 < rows <= 32 in A, the rest does not matter
/// NOTE: result += A*B
/// col major
static inline void gf16to3_matrix_mul_le32xCxC(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint16_t nrows,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    const uint32_t tail = nrows % 16;
    for (uint32_t j = 0; j < nrows_B; ++j) {
        const __m256i a1 = _mm256_loadu_si256((const __m256i *)(A + j*nrows));
        const __m256i a2 = gf16to3_matrix_load16(A + j*nrows + 16, tail);

        for (uint32_t i = 0; i < ncols_B; ++i) {
            const __m256i b = _mm256_set1_epi16(*(B + i*nrows_B + j));
            const __m256i c1 = gf16to3v_mul_u256(a1, b);
            const __m256i c2 = gf16to3v_mul_u256(a2, b);

            const __m256i r1 = _mm256_loadu_si256((const __m256i *)(result + i*nrows));
            const __m256i r2 = gf16to3_matrix_load16(result + i*nrows + 16, tail);
            const __m256i t1 = c1 ^ r1;
            const __m256i t2 = c2 ^ r2;

            _mm256_storeu_si256((__m256i *)(result + i*nrows), t1);
            gf16to3_matrix_write16(result + i*nrows + 16, t2, tail);
        }
    }
}

/// assumes 8 rows in A
/// \param result
/// \param A
/// \param B
/// \param nrows_B
/// \param ncols_B
static inline void gf16to3_matrix_mul_8xX(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    gf16to3_matrix_zero(result, 8, nrows_B);

    // TODO uneven number of columns in A
    for (uint32_t j = 0; j < nrows_B; j+=2) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(A + j*ncols_B));
        for (uint32_t i = 0; i < ncols_B; i++) {
            const __m128i b1 = _mm_set1_epi16(*(B + i*nrows_B + j + 0));
            const __m128i b2 = _mm_set1_epi16(*(B + i*nrows_B + j + 1));
            const __m256i b = _mm256_setr_m128i(b1, b2);
            const __m256i c = gf16to3v_mul_u256(a, b);

            const __m256i r = _mm256_loadu_si256((const __m256i *)(result + j*ncols_B));
            const __m256i t = c ^ r;

            _mm256_storeu_si256((__m256i *)(result + j*ncols_B), t);
        }
    }
}

/// assumes that B is a column vector
/// NOTE: result += A*B
/// \param result
/// \param A
/// \param B
/// \param nrows
/// \param ncols
static inline void gf16to3_matrix_mul_vector(gf16to3 *result,
                                             const gf16to3 *A,
                                             const gf16to3 *B,
                                             const uint32_t nrows,
                                             const uint32_t ncols) {
    const uint32_t limit = nrows % 16;
    for (uint32_t col = 0; col < ncols; ++col) {
        gf16to3 *r = result;
        const gf16to3 *m1 = A + col*nrows;
        const __m256i b1 = _mm256_set1_epi16(B[col]);
        uint32_t i = nrows;
        while (i >= 16) {
            const __m256i a1 = _mm256_loadu_si256((const __m256i *)m1);
            const __m256i c1 = gf16to3v_mul_u256(a1, b1);

            const __m256i r1 = _mm256_loadu_si256((const __m256i *)r);
            const __m256i c2 = c1 ^ r1;
            _mm256_storeu_si256((__m256i *)r, c2);

            i  -= 16;
            r  += 16;
            m1 += 16;
        }

        if (limit) {
            const __m256i a1 = gf16to3_matrix_load16(m1, limit);
            const __m256i c1 = gf16to3v_mul_u256(a1, b1);
            const __m256i r1 = gf16to3_matrix_load16(r, limit);
            const __m256i c2 = c1 ^ r1;
            gf16to3_matrix_write16(r, c2, limit);
        }
    }
}

/// @param matrix1 out = matrix2 + matrix3
/// @param matrix2 in
/// @param matrix3 in
/// @param n_rows
/// @param n_cols
static inline void gf16to3_matrix_add_gf16_XxX(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    const uint32_t bpc = gf16_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        const gf16 *col = matrix3 + j*bpc;
        gf16to3 *out = matrix1 + j*n_rows;
        const gf16to3 *in = matrix2 + j*n_rows;

        // NOTE: every column of a gf16 matrix starts at a byte boundary
        for (; (i+16) <= n_rows; i+= 16) {
            const __m256i m3 = gf16to3_vector_extend_gf16_x16(col + i/2);
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)(in + i));
            _mm256_storeu_si256((__m256i *)(out + i), m3 ^ m2);
        }

        for (; i < n_rows; i++) {
            out[i] = in[i] ^ gf16_matrix_get(matrix3, n_rows, i, j);
        }
    }
}

/// NOTE: cannot use the vector function, as nrows could be uneven
/// assumes 16 rows in m2 and m3
/// matrix1 = matrix2+matrix3
/// \param matrix1 out: over gf16to3
/// \param matrix2 in: over gf16to3
/// \param matrix3 in: over gf16
/// @param n_rows
/// @param n_cols
static inline void gf16to3_matrix_add_gf16_16x16(gf16to3 *matrix1,
                                                const gf16to3 *matrix2,
                                                const ff_t *matrix3,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        for (; (i+16) <= n_rows; i+= 16) {
            const __m256i m3 = gf16to3_vector_extend_gf16_x16(matrix3 + (j*n_rows/2) + i/2);
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)(matrix2 + j*n_rows + i));

            const __m256i m1 = m3 ^ m2;
            _mm256_storeu_si256((__m256i *)(matrix1 + j*n_rows + i), m1);
        }
    }
}

/// NOTE: cannot use the vector function, as nrows could be uneven
/// assumes 8 rows in m2 and m3
/// matrix1 = matrix2+matrix3
/// \param matrix1 out: over gf16to3
/// \param matrix2 in: over gf16to3
/// \param matrix3 in: over gf16
/// \param n_rows
/// \param n_cols
static inline void gf16to3_matrix_add_gf16_8x8(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        for (; (i+8) <= n_rows; i+= 8) {
            const __m128i m3 = gf16to3_vector_extend_gf16_x8(matrix3 + (j*n_rows/2) + i/2);
            const __m128i m2 = _mm_loadu_si128((const __m128i *)(matrix2 + j*n_rows + i));

            const __m128i m1 = m3 ^ m2;
            _mm_storeu_si128((__m128i *)(matrix1 + j*n_rows + i), m1);
        }
    }
}

/// \param[out] out Matrix over gf16to3
/// \param[in] input Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_map_gf16_u256(gf16to3 *out,
                                                const uint8_t *input,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    const uint32_t bpc = gf16_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        const gf16 *col = input + j*bpc;
        gf16to3 *o = out + j*n_rows;
        for (; (i+16) <= n_rows; i+= 16) {
            const __m256i m1 = gf16to3_vector_extend_gf16_x16(col + i/2);
            _mm256_storeu_si256((__m256i *)(o + i), m1);
        }

        if (i < n_rows) {
            uint8_t tmp[8];
            uint16_t tmp2[16] __attribute__((aligned(32)));
            gf16to3_matrix_load_gf16_col16(tmp, input, n_rows, i, j, n_rows - i);
            const __m256i m1 = gf16to3_vector_extend_gf16_x16(tmp);
            _mm256_store_si256((__m256i *)tmp2, m1);
            for (uint32_t k = 0; k < n_rows - i; k++) { o[i + k] = tmp2[k]; }
        }
    }
}

/// \param[out] matrix1 Matrix over gf16to3 = matrix2*matrix3
/// \param[in] matrix2 Matrix over gf16to3: n_rows x n_cols
/// \param[in] matrix3 Matrix over gf16: n_cols x n_cols
/// \param[in] n_rows number of rows in ma
/// \param[in] n_cols number of columns <= 16
static inline void gf16to3_matrix_mul_gf16_XxX(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    assert(n_cols <= 16);
    uint16_t row[16] __attribute__((aligned(32))) = {0};

    // precompute the extended columns of matrix3
    __m256i cols[16];
    for (uint32_t j = 0; j < n_cols; j++) {
        uint8_t tmp[8];
        gf16to3_matrix_load_gf16_col16(tmp, matrix3, n_cols, 0, j, n_cols);
        cols[j] = gf16to3_vector_extend_gf16_x16(tmp);
    }

    for (uint32_t k = 0; k < n_rows; k++) {
        /// load row k into buffer
        for (uint32_t j = 0; j < n_cols; j++) {
            row[j] = matrix2[j*n_rows + k];
        }

        const __m256i a1 = _mm256_load_si256((const __m256i *)row);
        for (uint32_t j = 0; j < n_cols; j++) {
            const __m256i tmp = gf16to3v_mul_u256(a1, cols[j]);
            matrix1[j*n_rows + k] = gf16to3_hadd_u256(tmp);
        }
    }
}

/// \brief result = matrix1 * matrix2
/// Assumes that matrix2 is a vector
/// NOTE: result += matrix1 * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_mul_vector_gf16_v2_u256(gf16to3 *result,
                                                          const ff_t *matrix1,
                                                          const gf16to3 *matrix2,
                                                          const uint32_t n_rows,
                                                          const uint32_t n_cols) {
    const uint32_t bytes = (n_rows+1)/2;
    uint8_t tmp[8];
    uint16_t tmp3[16] __attribute__((aligned(32)));
    const uint32_t limit = n_rows % 16;

    for (uint32_t k = 0; k < n_cols; k++) {
        uint32_t i = n_rows;

        const __m256i b = _mm256_set1_epi16(matrix2[k]);
        gf16to3    *r = result;
        const gf16 *m = matrix1 + k*bytes;

        while (i >= 16) {
            const __m256i a1 = gf16to3_vector_extend_gf16_x16(m);
            const __m256i a2 = a1 ^ _mm256_slli_epi16(a1, 8);
            const __m256i t2 = gf16v_mul_lo_u256(b, a2);
            const __m256i r1 = _mm256_loadu_si256((const __m256i *)r);
            const __m256i t3 = t2 ^ r1;
            _mm256_storeu_si256((__m256i *)r, t3);

            m += 8;
            r += 16;
            i -= 16;
        }

        if (limit) {
            gf16to3_vector_load_gf16_tail(tmp, m, limit);
            const __m256i a1 = gf16to3_vector_extend_gf16_x16(tmp);
            const __m256i a2 = a1 ^ _mm256_slli_epi16(a1, 8);
            const __m256i t2 = gf16v_mul_lo_u256(b, a2);
            _mm256_store_si256((__m256i *)tmp3, t2);

            for (uint32_t j = 0; j < limit; j++) { r[j] ^= tmp3[j];}
        }
    }
}

/// \brief result += matrix1 * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf16to3_matrix_mul_gf16_v2_u256(gf16to3 *result,
                                                   const ff_t *matrix1,
                                                   const gf16to3 *matrix2,
                                                   const uint32_t n_rows,
                                                   const uint32_t n_cols,
                                                   const uint32_t n_cols2) {
    // NOTE: every column of the gf16 matrix starts at a byte boundary
    for (uint32_t j = 0; j < n_cols2; j++) {
        gf16to3_matrix_mul_vector_gf16_v2_u256(result + j*n_rows, matrix1,
                                               matrix2 + j*n_cols, n_rows, n_cols);
    }
}

/// \brief matrix1 = matrix2 + matrix3
static inline void gf16to3_matrix_add_u256(gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const gf16to3 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf16to3_vector_add_u256_v2(matrix1, matrix2, matrix3, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
///
/// \param[out] matrix1 Matrix over gf16to3
/// \param[in] scalar  scalar over  gf16to3
/// \param[in] matrix2 Matrix over  gf16to3
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_scalar_u256(gf16to3 *matrix1,
                                                  const gf16to3 scalar,
                                                  const gf16to3 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf16to3_vector_scalar_add_u256(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over gf16to3
/// \param[in] matrix2 Matrix over  gf16to3
/// \param[in] scalar scalar over   gf16to3
/// \param[in] matrix3 Matrix over  gf16to3
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_multiple_3_XxX(gf16to3 *matrix1,
                                                 const gf16to3 *matrix2,
                                                 const gf16to3 scalar,
                                                 const gf16to3 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    gf16to3_vector_scalar_add_u256_v2(matrix1, matrix2, scalar, matrix3, n_rows*n_cols);
}

/// \brief result += matrix1 * matrix2, computed column wise:
///     result[:, j] += sum_k matrix2[k, j] * matrix1[:, k]
static inline void gf16to3_matrix_add_mul_u256(gf16to3 *result,
                                               const gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf16to3 b = gf16to3_matrix_get(matrix2, n_cols1, k, j);
            gf16to3_vector_scalar_add_u256(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf16to3_matrix_mul_u256(gf16to3 *result,
                                           const gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf16to3_matrix_zero(result, n_rows1, n_cols2);
    gf16to3_matrix_add_mul_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

/// \brief result = scalar * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] scalar value over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_scalar_gf16_u256(gf16to3 *result,
                                                   const gf16to3 scalar,
                                                   const gf16 *matrix2,
                                                   const uint32_t n_rows,
                                                   const uint32_t n_cols) {
    if ((n_rows % 8 == 0) && (n_cols == 1)) {
        gf16to3_vector_scalar_gf16_u256(result, scalar, matrix2, n_rows*n_cols);
        return;
    }

    uint8_t tmp[8];
    uint16_t tmp2[16] __attribute__((aligned(32)));

    const __m256i s = _mm256_set1_epi16(scalar);
    const uint32_t bytes = (n_rows + 1) / 2;

    for (uint32_t k = 0; k < n_cols; k++) {
        uint32_t i = n_rows;

        gf16to3 *r    = result  + k*n_rows;
        const gf16 *m = matrix2 + k*bytes;

        // avx2 code
        while (i >= 16u) {
            const __m256i t1 = gf16to3_vector_extend_gf16_x16(m);
            const __m256i t2 = gf16to3v_mul_u256(t1, s);

            _mm256_storeu_si256((__m256i *)r, t2);
            i   -= 16u;
            m  += 8u;
            r += 16u;
        }

        if (i) {
            gf16to3_vector_load_gf16_tail(tmp, m, i);

            const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
            const __m256i t2 = gf16to3v_mul_u256(t1, s);

            _mm256_store_si256((__m256i *)tmp2, t2);

            for (uint32_t j = 0; j < i; ++j) { r[j] = tmp2[j]; }
        }

    }
}

/// \brief result += scalar * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] scalar value over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_add_scalar_gf16_u256(gf16to3 *result,
                                                       const gf16to3 scalar,
                                                       const gf16 *matrix2,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {
    if ((n_rows % 8 == 0) && (n_cols == 1)) {
        gf16to3_vector_add_scalar_gf16_u256(result, scalar, matrix2, n_rows*n_cols);
        return;
    }

    uint8_t tmp[8];
    uint16_t tmp2[16] __attribute__((aligned(32)));

    const __m256i s = _mm256_set1_epi16(scalar);
    const uint32_t bytes = (n_rows + 1) / 2;

    for (uint32_t k = 0; k < n_cols; k++) {
        uint32_t i = n_rows;

        gf16to3 *r    = result  + k*n_rows;
        const gf16 *m = matrix2 + k*bytes;

        // avx2 code
        while (i >= 16u) {
            const __m256i t1 = gf16to3_vector_extend_gf16_x16(m);
            const __m256i t2 = gf16to3v_mul_u256(t1, s);
            const __m256i t3 = _mm256_loadu_si256((const __m256i *)r);

            _mm256_storeu_si256((__m256i *)r, t2^t3);
            i   -= 16u;
            m  += 8u;
            r += 16u;
        }

        if (i) {
            gf16to3_vector_load_gf16_tail(tmp, m, i);

            const __m256i t1 = gf16to3_vector_extend_gf16_x16(tmp);
            const __m256i t2 = gf16to3v_mul_u256(t1, s);

            _mm256_store_si256((__m256i *)tmp2, t2);

            for (uint32_t j = 0; j < i; ++j) { r[j] ^= tmp2[j]; }
        }
    }
}
#endif // end USE_AVX2

#ifdef USE_NEON

/// loads min(16, a) uint16_t limbs
static inline
uint16x8x2_t gf16to3_matrix_load16(const gf16to3 *m,
                                   const uint32_t a) {
    uint16_t tmp[16] = {0};
    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        tmp[k] = m[k];
    }

    return vld1q_u16_x2(tmp);
}

/// stores min(16, a) uint16_t limbs
static inline
void gf16to3_matrix_write16(gf16to3 *m,
                            const uint16x8x2_t t,
                            const uint32_t a) {
    uint16_t tmp[16];
    vst1q_u16_x2(tmp, t);
    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        m[k] = tmp[k];
    }
}

/// this implementation iterates over the rows of B
/// \param result[out]: A*B
/// \param A[in]: 16*16
/// \param B[in]: 16*16
static inline void gf16to3_matrix_mul_16x16(gf16to3 *result,
                                            const gf16to3 *A,
                                            const gf16to3 *B) {
    gf16to3_matrix_zero(result, 16, 16);
    const uint32_t nrows_B = 16;
    const uint32_t ncols_B = 16;

    for (uint32_t j = 0; j < nrows_B; ++j) {
        const uint16x8x2_t a = vld1q_u16_x2(A + j*ncols_B);
        for (uint32_t i = 0; i < ncols_B; ++i) {
            const uint16x8x2_t b = vdupq_n_u16_x2(*(B + i*nrows_B + j));
            const uint16x8x2_t c = gf16to3v_mul_u256(a, b);

            const uint16x8x2_t r = vld1q_u16_x2(result + i*ncols_B);
            vst1q_u16_x2(result + i*ncols_B, veorq_u16_x2(c, r));
        }
    }
}

/// Assumes 16 rows in A, the rest does not matter
/// col major
static inline void gf16to3_matrix_mul_16xX(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    const uint32_t nrows = 16;

    for (uint32_t j = 0; j < nrows_B; ++j) {
        const uint16x8x2_t a = vld1q_u16_x2(A + j*nrows);
        for (uint32_t i = 0; i < ncols_B; ++i) {
            const uint16x8x2_t b = vdupq_n_u16_x2(*(B + i*nrows_B + j));
            const uint16x8x2_t c = gf16to3v_mul_u256(a, b);

            const uint16x8x2_t r = vld1q_u16_x2(result + i*nrows);
            vst1q_u16_x2(result + i*nrows, veorq_u16_x2(c, r));
        }
    }
}

/// Assumes 16 < rows <= 32 in A, the rest does not matter
/// NOTE: result += A*B
/// col major
static inline void gf16to3_matrix_mul_le32xCxC(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint16_t nrows,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    const uint32_t tail = nrows % 16;
    for (uint32_t j = 0; j < nrows_B; ++j) {
        const uint16x8x2_t a1 = vld1q_u16_x2(A + j*nrows);
        const uint16x8x2_t a2 = gf16to3_matrix_load16(A + j*nrows + 16, tail);

        for (uint32_t i = 0; i < ncols_B; ++i) {
            const uint16x8x2_t b = vdupq_n_u16_x2(*(B + i*nrows_B + j));
            const uint16x8x2_t c1 = gf16to3v_mul_u256(a1, b);
            const uint16x8x2_t c2 = gf16to3v_mul_u256(a2, b);

            const uint16x8x2_t r1 = vld1q_u16_x2(result + i*nrows);
            const uint16x8x2_t r2 = gf16to3_matrix_load16(result + i*nrows + 16, tail);

            vst1q_u16_x2(result + i*nrows, veorq_u16_x2(c1, r1));
            gf16to3_matrix_write16(result + i*nrows + 16, veorq_u16_x2(c2, r2), tail);
        }
    }
}

/// assumes that B is a column vector
/// NOTE: result += A*B
/// \param result
/// \param A
/// \param B
/// \param nrows
/// \param ncols
static inline void gf16to3_matrix_mul_vector(gf16to3 *result,
                                             const gf16to3 *A,
                                             const gf16to3 *B,
                                             const uint32_t nrows,
                                             const uint32_t ncols) {
    for (uint32_t col = 0; col < ncols; ++col) {
        gf16to3_vector_scalar_add_u128(result, B[col], A + col*nrows, nrows);
    }
}

/// \param matrix1 out = matrix2 + matrix3
/// \param matrix2 in
/// \param matrix3 in
/// \param n_rows
/// \param n_cols
static inline void gf16to3_matrix_add_gf16_XxX(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    const uint32_t bpc = gf16_matrix_bytes_per_column(n_rows);
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        gf16to3 *out = matrix1 + j*n_rows;
        const gf16to3 *in = matrix2 + j*n_rows;
        const gf16 *col = matrix3 + j*bpc;
        for (; (i+16) <= n_rows; i+= 16) {
            const uint16x8x2_t m3 = gf16to3_vector_extend_gf16_x16(col + i/2);
            const uint16x8x2_t m2 = vld1q_u16_x2(in + i);
            vst1q_u16_x2(out + i, veorq_u16_x2(m3, m2));
        }

        for (; i < n_rows; i++) {
            out[i] = in[i] ^ gf16_matrix_get(matrix3, n_rows, i, j);
        }
    }
}

/// NOTE: cannot use the vector function, as nrows could be uneven
/// assumes 16 rows in m2 and m3
/// matrix1 = matrix2+matrix3
/// \param matrix1 out: over gf16to3
/// \param matrix2 in: over gf16to3
/// \param matrix3 in: over gf16
/// \param n_rows number of rows in all matrices
/// \param n_cols number of cols in all matrices
static inline void gf16to3_matrix_add_gf16_16x16(gf16to3 *matrix1,
                                                const gf16to3 *matrix2,
                                                const ff_t *matrix3,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        for (; (i+16) <= n_rows; i+= 16) {
            const uint16x8x2_t m3 = gf16to3_vector_extend_gf16_x16(matrix3 + (j*n_rows/2) + i/2);
            const uint16x8x2_t m2 = vld1q_u16_x2(matrix2 + j*n_rows + i);
            vst1q_u16_x2(matrix1 + j*n_rows + i, veorq_u16_x2(m3, m2));
        }
    }
}

/// NOTE: cannot use the vector function, as nrows could be uneven
/// assumes 8 rows in m2 and m3
/// matrix1 = matrix2+matrix3
/// \param matrix1 out: over gf16to3
/// \param matrix2 in: over gf16to3
/// \param matrix3 in: over gf16
/// \param n_rows number of rows in all matrices
/// \param n_cols number of cols in all matrices
static inline void gf16to3_matrix_add_gf16_8x8(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        for (; (i+8) <= n_rows; i+= 8) {
            const uint16x8_t m3 = gf16to3_vector_extend_gf16_x8(matrix3 +  (j*n_rows/2) + i/2);
            const uint16x8_t m2 = vld1q_u16(matrix2 + j*n_rows + i);
            vst1q_u16(matrix1 + j*n_rows + i, veorq_u16(m3, m2));
        }
    }
}

/// \param[out] out Matrix over gf16to3
/// \param[in] input Matrix over gf16
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_map_gf16_u128(gf16to3 *out,
                                                const uint8_t *input,
                                                const uint32_t n_rows,
                                                const uint32_t n_cols) {
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        gf16to3 *o = out + j*n_rows;
        for (; i < n_rows; i += 16) {
            const uint32_t n = MIN(16, n_rows - i);
            uint8_t tmp[8];
            uint16_t tmp2[16];
            gf16to3_matrix_load_gf16_col16(tmp, input, n_rows, i, j, n);
            vst1q_u16_x2(tmp2, gf16to3_vector_extend_gf16_x16(tmp));
            for (uint32_t k = 0; k < n; k++) { o[i + k] = tmp2[k]; }
        }
    }
}

/// \param[out] matrix1 Matrix over gf16to3 = matrix2*matrix3
/// \param[in] matrix2 Matrix over gf16to3: n_rows x n_cols
/// \param[in] matrix3 Matrix over gf16: n_cols x n_cols
/// \param[in] n_rows number of rows in ma
/// \param[in] n_cols number of columns <= 16
static inline void gf16to3_matrix_mul_gf16_XxX(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    assert(n_cols <= 16);
    uint16_t row[16] = {0};

    // precompute the extended columns of matrix3
    uint16x8x2_t cols[16];
    for (uint32_t j = 0; j < n_cols; j++) {
        uint8_t tmp[8];
        gf16to3_matrix_load_gf16_col16(tmp, matrix3, n_cols, 0, j, n_cols);
        cols[j] = gf16to3_vector_extend_gf16_x16(tmp);
    }

    for (uint32_t k = 0; k < n_rows; k++) {
        /// load row k into buffer
        for (uint32_t j = 0; j < n_cols; j++) {
            row[j] = matrix2[j*n_rows + k];
        }

        const uint16x8x2_t a1 = vld1q_u16_x2(row);
        for (uint32_t j = 0; j < n_cols; j++) {
            const uint16x8x2_t t = gf16to3v_mul_u256(a1, cols[j]);
            matrix1[j*n_rows + k] = gf16to3_hadd_u128(veorq_u16(t.val[0], t.val[1]));
        }
    }
}

/// \brief result += matrix1 * matrix2
/// Assumes that matrix2 is a vector
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_mul_vector_gf16_v2_u128(gf16to3 *result,
                                                          const ff_t *matrix1,
                                                          const gf16to3 *matrix2,
                                                          const uint32_t n_rows,
                                                          const uint32_t n_cols) {
    const uint32_t bytes = (n_rows+1)/2;
    uint8_t tmp[8];
    uint16_t tmp3[16];
    const uint32_t limit = n_rows % 16;

    for (uint32_t k = 0; k < n_cols; k++) {
        uint32_t i = n_rows;

        const uint16x8x2_t b = vdupq_n_u16_x2(matrix2[k]);
        gf16to3    *r = result;
        const gf16 *m = matrix1 + k*bytes;

        while (i >= 16) {
            const uint16x8x2_t a1 = gf16to3_vector_extend_gf16_x16(m);
            const uint16x8x2_t r1 = vld1q_u16_x2(r);
            const uint16x8x2_t t2 = gf16to3v_mul_u256(a1, b);
            vst1q_u16_x2(r, veorq_u16_x2(t2, r1));
            m += 8;
            r += 16;
            i -= 16;
        }

        if (limit) {
            gf16to3_vector_load_gf16_tail(tmp, m, limit);
            const uint16x8x2_t a1 = gf16to3_vector_extend_gf16_x16(tmp);
            const uint16x8x2_t t2 = gf16to3v_mul_u256(a1, b);
            vst1q_u16_x2(tmp3, t2);

            for (uint32_t j = 0; j < limit; j++) { r[j] ^= tmp3[j];}
        }
    }
}

/// \brief result += matrix1 * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf16to3_matrix_mul_gf16_v2_u128(gf16to3 *result,
                                                   const ff_t *matrix1,
                                                   const gf16to3 *matrix2,
                                                   const uint32_t n_rows,
                                                   const uint32_t n_cols,
                                                   const uint32_t n_cols2) {
    // NOTE: every column of the gf16 matrix starts at a byte boundary
    for (uint32_t j = 0; j < n_cols2; j++) {
        gf16to3_matrix_mul_vector_gf16_v2_u128(result + j*n_rows, matrix1,
                                               matrix2 + j*n_cols, n_rows, n_cols);
    }
}

/// \brief matrix1 = matrix2 + matrix3
static inline void gf16to3_matrix_add_u128(gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const gf16to3 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf16to3_vector_add_u128_v2(matrix1, matrix2, matrix3, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over gf16to3
/// \param[in] scalar  scalar over gf16to3
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1 and matrix2
/// \param[in] n_cols number of columns in matrix1 and matrix2
static inline void gf16to3_matrix_add_scalar_u128(gf16to3 *matrix1,
                                                  const gf16to3 scalar,
                                                  const gf16to3 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf16to3_vector_scalar_add_u128(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief matrix1 = matrix2 + scalar * matrix3
/// \param[out] matrix1 Matrix over gf16to3
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] scalar scalar over gf16to3
/// \param[in] matrix3 Matrix over gf16to3
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_multiple_3_XxX(gf16to3 *matrix1,
                                                 const gf16to3 *matrix2,
                                                 const gf16to3 scalar,
                                                 const gf16to3 *matrix3,
                                                 const uint32_t n_rows,
                                                 const uint32_t n_cols) {
    gf16to3_vector_scalar_add_u128_v2(matrix1, matrix2, scalar, matrix3, n_rows*n_cols);
}

/// \brief result += matrix1 * matrix2
static inline void gf16to3_matrix_add_mul_u128(gf16to3 *result,
                                               const gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf16to3 b = gf16to3_matrix_get(matrix2, n_cols1, k, j);
            gf16to3_vector_scalar_add_u128(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf16to3_matrix_mul_u128(gf16to3 *result,
                                           const gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf16to3_matrix_zero(result, n_rows1, n_cols2);
    gf16to3_matrix_add_mul_u128(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

/// \brief result = scalar * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] scalar value over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_scalar_gf16_u128(gf16to3 *result,
                                                   const gf16to3 scalar,
                                                   const gf16 *matrix2,
                                                   const uint32_t n_rows,
                                                   const uint32_t n_cols) {
    const uint32_t bytes = (n_rows + 1) / 2;
    for (uint32_t k = 0; k < n_cols; k++) {
        gf16to3_vector_scalar_gf16_u128(result + k*n_rows, scalar, matrix2 + k*bytes, n_rows);
    }
}

/// \brief result += scalar * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] scalar value over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
static inline void gf16to3_matrix_add_scalar_gf16_u128(gf16to3 *result,
                                                       const gf16to3 scalar,
                                                       const gf16 *matrix2,
                                                       const uint32_t n_rows,
                                                       const uint32_t n_cols) {
    const uint32_t bytes = (n_rows + 1) / 2;
    for (uint32_t k = 0; k < n_cols; k++) {
        gf16to3_vector_add_scalar_gf16_u128(result + k*n_rows, scalar, matrix2 + k*bytes, n_rows);
    }
}
#endif // end USE_NEON

#ifdef USE_AVX512
/// \brief matrix1 = matrix2 + matrix3
static inline void gf16to3_matrix_add_u512(gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const gf16to3 *matrix3,
                                           const uint32_t n_rows,
                                           const uint32_t n_cols) {
    gf16to3_vector_add_u512_v2(matrix1, matrix2, matrix3, n_rows*n_cols);
}

/// \brief matrix1 += scalar * matrix2
static inline void gf16to3_matrix_add_scalar_u512(gf16to3 *matrix1,
                                                  const gf16to3 scalar,
                                                  const gf16to3 *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    gf16to3_vector_scalar_add_u512(matrix1, scalar, matrix2, n_rows*n_cols);
}

/// \brief result += matrix1 * matrix2
static inline void gf16to3_matrix_add_mul_u512(gf16to3 *result,
                                               const gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const uint32_t n_rows1,
                                               const uint32_t n_cols1,
                                               const uint32_t n_cols2) {
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const gf16to3 b = gf16to3_matrix_get(matrix2, n_cols1, k, j);
            gf16to3_vector_scalar_add_u512(result + j*n_rows1, b, matrix1 + k*n_rows1, n_rows1);
        }
    }
}

/// \brief result = matrix1 * matrix2
static inline void gf16to3_matrix_mul_u512(gf16to3 *result,
                                           const gf16to3 *matrix1,
                                           const gf16to3 *matrix2,
                                           const uint32_t n_rows1,
                                           const uint32_t n_cols1,
                                           const uint32_t n_cols2) {
    gf16to3_matrix_zero(result, n_rows1, n_cols2);
    gf16to3_matrix_add_mul_u512(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif // end USE_AVX512
