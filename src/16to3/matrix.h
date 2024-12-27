#pragma once
#include "arith.h"
#include "vector.h"

#include "../helper.h"


// NOTE all matrices are column major
gf16to3* gf16to3_matrix_alloc(const uint32_t nrows,
                              const uint32_t ncols) {
    return (gf16to3 *)calloc(sizeof(gf16to3), nrows * ncols);
}

/// loads min(bytes, 4) from the i-th row and j-th col
/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param bytes
/// \return
uint64_t gf16to3_matrix_load4(const gf16to3 *m,
                              const uint32_t nrows,
                              const uint32_t i,
                              const uint32_t j,
                              const uint32_t bytes) {
    uint8_t tmp[4];
    for (uint32_t k = 0; k < MIN(bytes, 4); k++) {
        tmp[k] = m[j*nrows + i + k];
    }

    return *((uint64_t*)tmp);
}


#ifdef USE_AVX2

/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param bytes
/// \return
__m128i gf16to3_matrix_load8(const gf16to3 *m,
                              const uint32_t nrows,
                              const uint32_t i,
                              const uint32_t j,
                              const uint32_t bytes) {
    uint8_t tmp[16] __attribute__((aligned(32)));

    const uint32_t pos = j*nrows + i;
    uint8_t *p = (uint8_t *)(m + pos);
    for (uint32_t k = 0; k < MIN(bytes, 16); k++) {
        tmp[k] = p[k];
    }

    return _mm_load_si128((__m128i *)tmp);
}

// loads min(16, a)
__m256i gf16to3_matrix_load16(const gf16to3 *m,
                              const uint32_t a) {

    uint16_t tmp[16] __attribute__((aligned(32)));

    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        tmp[k] = m[k];
    }

    return _mm256_load_si256((__m256i *)tmp);
}

void gf16to3_matrix_write16(gf16to3 *m,
                            const __m256i t,
                            const uint32_t a) {
    uint16_t tmp[16] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)tmp, t);
    for (uint32_t k = 0; k < MIN(a, 16); k++) {
        m[k] = tmp[k];
    }
}

///
/// @param m
/// @param nrows
/// @param i
/// @param j
/// @param bytes
/// @param a
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

#endif

gf16to3 gf16to3_matrix_get(const gf16to3 *m,
                           const uint32_t nrows,
                           const uint32_t i,
                           const uint32_t j) {
    return m[j*nrows + i];
}

void gf16to3_matrix_set(gf16to3 *m,
                        const uint32_t nrows,
                        const uint32_t i,
                        const uint32_t j,
                        const gf16to3 a) {
    m[j*nrows + i] = a;
}

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

void gf16to3_matrix_zero(gf16to3 *m,
                 const uint32_t nrows,
                 const uint32_t ncols) {
    memset(m, 0, sizeof(gf16to3)*nrows*ncols);
}


void gf16to3_matrix_random(gf16to3 *m,
                         const uint32_t nrows,
                         const uint32_t ncols) {
    for(uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            gf16to3_matrix_set(m, nrows, i, j, rand() & 0xFFF);
        }
    }
}

void gf16to3_matrix_id(gf16to3 *m,
               const uint32_t nrows,
               const uint32_t ncols) {
    gf16to3_matrix_zero(m, nrows, ncols);
    for (uint32_t i = 0; i < MIN(nrows, ncols); ++i) {
        gf16to3_matrix_set(m, nrows, i, i, 1);
    }
}

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

/// \brief matrix1 += scalar * matrix2
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff_mu
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf16to3_matrix_add_multiple_2(gf16to3 *matrix1,
                                          gf16to3 scalar,
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


#ifdef USE_AVX2
#include <immintrin.h>


/// this implementation iterates over the rows of B
/// \param result
/// \param A
/// \param B
static inline void gf16to3_matrix_mul_16x16(gf16to3 *result,
                                            const gf16to3 *A,
                                            const gf16to3 *B) {
    gf16to3_matrix_zero(result, 16, 16);
    const uint32_t nrows_B = 16;
    const uint32_t ncols_B = 16;

    for (uint32_t j = 0; j < nrows_B; ++j) {
        const __m256i a = _mm256_loadu_si256((const __m256i *)(A + j*ncols_B));
        for (uint32_t i = 0; i < ncols_B; ++i) {
            const __m256i b = _mm256_set1_epi16(*(short *)(B + i*nrows_B + j));
            const __m256i c = gf16to3v_mul_u256(a, b);

            const __m256i r = _mm256_loadu_si256((const __m256i *)(result + i*ncols_B));
            const __m256i t = c ^ r;

            _mm256_storeu_si256((__m256i *)(result + i*ncols_B), t);
        }
    }
}

/// Assumes 16 rows in A, the rest doesnt matter
/// col major
static inline void gf16to3_matrix_mul_16xX(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    const uint32_t nrows = 16;
    gf16to3_matrix_zero(result, nrows, ncols_B);

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

/// Assumes 16 < rows <= 32 in A, the rest doesnt matter
/// col major
static inline void gf16to3_matrix_mul_le32xCxC(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint16_t nrows,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    gf16to3_matrix_zero(result, nrows, ncols_B);

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
            const __m128i b1 = _mm_set1_epi16(*(short *)(B + i*nrows_B + j + 0));
            const __m128i b2 = _mm_set1_epi16(*(short *)(B + i*nrows_B + j + 1));
            const __m256i b = _mm256_setr_m128i(b1, b2);
            const __m256i c = gf16to3v_mul_u256(a, b);

            const __m256i r = _mm256_loadu_si256((const __m256i *)(result + j*ncols_B));
            const __m256i t = c ^ r;

            _mm256_storeu_si256((__m256i *)(result + j*ncols_B), t);
        }
    }
}

/// assumes that B is a column vector
/// \param result
/// \param A
/// \param B
/// \param nrows_B
/// \param ncols_B
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
#endif // USE_AVX2

/**
 * \brief matrix1 = matrix2 + matrix3
 *
 * \param[out] matrix1 Matrix over ff_mu
 * \param[in] matrix2 Matrix over ff_mu
 * \param[in] matrix3 Matrix over ff
 * \param[in] n_rows number of rows
 * \param[in] n_cols number of columns
 */
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

/// TODO test
/// matrix1 += scalar
/// \param[out] matrix1 Matrix over ff_mu
/// \param[in] scalar scalar over ff_mu
/// \param[in] matrix2 Matrix over ff
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_add_scalar_gf16(gf16to3 *matrix1,
                                                  const gf16to3 scalar,
                                                  const ff_t *matrix2,
                                                  const uint32_t n_rows,
                                                  const uint32_t n_cols) {
    for (uint32_t i = 0; i < n_rows; i++) {
        for (uint32_t j = 0; j < n_cols; j++) {
            gf16to3 entry2, entry3;

            const gf16to3 entry1 = gf16to3_matrix_get(matrix1, n_rows, i, j);
            entry2 = gf16_matrix_get(matrix2, n_rows, i, j);
            entry3 = entry1 ^ gf16to3_mul_gf16(scalar, entry2);
            gf16to3_matrix_set(matrix1, n_rows, i, j, entry3);
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

#ifdef USE_AVX2

/// assumes <=16 rows in m2 and m3
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
    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        const uint32_t off = (j*n_rows+1)/2;
        for (; (i+16) <= n_rows; i+= 16) {
            const uint32_t t11 = *((uint32_t *)(matrix3 + off + i + 0));
            const uint32_t t12 = *((uint32_t *)(matrix3 + off + i + 4));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t22, t21);
            const __m256i m3 = _mm256_cvtepu8_epi16(t1);

            const __m256i m2 = _mm256_loadu_si256((const __m256i *)(matrix2 + j*n_rows + i*2));

            const __m256i m1 = m3 ^ m2;
            _mm256_storeu_si256((__m256i *)(matrix1 + j*n_rows + i*2), m1);
        }

        for (; (i+8) <= n_rows; i+= 8) {
            const uint32_t t11 = *((uint32_t *)(matrix3 + off + i + 0));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i m3 = _mm_cvtepu8_epi16(t1);

            const __m128i m2 = _mm_loadu_si128((const __m128i *)(matrix2 + j*n_rows + i*2));

            const __m128i m1 = m3 ^ m2;
            _mm_storeu_si128((__m128i *)(matrix1 + j*n_rows + i*2), m1);
        }

        if (i < n_rows) {
            const uint32_t rbytes = (n_rows - i + 1) / 2;
            const uint32_t wbytes = (n_rows - i) * 2;
            const uint32_t t11 = gf16_matrix_load4(matrix3, n_rows, i, j, rbytes);
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i m3 = _mm_cvtepu8_epi16(t1);

            // const __m128i m2 = _mm_loadu_si128((const __m128i *)(matrix2 + j*n_rows + i*2));
            const __m128i m2 = gf16to3_matrix_load8(matrix2, n_rows, i, j, wbytes);
            const __m128i m1 = m3 ^ m2;
            gf16to3_matrix_store16(matrix1, n_rows, i, j, wbytes, m1);
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
            const uint32_t t11 = *((uint32_t *)(matrix3 + (j*n_rows/2) + i + 0));
            const uint32_t t12 = *((uint32_t *)(matrix3 + (j*n_rows/2) + i + 4));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t22, t21);
            const __m256i m3 = _mm256_cvtepu8_epi16(t1);
            
            const __m256i m2 = _mm256_loadu_si256((const __m256i *)(matrix2 + j*n_rows + i*2));

            const __m256i m1 = m3 ^ m2;
            _mm256_storeu_si256((__m256i *)(matrix1 + j*n_rows + i*2), m1);
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
            const uint32_t t11 = *((uint32_t *)(matrix3 + (j*n_rows/2) + i + 0));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i m3 = _mm_cvtepu8_epi16(t1);

            const __m128i m2 = _mm_loadu_si128((const __m128i *)(matrix2 + j*n_rows + i*2));

            const __m128i m1 = m3 ^ m2;
            _mm_storeu_si128((__m128i *)(matrix1 + j*n_rows + i*2), m1);
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

    for (uint32_t j = 0; j < n_cols; j++) {
        uint32_t i = 0;
        const uint32_t off = (j*n_rows+1)/2;
        for (; (i+16) <= n_rows; i+= 16) {
            const uint32_t t11 = *((uint32_t *)(input + off + i + 0));
            const uint32_t t12 = *((uint32_t *)(input + off + i + 4));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t22, t21);
            const __m256i m1 = _mm256_cvtepu8_epi16(t1);
            _mm256_storeu_si256((__m256i *)(out + j*n_rows + i*2), m1);
        }

        for (; (i+8) <= n_rows; i+= 8) {
            const uint32_t t11 = *((uint32_t *)(input + off + i + 0));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i m1 = _mm_cvtepu8_epi16(t1);

            _mm_storeu_si128((__m128i *)(out + j*n_rows + i*2), m1);
        }

        if (i < n_rows) {
            const uint32_t rbytes = (n_rows - i + 1) / 2;
            const uint32_t wbytes = (n_rows - i) * 2;
            const uint32_t t11 = gf16_matrix_load4(input, n_rows, i, j, rbytes);
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i m1 = _mm_cvtepu8_epi16(t1);
            gf16to3_matrix_store16(out, n_rows, i, j, wbytes, m1);
        }
    }
}

/// TODO only working for 16x16
/// \param[out] matrix1 Matrix over gf16to3 = matrix2*matrix3
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] matrix3 Matrix over gf16
/// \param[in] n_rows number of rows in ma
/// \param[in] n_cols number of columns
static inline void gf16to3_matrix_mul_gf16_XxX(gf16to3 *matrix1,
                                               const gf16to3 *matrix2,
                                               const ff_t *matrix3,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols) {
    /// NOTE limitation of 16
    uint16_t row[16] __attribute__((aligned(64))) = {0};

    for (uint32_t k = 0; k < n_rows; k++) {
        /// load row k into buffer
        for (uint32_t j = 0; j < n_cols; j++) {
            row[j] = matrix2[j*n_rows + k];
        }

        const __m256i a1 = _mm256_load_si256((const __m256i *)row);
        __m256i tmp;

        for (uint32_t j = 0; j < n_cols; j++) {
            uint32_t i = 0;
            const uint32_t off = (j*n_rows+1)/2;
            for (; (i+16) <= n_rows; i+= 16) {
                const uint32_t t11 = *((uint32_t *)(matrix3 + off + i + 0));
                const uint32_t t12 = *((uint32_t *)(matrix3 + off + i + 4));
                const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
                const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
                const __m128i t1 = _mm_set_epi64x(t22, t21);
                const __m256i m3 = _mm256_cvtepu8_epi16(t1);
                tmp = gf16to3v_mul_u256(a1, m3);
            }

            for (; (i+8) <= n_rows; i+= 8) {
                const uint32_t t11 = *((uint32_t *)(matrix3 + off + i + 0));
                const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
                const __m128i t1 = _mm_set_epi64x(t21, t21);
                const __m128i m31 = _mm_cvtepu8_epi16(t1);
                const __m256i m3 = _mm256_setr_m128i(m31, _mm_setzero_si128());
                tmp = gf16to3v_mul_u256(a1, m3);
            }

            if (n_rows >= 8) {
                _mm256_store_si256((__m256i *)row, tmp);
            }

            for (;i < n_rows; i++) {
                uint8_t c = gf16_matrix_get(matrix3, n_rows, j, i);
                row[i] = gf16to3_mul_gf16(row[i], c);
            }

            // sum up
            gf16to3 sum = 0;
            for (uint32_t t = 0; t < n_cols; t++) {
                sum ^= row[t];
            }

            // write back
            matrix1[j*n_rows +k] = sum;
        }
    }
}

/// TODO testing
/// \brief result = matrix1 * matrix2
/// \param[out] result Matrix over gf16to3
/// \param[in] matrix1 Matrix over gf16
/// \param[in] matrix2 Matrix over gf16to3
/// \param[in] n_rows number of rows in matrix1
/// \param[in] n_cols number of columns and rows in matrix1 and matrix2 respectively
/// \param[in] n_cols2 number of columns in matrix2
static inline void gf16to3_matrix_mul_gf16_XxX_v2(gf16to3 *result,
                                               const ff_t *matrix1,
                                               const gf16to3 *matrix2,
                                               const uint32_t n_rows,
                                               const uint32_t n_cols,
                                               const uint32_t n_cols2) {
    memset(result, 0, n_rows * n_cols2 * sizeof(gf16to3));
    for (uint32_t k = 0; k < n_cols; k++) {
        /// load column k matrix A into register
        uint32_t i = 0, off=0;
        __m256i m1;

        for (; (i+16) <= n_rows; i+= 16) {
            const uint32_t t11 = *((uint32_t *)(matrix1 + off + i + 0));
            const uint32_t t12 = *((uint32_t *)(matrix1 + off + i + 4));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const uint64_t t22 = _pdep_u64(t12, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t22, t21);
            m1 = _mm256_cvtepu8_epi16(t1);
        }

        for (; (i+8) <= n_rows; i+= 8) {
            const uint32_t t11 = *((uint32_t *)(matrix1 + off + i + 0));
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i t2 = _mm_cvtepu8_epi16(t1);
            m1 = _mm256_setr_m128i(t2, _mm_setzero_si128());
        }

        if (i < n_rows) {
            const uint32_t rbytes = (n_rows - i + 1) / 2;
            const uint32_t t11 = gf16_matrix_load4(matrix1, n_rows, i, k, rbytes);
            const uint64_t t21 = _pdep_u64(t11, 0x0F0F0F0F0F0F0F0F);
            const __m128i t1 = _mm_set_epi64x(t21, t21);
            const __m128i t2 = _mm_cvtepu8_epi16(t1);
            m1 = _mm256_setr_m128i(t2, _mm_setzero_si128());
        }


        for (uint32_t j = 0; j < n_cols2; j++) {
            const __m256i b = _mm256_set1_epi16(gf16to3_matrix_get(matrix2, n_cols, k, j));
            const __m256i c = gf16v_mul_u256(b, m1);
            /// TODO assumes ncols == 16
            const __m256i r = _mm256_loadu_si256((__m256i *)(result + j*n_rows));
            const __m256i t = r ^ m1;
            _mm256_storeu_si256((__m256i *)(result + j*n_rows), t);
        }
    }
}

/// \brief matrix1 += scalar * matrix2
///
/// \param[out] matrix1 Matrix over gf16to3
/// \param[in] scalar  scalar over  gf16to3
/// \param[in] matrix2 Matrix over  gf16to3
/// \param[in] n_rows number of rows
/// \param[in] n_cols number of columns
static void gf16to3_matrix_add_multiple_2_XxX(gf16to3 *matrix1,
                                              gf16to3 scalar,
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
    gf16to3_vector_scalar_add_u256_3(matrix1, matrix2, scalar, matrix3, n_rows*n_cols);
}


#endif
