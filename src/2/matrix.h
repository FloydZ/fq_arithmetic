#pragma once

#include "arith.h"


#define gf2_matrix_bytes_per_column(n_rows) (((n_rows) >> 3) + (((n_rows) & 0x1) | (((n_rows) & 0x2) >> 1) | (((n_rows) & 0x4) >> 2)))
#define gf2_matrix_bytes_size(n_rows, n_cols) ((gf2_matrix_bytes_per_column(n_rows)) * (n_cols))


static inline void gf2_matrix_set_to_ff(gf2 *matrix, 
                                        const uint32_t n_rows, 
                                        const uint32_t n_cols) {
    if (n_rows & 0x7) {
        const uint32_t matrix_height =  gf2_matrix_bytes_per_column(n_rows);
        const uint32_t matrix_height_x = matrix_height -  1;

        gf2 mask = 0xff >> (8 - (n_rows % 8));

        for (uint32_t i = 0; i < n_cols; i++) {
            matrix[i * matrix_height + matrix_height_x ] &= mask;
        }
    }
}

/// \return matrix[i, j]
static inline gf2 gf2_matrix_get(const gf2 *matrix, 
                                 const uint32_t n_rows, 
                                 const uint32_t i,
                                 const uint32_t j) {
    const uint32_t nbytes_col = gf2_matrix_bytes_per_column(n_rows);
    const uint32_t idx_line = i / 8;
    const uint32_t bit_line = i % 8;

    return (matrix[nbytes_col * j + idx_line] >> bit_line) & 0x01;
}


///  matrix[i, j] = scalar
static inline void gf2_matrix_set(gf2 *matrix,
                                  const uint32_t n_rows, 
                                  const uint32_t i,
                                  const uint32_t j,
                                  const gf2 scalar) {
    const uint32_t nbytes_col = gf2_matrix_bytes_per_column(n_rows);
    const uint32_t idx_line = i / 8;
    const uint32_t bit_line = i % 8;
    const uint8_t mask = 0xff ^ (1 << bit_line);
    matrix[nbytes_col*j + idx_line] = (matrix[nbytes_col*j + idx_line] & mask) ^ (scalar << bit_line);
}

/// matrix1 = matrix2 + matrix3
static inline void gf2_matrix_add(gf2 *matrix1, 
                                  const gf2 *matrix2, 
                                  const gf2 *matrix3,
		                          const uint32_t n_rows, 
                                  const uint32_t n_cols) {
    const uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar *matrix2
static inline void gf2_matrix_scalar_add(gf2 *matrix1, 
                                         const gf2 scalar, 
                                         const gf2 *matrix2,
                                         const uint32_t n_rows, 
                                         const uint32_t n_cols) {
    const uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);

    const uint8_t t = -scalar;
    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] ^= t&matrix2[i];
    }
}

/// result = matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf2_matrix_mul(gf2 *result, 
                                  const gf2 *matrix1, 
                                  const gf2 *matrix2,
                                  const uint32_t n_rows1, 
                                  const uint32_t n_cols1, 
                                  const uint32_t n_cols2) {
    uint32_t i, j, k;
    gf2 entry_i_k, entry_k_j, entry_i_j;

    for (i = 0; i < n_rows1; i++) {
        for (j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (k = 0; k < n_cols1; k++) {
                entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2_mul(entry_i_k, entry_k_j);
            }

            gf2_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}


#ifdef USE_AVX2 

/// matrix1 = matrix2 + matrix3
static inline void gf2_matrix_add_u256(gf2 *matrix1, 
                                       const gf2 *matrix2, 
                                       const gf2 *matrix3,
		                               const uint32_t n_rows, 
                                       const uint32_t n_cols) {
    uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);
    while(n_bytes >= 32) {
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)matrix2);
        const __m256i t3 = _mm256_loadu_si256((const __m256i *)matrix3);
        const __m256i t4 = t2 ^ t3;
        _mm256_storeu_si256((__m256i *)matrix1, t4);
        n_bytes -= 32;
        matrix1 += 32;
        matrix2 += 32;
        matrix3 += 32;
    }

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar*matrix2
static inline void gf2_matrix_scalar_add_u256(gf2 *matrix1, 
                                              const gf2 scalar, 
                                              const gf2 *matrix2,
                                              const uint32_t n_rows, 
                                              const uint32_t n_cols) {
    uint32_t n_bytes = gf2_matrix_bytes_size(n_rows, n_cols);
    const uint8_t m = -scalar;
    const __m256i m256 = _mm256_set1_epi8(m);
    while(n_bytes >= 32) {
        const __m256i t2 = _mm256_loadu_si256((const __m256i *)matrix2);
        const __m256i t1 = m ^ t2;
        _mm256_storeu_si256((__m256i *)matrix1, t1);
        n_bytes -= 32;
        matrix1 += 32;
        matrix2 += 32;
    }

    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] ^= m&matrix2[i];
    }
}

/// result = matrix1 * matrix2
/// matrix1 of size n_rows1 * n_cols1
/// matrix2 of size n_cols1 * n_cols2
/// result  of size n_rows1 * n_cols2
static inline void gf2_matrix_mul_u256(gf2 *result, 
                                       const gf2 *matrix1, 
                                       const gf2 *matrix2,
                                       const uint32_t n_rows1, 
                                       const uint32_t n_cols1, 
                                       const uint32_t n_cols2) {
    uint32_t i, j, k;
    gf2 entry_i_k, entry_k_j, entry_i_j;

    for (i = 0; i < n_rows1; i++) {
        for (j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (k = 0; k < n_cols1; k++) {
                entry_i_k = gf2_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf2_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf2_mul(entry_i_k, entry_k_j);
            }

            gf2_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}


#endif
