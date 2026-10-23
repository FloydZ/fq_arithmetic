#pragma once
#include <string.h>

#include "arith.h"
#include "vector.h"


#define gf16_matrix_bytes_per_column(n_rows) ((n_rows >> 1u) + (n_rows & 1u))
#define gf16_matrix_bytes_size(n_rows, n_cols) ((gf16_matrix_bytes_per_column(n_rows)) * (n_cols))

/// loads min(bytes, 4) bytes starting at the i-th row (i must be even) of the j-th column
/// \param m
/// \param nrows
/// \param i
/// \param j
/// \param bytes
static inline
uint32_t gf16_matrix_load4(const ff_t *m,
                           const uint32_t nrows,
                           const uint32_t i,
                           const uint32_t j,
                           const uint32_t bytes) {
    uint8_t tmp[4] = {0};
    const uint32_t pos = gf16_matrix_bytes_per_column(nrows) * j + i/2;
    for (uint32_t k = 0; k < MIN(bytes, 4); k++) {
        tmp[k] = m[pos + k];
    }

    uint32_t ret;
    memcpy(&ret, tmp, 4);
    return ret;
}

/// \param n_rows
/// \param n_cols
/// \return zero initialized matrix
static inline
ff_t* gf16_matrix_alloc(const uint32_t n_rows,
                        const uint32_t n_cols) {
    return (ff_t*)calloc(1, gf16_matrix_bytes_size(n_rows, n_cols));
}

/// \param matrix
/// \param n_rows
/// \param i
/// \param j
/// \return m[i, j]
static inline
ff_t gf16_matrix_get(const ff_t *matrix,
                      const uint32_t n_rows,
                      const uint32_t i,
                      const uint32_t j) {
    const uint32_t nbytes_col = gf16_matrix_bytes_per_column(n_rows);
    if (i & 1u) { // i is odd
        return  matrix[nbytes_col * j + (i >> 1)] >> 4;
    }
    else {
        return matrix[nbytes_col * j + (i >> 1)] & 0x0f;
    }
}

/// m[i, j] = scalar
/// \param matrix
/// \param n_rows
/// \param i
/// \param j
/// \param scalar
static inline
void gf16_matrix_set(ff_t *matrix,
                     const uint32_t n_rows,
                     const uint32_t i,
                     const uint32_t j,
                     const ff_t scalar) {
    const uint32_t nbytes_col = gf16_matrix_bytes_per_column(n_rows);
    const uint32_t target_byte_id = nbytes_col * j + (i >> 1);
    if (i & 1) // i is odd
    {
        matrix[target_byte_id] &= 0x0f;
        matrix[target_byte_id] |= ((scalar & 0xf) << 4);
    }
    else {
        matrix[target_byte_id] &= 0xf0;
        matrix[target_byte_id] |= (scalar & 0xf);
    }
}


// A[row1] -= a*A[row2]
static inline
void gf16_sub_row(uint8_t *A,
                  const uint32_t nrows,
                  const uint32_t ncols,
                  const uint32_t row1,
                  const uint32_t row2,
                  const uint8_t a) {
    for (uint32_t i = 0; i < ncols; i++) {
        uint8_t d1 = gf16_matrix_get(A, nrows, row1, i);
        uint8_t d2 = gf16_matrix_get(A, nrows, row2, i);
        d1 ^= gf16_mul(a, d2);
        gf16_matrix_set(A, nrows, row1, i, d1);
    }
}

/// NOTE: assumes that A is stored row-major (= transposed col-major),
///     each row consists of ceil(ncols/2) bytes.
// A[row1] -= a*A[row2]
static inline
void gf16_sub_row_transpose(uint8_t *A,
                            const uint32_t nrows,
                            const uint32_t ncols,
                            const uint32_t row1,
                            const uint32_t row2,
                            const uint8_t a) {
    (void)nrows;
    const uint32_t bytes = (ncols + 1)/2;
    uint32_t i = 0;
#if defined(USE_AVX2)
    const __m256i multab_l = gf16v_tbl32_multab(a);
    const __m256i multab_h = _mm256_slli_epi16(multab_l, 4 );
    const __m256i f = _mm256_set1_epi8(0xf);

    for (;i+32 <= bytes; i+=32) {
        const __m256i row1_256 = _mm256_loadu_si256((__m256i *)(A +row1*bytes + i));
        const __m256i row2_256 = _mm256_loadu_si256((__m256i *)(A +row2*bytes + i));
        const __m256i d = linear_transform_8x8_256b(multab_l, multab_h, row2_256, f);
        const __m256i e = d ^ row1_256;
        _mm256_storeu_si256((__m256i *)(A +row1*bytes + i), e);
    }

#endif
    const uint8_t aa = (a & 0xf) ^ ((a & 0xf) << 4);
    for (; i < bytes; i++) {
        A[row1*bytes + i] ^= gf16v_mul(A[row2*bytes + i], aa);
    }
}

///
/// \param A
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_print(const uint8_t *A,
                       const uint32_t nrows,
                       const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            printf("%2u ", gf16_matrix_get(A, nrows, i, j));
        }
        printf("\n");
    }
    printf("\n\n");
}

///
/// \param A
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_print_transposed(const uint8_t *A,
                                  const uint32_t nrows,
                                  const uint32_t ncols) {
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            printf("%2u ", gf16_matrix_get(A, nrows, j, i));
        }
        printf("\n");
    }
    printf("\n\n");
}

/// \param C
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_zero(uint8_t *C,
				    const size_t nrows,
				    const size_t ncols) {
    memset((void *)C, 0, gf16_matrix_bytes_size(nrows, ncols));
}

/// \param C
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_id(uint8_t *C,
				    const size_t nrows,
				    const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    gf16_matrix_set(C, nrows, i, j, i==j);
		}
	}
}

/// \param C
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_random(uint8_t *C,
				        const size_t nrows,
				        const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    gf16_matrix_set(C, nrows, i, j, rand()&0xF);
		}
	}
}

/// generates a random matrix of full rank (min(nrows, ncols)):
/// starts with [I | R] (or [I ; R] if nrows > ncols) and applies random
/// row operations, which do not change the rank.
/// \param C
/// \param nrows
/// \param ncols
static inline
void gf16_matrix_rng_full_rank(uint8_t *C,
				               const size_t nrows,
				               const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    const ff_t v = (i < ncols && j < nrows) ? (i == j) : (rand()&0xF);
		    gf16_matrix_set(C, nrows, i, j, v);
		}
	}

	for (uint32_t i = 0; i < nrows; i++) {
	    for (uint32_t j = 0; j < nrows; j++) {
	        if (i == j) { continue;}
	        uint32_t r = rand();
	        if (r&1) {
	            gf16_sub_row(C, nrows, ncols, i ,j, (r >> 1) & 0xF);
	        }
	    }
	}
}

/// A, B, C are in row-majow form, one element per byte
static inline
void gf16_matrix_mul_rowmajor_u8(uint8_t *C,
                                 const uint8_t *A,
                                 const uint8_t *B,
                                 const size_t nrows_A,
                                 const size_t ncols_A,
                                 const size_t ncols_B) {
	for (uint32_t i = 0; i < nrows_A; i++) {
		for (uint32_t j = 0; j < ncols_B; j++) {
			ff_t t = 0;
			for (uint32_t k = 0; k < ncols_A; k++) {
				t = gf16_add(t, gf16_mul(A[i*ncols_A + k], B[ncols_B*k + j]));
			}

			C[i*ncols_B + j] = t;
		}
	}
}

/// result = matrix1 * matrix2, column major
/// \param result[out]: n_rows1 x n_cols2
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
/// \param n_rows1
/// \param n_cols1
/// \param n_cols2
static inline
void gf16_matrix_mul(ff_t *result,
                         const ff_t *matrix1,
                         const ff_t *matrix2,
                         const uint32_t n_rows1,
                         const uint32_t n_cols1,
                         const uint32_t n_cols2) {
    uint32_t i, j, k;
    ff_t entry_i_k, entry_k_j, entry_i_j;

    const uint32_t matrix_height =  gf16_matrix_bytes_per_column(n_rows1);
    const uint32_t matrix_height_x = matrix_height -  1;

    for (i = 0; i < n_rows1; i++) {
        for (j = 0; j < n_cols2; j++) {
            entry_i_j = 0;

            for (k = 0; k < n_cols1; k++) {
                entry_i_k = gf16_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf16_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16_mul(entry_i_k, entry_k_j);
            }

            gf16_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }

    if (n_rows1 & 1) {
        for (i = 0; i < n_cols2; i++) {
            result[i * matrix_height + matrix_height_x] &= 0x0f;
        }
    }
}

/// B = A^T
/// \param B[out]: ncols x nrows
/// \param A[in]: nrows x ncols
/// \param nrows[in]
/// \param ncols[in]
static inline
void gf16_matrix_transpose(uint8_t *B,
                           const uint8_t *const A,
				           const size_t nrows,
                           const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
            const ff_t a = gf16_matrix_get(A, nrows, i, j);
            gf16_matrix_set(B, ncols, j, i, a);
		}
	}
}

/// \param out[out]: = in
/// \param in[in]:
/// \param nrows[in]
/// \param ncols[in]
static inline
void gf16_matrix_copy(ff_t *out,
                      const ff_t *in,
                      const uint32_t nrows,
                      const uint32_t ncols) {
    memcpy(out, in, gf16_matrix_bytes_size(nrows, ncols));
}

/// matrix1 = matrix2 + matrix3
/// \param matrix1[out]
/// \param matrix2[in]
/// \param matrix3[in]
/// \param nrows[in]
/// \param ncols[in]
static inline
void gf16_matrix_add(ff_t *matrix1,
                     const ff_t *matrix2,
                     const ff_t *matrix3,
                     const uint32_t nrows,
                     const uint32_t ncols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(nrows, ncols);
    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] = matrix2[i] ^ matrix3[i];
    }
}

/// matrix1 += scalar * matrix2
/// \param matrix1[in/out]
/// \param scalar[in]
/// \param matrix2[in]
/// \param nrows[in]
/// \param ncols[in]
static inline
void gf16_matrix_add_scalar(ff_t *matrix1,
                            const ff_t scalar,
                            const ff_t *matrix2,
                            const uint32_t nrows,
                            const uint32_t ncols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(nrows, ncols);
    const uint8_t s = (scalar & 0xf) ^ ((scalar & 0xf) << 4);
    for (uint32_t i = 0; i < n_bytes; i++) {
        matrix1[i] ^= gf16v_mul(matrix2[i], s);
    }
}

/// result += matrix1 * matrix2, column major
/// \param result[in/out]: n_rows1 x n_cols2
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
/// \param n_rows1
/// \param n_cols1
/// \param n_cols2
static inline
void gf16_matrix_add_mul(ff_t *result,
                         const ff_t *matrix1,
                         const ff_t *matrix2,
                         const uint32_t n_rows1,
                         const uint32_t n_cols1,
                         const uint32_t n_cols2) {
    for (uint32_t i = 0; i < n_rows1; i++) {
        for (uint32_t j = 0; j < n_cols2; j++) {
            ff_t entry_i_j = gf16_matrix_get(result, n_rows1, i, j);
            for (uint32_t k = 0; k < n_cols1; k++) {
                const ff_t entry_i_k = gf16_matrix_get(matrix1, n_rows1, i, k);
                const ff_t entry_k_j = gf16_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16_mul(entry_i_k, entry_k_j);
            }

            gf16_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}

/// TODO: add all of these functions to `cryptanalysislib`
/// org source: https://github.com/PQCMayo/MAYO-C/blob/f2aab8ab3a0052272905bdfb57048c150938b8e5/src/mayo.c#L102
/// NOTE: A is a col-major input matrix
/// NOTE: B is a col-major output matrix
/// NOTE: stride in bytes
static inline void gf16_transpose_16x16(uint8_t *B,
                                 const uint8_t *A,
                                 const uint32_t stride){
    static const uint64_t even_nibbles = 0x0f0f0f0f0f0f0f0f;
    static const uint64_t even_bytes   = 0x00ff00ff00ff00ff;
    static const uint64_t even_2bytes  = 0x0000ffff0000ffff;
    static const uint64_t even_half    = 0x00000000ffffffff;

    // read into tmp registers
    uint64_t M[16];
    const uint64_t *M2 = (const uint64_t *)A;
    for (size_t i = 0; i < 16; i++) {
        M[i] = M2[i];
    }
    for (size_t i = 0; i < 16; i+=2) {
        const uint64_t t = ((M[i] >> 4 ) ^ M[i+1]) & even_nibbles;
        M[i  ] ^= t << 4;
        M[i+1] ^= t;
    }
    for (size_t i = 0; i < 16; i+=4) {
        const uint64_t t0 = ((M[i  ] >> 8) ^ M[i+2]) & even_bytes;
        const uint64_t t1 = ((M[i+1] >> 8) ^ M[i+3]) & even_bytes;
        M[i  ] ^= (t0 << 8);
        M[i+1] ^= (t1 << 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }

    for (size_t i = 0; i < 4; i++) {
        const uint64_t t0 = ((M[i  ] >> 16) ^ M[i+ 4]) & even_2bytes;
        const uint64_t t1 = ((M[i+8] >> 16) ^ M[i+12]) & even_2bytes;
        M[i   ] ^= t0 << 16;
        M[i+ 8] ^= t1 << 16;
        M[i+ 4] ^= t0;
        M[i+12] ^= t1;
    }

    for (size_t i = 0; i < 8; i++) {
        const uint64_t t = ((M[i]>>32) ^ M[i+8]) & even_half;
        M[i  ] ^= t << 32;
        M[i+8] ^= t;
    }

    // write back
    for (uint32_t i = 0; i < 16; i++) {
        *((uint64_t *)(B + i*stride)) = M[i];
    }
}

// TODO stride
static inline void gf16_transpose_64x64(uint8_t *B,
                                 const uint8_t *const A,
                                 const uint32_t stride) {
    gf16_transpose_16x16(B,      A,      32);
    gf16_transpose_16x16(B+ 512, A+ 8,   32);
    gf16_transpose_16x16(B+1024, A+16,   32);
    gf16_transpose_16x16(B+1536, A+24,   32);
    gf16_transpose_16x16(B+  8,  A+512,  32);
    gf16_transpose_16x16(B+ 520, A+520,  32);
    gf16_transpose_16x16(B+1032, A+528,  32);
    gf16_transpose_16x16(B+1544, A+536,  32);
    gf16_transpose_16x16(B+  16, A+1024, 32);
    gf16_transpose_16x16(B+ 528, A+1032, 32);
    gf16_transpose_16x16(B+1040, A+1040, 32);
    gf16_transpose_16x16(B+1552, A+1048, 32);
    gf16_transpose_16x16(B+  24, A+1536, 32);
    gf16_transpose_16x16(B+ 536, A+1544, 32);
    gf16_transpose_16x16(B+1048, A+1552, 32);
    gf16_transpose_16x16(B+1560, A+1560, 32);
}

#ifdef USE_AVX2
// NOTE: stride in bytes
static inline void gf16_transpose_32x32_avx2(uint8_t *B,
                                      const uint8_t *const A,
                                      const uint32_t stride) {
    __m128i M[32];
    const __m128i mask1 = _mm_set1_epi8  (0x0F),
                  mask2 = _mm_set1_epi16 (0x00FF),
                  mask3 = _mm_set1_epi32 (0x0000FFFF),
                  mask4 = _mm_set1_epi64x(0x00000000FFFFFFFF),
                  mask5 = _mm_setr_epi64((__m64)0xFFFFFFFFFFFFFFFF, (__m64)0ul);

    for (uint32_t i = 0; i < 32; i++) {
        M[i] = _mm_loadu_si128((__m128i *)(A+i*16));
    }

    for (uint32_t i = 0; i < 32; i+=2) {
        const __m128i t = (_mm_srli_epi64(M[i], 4) ^ M[i+1]) & mask1;
        M[i+0] ^= _mm_slli_epi64(t, 4);
        M[i+1] ^= t;
    }

    for (uint32_t i = 0; i < 32; i+=4) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 8) ^ M[i+2]) & mask2;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 8) ^ M[i+3]) & mask2;
        M[i+0] ^= _mm_slli_epi64(t0, 8);
        M[i+1] ^= _mm_slli_epi64(t1, 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }

    for (uint32_t i = 0; i < 32; i+=8) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 16) ^ M[i+4]) & mask3;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 16) ^ M[i+5]) & mask3;
        const __m128i t2 = (_mm_srli_epi64(M[i+2], 16) ^ M[i+6]) & mask3;
        const __m128i t3 = (_mm_srli_epi64(M[i+3], 16) ^ M[i+7]) & mask3;
        M[i+0] ^= _mm_slli_epi64(t0, 16);
        M[i+1] ^= _mm_slli_epi64(t1, 16);
        M[i+2] ^= _mm_slli_epi64(t2, 16);
        M[i+3] ^= _mm_slli_epi64(t3, 16);
        M[i+4] ^= t0;
        M[i+5] ^= t1;
        M[i+6] ^= t2;
        M[i+7] ^= t3;
    }

    for (uint32_t i = 0; i < 32; i+=16) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 32) ^ M[i+ 8]) & mask4;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 32) ^ M[i+ 9]) & mask4;
        const __m128i t2 = (_mm_srli_epi64(M[i+2], 32) ^ M[i+10]) & mask4;
        const __m128i t3 = (_mm_srli_epi64(M[i+3], 32) ^ M[i+11]) & mask4;
        const __m128i t4 = (_mm_srli_epi64(M[i+4], 32) ^ M[i+12]) & mask4;
        const __m128i t5 = (_mm_srli_epi64(M[i+5], 32) ^ M[i+13]) & mask4;
        const __m128i t6 = (_mm_srli_epi64(M[i+6], 32) ^ M[i+14]) & mask4;
        const __m128i t7 = (_mm_srli_epi64(M[i+7], 32) ^ M[i+15]) & mask4;
        M[i+ 0] ^= _mm_slli_epi64(t0, 32);
        M[i+ 1] ^= _mm_slli_epi64(t1, 32);
        M[i+ 2] ^= _mm_slli_epi64(t2, 32);
        M[i+ 3] ^= _mm_slli_epi64(t3, 32);
        M[i+ 4] ^= _mm_slli_epi64(t4, 32);
        M[i+ 5] ^= _mm_slli_epi64(t5, 32);
        M[i+ 6] ^= _mm_slli_epi64(t6, 32);
        M[i+ 7] ^= _mm_slli_epi64(t7, 32);
        M[i+ 8] ^= t0;
        M[i+ 9] ^= t1;
        M[i+10] ^= t2;
        M[i+11] ^= t3;
        M[i+12] ^= t4;
        M[i+13] ^= t5;
        M[i+14] ^= t6;
        M[i+15] ^= t7;
    }

    for (uint32_t i = 0; i < 16; i++) {
        const __m128i t0 = (_mm_srli_si128(M[i+ 0], 8) ^ M[i+16]) & mask5;
        M[i   ] ^= _mm_slli_si128(t0, 8);
        M[i+16] ^= t0;
    }

    // write out
    for (uint32_t i = 0; i < 32; i++) {
        _mm_storeu_si128((__m128i *)(B + i*stride), M[i]);
    }
}

// TODO add to cryptanalysislib
// NOTE: stride in bytes
static inline void gf16_transpose_64x64_avx2(uint8_t *B,
                                      const uint8_t *const A,
                                      const uint32_t src_stride,
                                      const uint32_t dst_stride) {
    __m256i M[64];
    const __m256i mask1 = _mm256_set1_epi8  (0x0F),
                  mask2 = _mm256_set1_epi16 (0x00FF),
                  mask3 = _mm256_set1_epi32 (0x0000FFFF),
                  mask4 = _mm256_set1_epi64x(0x00000000FFFFFFFF),
                  mask5 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0),
                  mask6 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0);
    for (uint32_t i = 0; i < 64; i++) {
        M[i] = _mm256_loadu_si256((__m256i *)(A+i*src_stride));
    }

    for (uint32_t i = 0; i < 64; i+=2) {
        const __m256i t = (_mm256_srli_epi64(M[i], 4) ^ M[i+1]) & mask1;
        M[i+0] ^= _mm256_slli_epi64(t, 4);
        M[i+1] ^= t;
    }

    for (uint32_t i = 0; i < 64; i+=4) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 8) ^ M[i+2]) & mask2;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 8) ^ M[i+3]) & mask2;
        M[i+0] ^= _mm256_slli_epi64(t0, 8);
        M[i+1] ^= _mm256_slli_epi64(t1, 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }

    for (uint32_t i = 0; i < 64; i+=8) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 16) ^ M[i+4]) & mask3;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 16) ^ M[i+5]) & mask3;
        const __m256i t2 = (_mm256_srli_epi64(M[i+2], 16) ^ M[i+6]) & mask3;
        const __m256i t3 = (_mm256_srli_epi64(M[i+3], 16) ^ M[i+7]) & mask3;
        M[i+0] ^= _mm256_slli_epi64(t0, 16);
        M[i+1] ^= _mm256_slli_epi64(t1, 16);
        M[i+2] ^= _mm256_slli_epi64(t2, 16);
        M[i+3] ^= _mm256_slli_epi64(t3, 16);
        M[i+4] ^= t0;
        M[i+5] ^= t1;
        M[i+6] ^= t2;
        M[i+7] ^= t3;
    }

    for (uint32_t i = 0; i < 64; i+=16) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 32) ^ M[i+ 8]) & mask4;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 32) ^ M[i+ 9]) & mask4;
        const __m256i t2 = (_mm256_srli_epi64(M[i+2], 32) ^ M[i+10]) & mask4;
        const __m256i t3 = (_mm256_srli_epi64(M[i+3], 32) ^ M[i+11]) & mask4;
        const __m256i t4 = (_mm256_srli_epi64(M[i+4], 32) ^ M[i+12]) & mask4;
        const __m256i t5 = (_mm256_srli_epi64(M[i+5], 32) ^ M[i+13]) & mask4;
        const __m256i t6 = (_mm256_srli_epi64(M[i+6], 32) ^ M[i+14]) & mask4;
        const __m256i t7 = (_mm256_srli_epi64(M[i+7], 32) ^ M[i+15]) & mask4;
        M[i+ 0] ^= _mm256_slli_epi64(t0, 32);
        M[i+ 1] ^= _mm256_slli_epi64(t1, 32);
        M[i+ 2] ^= _mm256_slli_epi64(t2, 32);
        M[i+ 3] ^= _mm256_slli_epi64(t3, 32);
        M[i+ 4] ^= _mm256_slli_epi64(t4, 32);
        M[i+ 5] ^= _mm256_slli_epi64(t5, 32);
        M[i+ 6] ^= _mm256_slli_epi64(t6, 32);
        M[i+ 7] ^= _mm256_slli_epi64(t7, 32);
        M[i+ 8] ^= t0;
        M[i+ 9] ^= t1;
        M[i+10] ^= t2;
        M[i+11] ^= t3;
        M[i+12] ^= t4;
        M[i+13] ^= t5;
        M[i+14] ^= t6;
        M[i+15] ^= t7;
    }

    for (uint32_t i = 0; i < 16; i++) {
        const __m256i t0 = (_mm256_srli_si256(M[i+ 0], 8) ^ M[i+16]) & mask5;
        const __m256i t1 = (_mm256_srli_si256(M[i+32], 8) ^ M[i+48]) & mask5;
        M[i   ] ^= _mm256_slli_si256(t0, 8);
        M[i+32] ^= _mm256_slli_si256(t1, 8);
        M[i+16] ^= t0;
        M[i+48] ^= t1;
    }

    for (uint32_t i = 0; i < 32; i++) {
        const __m256i t = (_mm256_permute2x128_si256(M[i+0], M[i+0], 0b10000001) ^ M[i+32]) & mask6;
        M[i   ] ^= _mm256_permute2x128_si256(t, t, 0b01000); //
        M[i+32] ^= t;
    }

    // write out
    for (uint32_t i = 0; i < 64; i++) {
        _mm256_storeu_si256((__m256i *)(B + i*dst_stride), M[i]);
    }
}

#elif defined(USE_NEON)

/// source original from: https://github.com/pqov/pqov-paper
/// \param dest[out]
/// \param dest_vec_len[in]
/// \param src[in]
/// \param src_vec_len[in]
static inline  void byte_transpose_16x16_neon(uint8_t *dest,
                                              unsigned dest_vec_len,
                                              const uint8_t *src,
                                              unsigned src_vec_len ) {
   uint8x16_t r0  = vtrn1q_u64(vld1q_u8(src+0*src_vec_len), vld1q_u8(src+8*src_vec_len));
   uint8x16_t r8  = vtrn2q_u64(vld1q_u8(src+0*src_vec_len), vld1q_u8(src+8*src_vec_len));
   uint8x16_t r1  = vtrn1q_u64(vld1q_u8(src+1*src_vec_len), vld1q_u8(src+9*src_vec_len));
   uint8x16_t r9  = vtrn2q_u64(vld1q_u8(src+1*src_vec_len), vld1q_u8(src+9*src_vec_len));
   uint8x16_t r2  = vtrn1q_u64(vld1q_u8(src+2*src_vec_len), vld1q_u8(src+10*src_vec_len));
   uint8x16_t r10 = vtrn2q_u64(vld1q_u8(src+2*src_vec_len), vld1q_u8(src+10*src_vec_len));
   uint8x16_t r3  = vtrn1q_u64(vld1q_u8(src+3*src_vec_len), vld1q_u8(src+11*src_vec_len));
   uint8x16_t r11 = vtrn2q_u64(vld1q_u8(src+3*src_vec_len), vld1q_u8(src+11*src_vec_len));
   uint8x16_t r4  = vtrn1q_u64(vld1q_u8(src+4*src_vec_len), vld1q_u8(src+12*src_vec_len));
   uint8x16_t r12 = vtrn2q_u64(vld1q_u8(src+4*src_vec_len), vld1q_u8(src+12*src_vec_len));
   uint8x16_t r5  = vtrn1q_u64(vld1q_u8(src+5*src_vec_len), vld1q_u8(src+13*src_vec_len));
   uint8x16_t r13 = vtrn2q_u64(vld1q_u8(src+5*src_vec_len), vld1q_u8(src+13*src_vec_len));
   uint8x16_t r6  = vtrn1q_u64(vld1q_u8(src+6*src_vec_len), vld1q_u8(src+14*src_vec_len));
   uint8x16_t r14 = vtrn2q_u64(vld1q_u8(src+6*src_vec_len), vld1q_u8(src+14*src_vec_len));
   uint8x16_t r7  = vtrn1q_u64(vld1q_u8(src+7*src_vec_len), vld1q_u8(src+15*src_vec_len));
   uint8x16_t r15 = vtrn2q_u64(vld1q_u8(src+7*src_vec_len), vld1q_u8(src+15*src_vec_len));

   uint8x16_t s0  = vtrn1q_u32(r0,  r4);
   uint8x16_t s4  = vtrn2q_u32(r0,  r4);
   uint8x16_t s1  = vtrn1q_u32(r1,  r5);
   uint8x16_t s5  = vtrn2q_u32(r1,  r5);
   uint8x16_t s2  = vtrn1q_u32(r2,  r6);
   uint8x16_t s6  = vtrn2q_u32(r2,  r6);
   uint8x16_t s3  = vtrn1q_u32(r3,  r7);
   uint8x16_t s7  = vtrn2q_u32(r3,  r7);
   uint8x16_t s8  = vtrn1q_u32(r8,  r12);
   uint8x16_t s12 = vtrn2q_u32(r8,  r12);
   uint8x16_t s9  = vtrn1q_u32(r9,  r13);
   uint8x16_t s13 = vtrn2q_u32(r9,  r13);
   uint8x16_t s10 = vtrn1q_u32(r10, r14);
   uint8x16_t s14 = vtrn2q_u32(r10, r14);
   uint8x16_t s11 = vtrn1q_u32(r11, r15);
   uint8x16_t s15 = vtrn2q_u32(r11, r15);

   r0 = vtrn1q_u16( s0 , s2 );
   r2 = vtrn2q_u16( s0 , s2 );
   r1 = vtrn1q_u16( s1 , s3 );
   r3 = vtrn2q_u16( s1 , s3 );
   r4 = vtrn1q_u16( s4 , s6 );
   r6 = vtrn2q_u16( s4 , s6 );
   r5 = vtrn1q_u16( s5 , s7 );
   r7 = vtrn2q_u16( s5 , s7 );
   r8 = vtrn1q_u16( s8 , s10 );
   r10 = vtrn2q_u16( s8 , s10 );
   r9 = vtrn1q_u16( s9 , s11 );
   r11 = vtrn2q_u16( s9 , s11 );
   r12 = vtrn1q_u16( s12 , s14 );
   r14 = vtrn2q_u16( s12 , s14 );
   r13 = vtrn1q_u16( s13 , s15 );
   r15 = vtrn2q_u16( s13 , s15 );

   vst1q_u8( dest+0*dest_vec_len , vtrn1q_u8( r0 , r1 ) );
   vst1q_u8( dest+1*dest_vec_len , vtrn2q_u8( r0 , r1 ) );
   vst1q_u8( dest+2*dest_vec_len , vtrn1q_u8( r2 , r3 ) );
   vst1q_u8( dest+3*dest_vec_len , vtrn2q_u8( r2 , r3 ) );
   vst1q_u8( dest+4*dest_vec_len , vtrn1q_u8( r4 , r5 ) );
   vst1q_u8( dest+5*dest_vec_len , vtrn2q_u8( r4 , r5 ) );
   vst1q_u8( dest+6*dest_vec_len , vtrn1q_u8( r6 , r7 ) );
   vst1q_u8( dest+7*dest_vec_len , vtrn2q_u8( r6 , r7 ) );
   vst1q_u8( dest+8*dest_vec_len , vtrn1q_u8( r8 , r9 ) );
   vst1q_u8( dest+9*dest_vec_len , vtrn2q_u8( r8 , r9 ) );
   vst1q_u8( dest+10*dest_vec_len , vtrn1q_u8( r10 , r11 ) );
   vst1q_u8( dest+11*dest_vec_len , vtrn2q_u8( r10 , r11 ) );
   vst1q_u8( dest+12*dest_vec_len , vtrn1q_u8( r12 , r13 ) );
   vst1q_u8( dest+13*dest_vec_len , vtrn2q_u8( r12 , r13 ) );
   vst1q_u8( dest+14*dest_vec_len , vtrn1q_u8( r14 , r15 ) );
   vst1q_u8( dest+15*dest_vec_len , vtrn2q_u8( r14 , r15 ) );
}
#endif

/// B = A^T
/// uses the 64x64 AVX2 kernel for all full 64x64 blocks, and a generic
/// element-wise approach for the remaining rows/columns.
/// \param B[out]: ncols x nrows
/// \param A[in]: nrows x ncols
static inline
void gf16_matrix_transpose_opt(uint8_t *B,
                               const uint8_t *const A,
                               const uint32_t nrows,
                               const uint32_t ncols) {
    const uint32_t bsize = 64u;
    // number of rows/cols handled by the 64x64 kernel
    uint32_t R = 0, C = 0;
#ifdef USE_AVX2
    R = (nrows / bsize) * bsize;
    C = (ncols / bsize) * bsize;
    const uint32_t src_stride = gf16_matrix_bytes_per_column(nrows);
    const uint32_t dst_stride = gf16_matrix_bytes_per_column(ncols);
    for (uint32_t r = 0; r < R; r += bsize) {
        for (uint32_t c = 0; c < C; c += bsize) {
            const uint8_t *src_origin = A + c*src_stride + r/2;
                  uint8_t *dst_origin = B + r*dst_stride + c/2;
            gf16_transpose_64x64_avx2(dst_origin, src_origin, src_stride, dst_stride);
        }
    }
#endif

    // the last columns of A for all rows covered by the kernel
    for (uint32_t i = 0; i < R; i++) {
        for (uint32_t j = C; j < ncols; j++) {
            const ff_t a = gf16_matrix_get(A, nrows, i, j);
            gf16_matrix_set(B, ncols, j, i, a);
        }
    }

    // the last rows of A
    for (uint32_t i = R; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const ff_t a = gf16_matrix_get(A, nrows, i, j);
            gf16_matrix_set(B, ncols, j, i, a);
        }
    }
}

static inline
void gf16_swap_cols(uint8_t *A,
                    const uint32_t nrows,
                    const uint32_t i,
                    const uint32_t j) {
    if (i == j) { return; }
    const uint32_t bytes = (nrows+1)/2;

    uint32_t k = 0;
#if defined(USE_AVX2)
    for (;k+32 <= bytes; k+=32) {
        const __m256i a = _mm256_loadu_si256((__m256i *)(A +i*bytes + k));
        const __m256i b = _mm256_loadu_si256((__m256i *)(A +j*bytes + k));
        _mm256_storeu_si256((__m256i *)(A +j*bytes + k), a);
        _mm256_storeu_si256((__m256i *)(A +i*bytes + k), b);
    }
#endif
    for (; k < bytes; k++) {
        const uint8_t tmp = A[i*bytes + k];
        A[i*bytes + k] = A[j*bytes + k];
        A[j*bytes + k] = tmp;
    }
}

static inline
void gf16_swap_rows(uint8_t *A,
                    const uint32_t nrows,
                    const uint32_t ncols,
                    const uint32_t i,
                    const uint32_t j) {
    if (i == j) { return; }
    for (uint32_t k = 0; k < ncols; k++) {
        const uint8_t tmp1 = gf16_matrix_get(A, nrows, i, k);
        const uint8_t tmp2 = gf16_matrix_get(A, nrows, j, k);
        gf16_matrix_set(A, nrows, i, k, tmp2);
        gf16_matrix_set(A, nrows, j, k, tmp1);
    }
}

/// NOTE: assumes that A is stored row-major (= transposed col-major),
///     each row consists of ceil(ncols/2) bytes.
static inline
void gf16_swap_rows_transpose(uint8_t *A,
                               const uint32_t nrows,
                               const uint32_t ncols,
                               const uint32_t i,
                               const uint32_t j) {
    (void)nrows;
    if (i == j) { return; }

    const uint32_t bytes = (ncols + 1)/2;
    uint32_t k = 0;
#if defined(USE_AVX2)
    for (;k+32 <= bytes; k+=32) {
        const __m256i a = _mm256_loadu_si256((__m256i *)(A +i*bytes + k));
        const __m256i b = _mm256_loadu_si256((__m256i *)(A +j*bytes + k));
        _mm256_storeu_si256((__m256i *)(A +j*bytes + k), a);
        _mm256_storeu_si256((__m256i *)(A +i*bytes + k), b);
    }
#endif
    for (;k < bytes; k++) {
        SWAP(A[i*bytes + k], A[j*bytes + k]);
    }
}

/// NOTE: assumes that A is col-major
/// A[row] *= c;
static inline
void gf16_solve_row(uint8_t *A,
                    const uint32_t nrows,
                    const uint32_t ncols,
                    const uint32_t row,
                    const uint8_t c) {
    for (uint32_t i = 0; i < ncols; i++) {
        uint8_t d = gf16_matrix_get(A, nrows, row, i);
        d = gf16_mul(c, d);
        gf16_matrix_set(A, nrows, row, i, d);
    }
}


/// NOTE: assumes that A is row-major, each row consists of ceil(ncols/2) bytes
/// A[row] *= c;
static inline
void gf16_solve_row_transpose(uint8_t *A,
                              const uint32_t nrows,
                              const uint32_t ncols,
                              const uint32_t row,
                              const uint8_t c) {
    (void)nrows;
    if (c == 1) { return; }
    const uint32_t bytes = (ncols + 1)/2;
    uint32_t i = 0;
#if defined(USE_AVX2)
    const __m256i multab_l = gf16v_tbl32_multab(c);
    const __m256i multab_h = _mm256_slli_epi16(multab_l, 4 );
    const __m256i f = _mm256_set1_epi8(0xf);

    for (;i+32 <= bytes; i+=32) {
        const __m256i a = _mm256_loadu_si256((__m256i *)(A +row*bytes + i));
        const __m256i d = linear_transform_8x8_256b(multab_l, multab_h, a, f);
        _mm256_storeu_si256((__m256i *)(A +row*bytes + i), d);
    }
#endif
    const uint8_t cc = (c & 0xf) ^ ((c & 0xf) << 4);
    for (; i < bytes; i++) {
        A[row*bytes + i] = gf16v_mul(A[row*bytes + i], cc);
    }
}


/// returns 1 on error, 0 else
/// solves Ax=b
static inline
uint32_t gf16_solve(uint8_t *A,
                    uint8_t *b,
                    const uint32_t nrows,
                    const uint32_t ncols) {
    for (uint32_t row = 0; row < nrows; row++) {
        // find pivot
        uint32_t i = row;
        for (; i < nrows; i++) {
            if (gf16_matrix_get(A, nrows, i, row)) {
                break;
            }
        }
        if (i == nrows) {return 1;}
        const uint8_t c = gf16_inv(gf16_matrix_get(A, nrows, i, row));
        gf16_swap_rows(A, nrows, ncols, i, row);
        if (i != row) {
            const gf16 t1 = gf16_vector_get(b, i);
            gf16_vector_set(b, i, gf16_vector_get(b, row));
            gf16_vector_set(b, row, t1);
        }
        /// solve current pivot row
        gf16_solve_row(A, nrows, ncols, row, c);
        gf16_vector_set(b, row, gf16_mul(gf16_vector_get(b, row), c));

        // solve remaining rows below
        for (uint32_t j = row+1; j < nrows; j++) {
            const uint8_t d = gf16_matrix_get(A, nrows, j, row);
            gf16_sub_row(A, nrows, ncols, j, row, d);
            gf16_vector_set(b, j, gf16_vector_get(b, j) ^ gf16_mul(gf16_vector_get(b, row), d));
        }

        // solve remaining rows above
        for (uint32_t j = 0; j < row; j++) {
            const uint8_t d = gf16_matrix_get(A, nrows, j, row);
            gf16_sub_row(A, nrows, ncols, j, row, d);
            gf16_vector_set(b, j, gf16_vector_get(b, j) ^ gf16_mul(gf16_vector_get(b, row), d));
        }
    }
    return 0;
}

/// solves Bx=b via Gauss-Jordan elimination on the transposed (= row-major)
/// matrix. On return B is in reduced row echelon form.
/// \param B[in/out]: nrows x ncols matrix, col-major
/// \param A[tmp]: tmp matrix, must be size of transposed of B
/// \param b[in/out]: vector of nrows elements
/// \param nrows
/// \param ncols
/// \return number of pivot rows found (= nrows, if B has full row rank and
///     the left nrows x nrows submatrix is invertible)
static inline
uint32_t gf16_solve_transpose(uint8_t *B,
                              uint8_t *A,
                              uint8_t *b,
                              const uint32_t nrows,
                              const uint32_t ncols) {
    // A is ncols x nrows col-major, i.e. B in row-major form:
    // A(c, r) = B(r, c), row r of B starts at byte r*ceil(ncols/2) of A.
    gf16_matrix_transpose_opt(A, B, nrows, ncols);

    uint32_t row = 0;
    for (; (row < nrows) && (row < ncols); row++) {
        // find pivot: B(i, row) != 0 for i >= row
        uint32_t i = row;
        for (; i < nrows; i++) {
            if (gf16_matrix_get(A, ncols, row, i)) {
                break;
            }
        }

        if (i == nrows) { break; }

        const uint8_t c = gf16_inv(gf16_matrix_get(A, ncols, row, i));
        gf16_swap_rows_transpose(A, nrows, ncols, i, row);
        if (i != row) {
            const gf16 t1 = gf16_vector_get(b, i);
            gf16_vector_set(b, i, gf16_vector_get(b, row));
            gf16_vector_set(b, row, t1);
        }

        /// solve current pivot row
        gf16_solve_row_transpose(A, nrows, ncols, row, c);
        const gf16 t = gf16_vector_get(b, row);
        gf16_vector_set(b, row, gf16_mul(t, c));

        // eliminate all other rows
        for (uint32_t j = 0; j < nrows; j++) {
            if (j == row) { continue; }
            const uint8_t d = gf16_matrix_get(A, ncols, row, j);
            gf16_sub_row_transpose(A, nrows, ncols, j, row, d);
            gf16_vector_set(b, j, gf16_vector_get(b, j) ^ gf16_mul(gf16_vector_get(b, row), d));
        }
    }

    gf16_matrix_transpose_opt(B, A, ncols, nrows);
    return row;
}

#ifdef USE_AVX2
/// org code from:https://github.com/PQCMayo/MAYO-C/blob/nibbling-mayo/src/AVX2/echelon_form.h
/// put matrix in row echelon form with ones on first nonzero entries in constant time
/// NOTE: assumes the input matrix in row major form
/// NOTE:
static inline void row_echelon_form_compressed(unsigned char *A,
                                               const int _nrows,
                                               const int _ncols) {
    (void) _nrows;
    (void) _ncols;
#ifndef NROWS
#define NROWS 32
#endif
#ifndef NCOLS
#define NCOLS 32
#endif

#define AVX_REGS_PER_ROW ((NCOLS + 63) / 64)
#define MAX_COLS (AVX_REGS_PER_ROW * 64)

    __m256i _pivot_row[AVX_REGS_PER_ROW];
    __m256i A_avx     [AVX_REGS_PER_ROW * NROWS];

    unsigned char* pivot_row_bytes = (unsigned char*) _pivot_row;
    unsigned char* A_bytes = (unsigned char*) A_avx;

    // load A in the tail of AVX2 registers
    for (uint32_t i = 0; i < NROWS; i++) {
        for (uint32_t j = 0; j < (NCOLS/2); j++) {
            A_bytes[i*(MAX_COLS/2) + ((MAX_COLS/2) - (NCOLS/2)) + j] = A[i*(NCOLS/2) + j ];
        }
    }
    const __m256i xf = _mm256_set1_epi8(0xf);

    // pivot row is secret, pivot col is not
    unsigned char inverse;
    int pivot_row = 0;
    int pivot_col = MAX(MAX_COLS - NCOLS, 0);
    for (; pivot_col < MAX_COLS-256; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-192; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-128; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-64; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }

    // write the matrix A back
    for (uint32_t i = 0; i < NROWS; i++) {
        for (uint32_t j = 0; j < (NCOLS/2); j++) {
            A[i * (NCOLS/2) + j] = A_bytes[i*AVX_REGS_PER_ROW*32 + ((MAX_COLS/2) - (NCOLS/2)) + j];
        }
    }
#undef AVX_REGS_PER_ROW
#undef MAX_COLS
}

/// org code from:https://github.com/PQCMayo/MAYO-C/blob/nibbling-mayo/src/AVX2/echelon_form.h
/// put matrix in row echelon form with ones on first nonzero entries in constant time
/// NOTE: assumes the input matrix in row major form
/// NOTE:
static inline void row_echelon_form(unsigned char *A,
                                    const uint32_t _nrows,
                                    const uint32_t _ncols) {
    (void) _nrows;
    (void) _ncols;
#define AVX_REGS_PER_ROW ((NCOLS + 31) / 32)
#define MAX_COLS (AVX_REGS_PER_ROW * 32)

    __m256i _pivot_row[AVX_REGS_PER_ROW];
    __m256i A_avx[AVX_REGS_PER_ROW* NROWS];

    unsigned char* pivot_row_bytes = (unsigned char*) _pivot_row;
    unsigned char* A_bytes = (unsigned char*) A_avx;

    // load A in the tail of AVX2 registers
    for (int i = 0; i < NROWS; i++) {
        for (int j = 0; j < NCOLS; j++)
        {
            A_bytes[i*MAX_COLS + (MAX_COLS - NCOLS) + j] = A[ i*NCOLS + j ];
        }
    }

    // pivot row is secret, pivot col is not
    unsigned char inverse;
    int pivot_row = 0;
    int pivot_col = MAX(MAX_COLS - NCOLS,0);
    for (; pivot_col < MAX_COLS-128; pivot_col++) {
#include "echelon_form_loop.h"
    }
    for (; pivot_col < MAX_COLS-96; pivot_col++) {
#include "echelon_form_loop.h"
    }
    for (; pivot_col < MAX_COLS-64; pivot_col++) {
#include "echelon_form_loop.h"
    }
    for (; pivot_col < MAX_COLS-32; pivot_col++) {
#include "echelon_form_loop.h"
    }
    for (; pivot_col < MAX_COLS; pivot_col++) {
#include "echelon_form_loop.h"
    }

    // write the matrix A back
    for (int i = 0; i < NROWS; i++) {
        for (int j = 0; j < NCOLS; j++) {
            A[i * NCOLS + j] = A_bytes[i*AVX_REGS_PER_ROW*32 + (MAX_COLS - NCOLS) + j];
        }
    }
#undef AVX_REGS_PER_ROW
#undef MAX_COLS
}

/// src:
/// c = a * b;
static inline void gf16mat_prod_16x16_avx2(uint8_t * c,
                                    const uint8_t * matA,
                                    const uint8_t * b ) {
    __m256i multabs[16];
    uint8_t temp[16] __attribute__((aligned(16)));
    for(uint32_t i=0;i<8;i++) { temp[i]=b[i]; }
    __m128i x0 = _mm_load_si128((const __m128i*)temp);
    xmmx2_t xx = gf16v_split_16to32_sse2( x0 );
    gf16v_generate_multab_16_avx2(multabs , xx.v0 , 16 );
    mat_simd_16x16(c, matA, multabs);
}





/// around 30-40% faster than the normal approach
/// Implements the full 16x16 matrix-matrix multiplication
/// NOTE: eventhough this implementation assumes 16columns on 16 elements input matrix, its ok to input
///         matrices of dimension 16x15 (e.g. only 15 elements per column).
/// \param c out matrix
/// \param a in matrix
/// \param b in matrix
static inline
void gf16mat_prod_16x16_avx2_wrapper_v2(uint8_t *__restrict c,
                                        const uint8_t *__restrict__ a,
                                        unsigned int height_A,
                                        const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    const uint32_t *b32 = (const uint32_t *) b;
    uint64_t *c64 = (uint64_t *) c;

    const uint32_t mask = 0x0f0f0f0f;
    const __m256i Amask = _mm256_setr_epi32(0,2,4,6,8,10,12,14);
    const __m256i perm  = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);

    __m256i tmp,acc;
    const __m256i Al1 = _mm256_i32gather_epi32((const int *)a32 +  0u, Amask, 4u);
    const __m256i Al2 = _mm256_i32gather_epi32((const int *)a32 + 16u, Amask, 4u);
    const __m256i Ah1 = _mm256_i32gather_epi32((const int *)a32 +  1u, Amask, 4u);
    const __m256i Ah2 = _mm256_i32gather_epi32((const int *)a32 + 17u, Amask, 4u);

    for (uint32_t i = 0; i < height_A; ++i) {
        // load first 4bytes
        uint32_t d11 = b32[2*i+0] & mask;
        uint32_t d12 = (b32[2*i+0] >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); //0b10101010

        // load the next 4 bytes
        uint32_t d21 = b32[2*i+1] & mask;
        uint32_t d22 = (b32[2*i+1] >> 4) & mask;

        __m256i B21 = _mm256_set1_epi32(d21);
        __m256i B22 = _mm256_set1_epi32(d22);
        B21 = _mm256_shuffle_epi8(B21, perm);
        B22 = _mm256_shuffle_epi8(B22, perm);
        const __m256i B2 = _mm256_blend_epi32(B21, B22, 0xaa);


        acc = gf16v_mul_lo_u256(Al1, B1);
        tmp = gf16v_mul_lo_u256(Al2, B2);
        __m256i tmp1 = _mm256_xor_si256(acc, tmp);

        acc = gf16v_mul_lo_u256(Ah1, B1);
        tmp = gf16v_mul_lo_u256(Ah2, B2);
        __m256i tmp2 = _mm256_xor_si256(acc, tmp);

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        c64[i] = _mm256_extract_epi64(ret, 0);
    }
}

/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2` which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
static inline
void gf16mat_prod_le8xle8_avx2_wrapper_v2(uint8_t *__restrict c,
                                          const uint8_t *__restrict__ a,
                                          const unsigned int nr_cols_B,
                                          const unsigned int column_A_bytes,
                                          const unsigned int nr_cols_A,
                                          const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes <= 8);
    assert(nr_cols_A <= 8);

    /// make sure this number is smaller/equal than 4.
    const uint32_t nr_bytes_B_col = nr_cols_A>>1;
    const uint32_t mask = (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const __m256i Amask = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i zero = _mm256_setzero_si256();

    const uint32_t n = nr_cols_A;
    const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 < n ? -1 : 0, 2 < n ? -1 : 0, 3 < n ? -1 : 0, 4 < n ? -1 : 0, 5 < n ? -1 : 0, 6 < n ? -1 : 0, 7 < n ? -1 : 0);
    __m256i tmp1, tmp2, Al1, Ah1 = zero;

    const uint32_t BMask = (1u << (8u*nr_bytes_B_col)) - 1u;
    Al1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 0u, Amask, gather_mask, 4u);
    if (column_A_bytes > 4)
        Ah1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 1u, Amask, gather_mask, 4u);


    uint64_t rdata = 0;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B-1) {
            for (uint32_t i = 0; i < (nr_cols_B * nr_bytes_B_col) % 4; i++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + i] << (i * 8);
            }
        }
         b_data = *((uint32_t *)(b+(nr_bytes_B_col * i))) & BMask;
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16v_mul_lo_u256(Al1, B1);
        if (column_A_bytes > 4)
            tmp2 = gf16v_mul_lo_u256(Ah1, B1);
        else
            tmp2 = zero;

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        rdata = _mm256_extract_epi64(ret, 0);
        if (i < (nr_cols_B-1))
            *((uint64_t *) (c + i * column_A_bytes)) = rdata;
    }


    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}


/// this function differs from `gf16mat_prod_le8xle8_avx2_wrapper_v2`, by allowing the matrix A to have a column length
/// greater than 8.
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
static inline
void gf16mat_prod_gr8xle8_avx2_wrapper(uint8_t *__restrict c,
                                       const uint8_t *__restrict__ a,
                                       const unsigned int nr_cols_B,
                                       const unsigned int column_A_bytes,
                                       const unsigned int nr_cols_A,
                                       const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes > 8);
    assert(column_A_bytes < 12);
    assert(column_A_bytes == 10 || column_A_bytes == 11);

    const uint32_t nr_bytes_B_col = nr_cols_A>>1;
    const uint32_t mask = nr_bytes_B_col%4==0 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    __m256i tmp1, tmp2, tmp3;

    // old and original code
    //const __m256i Ashfl = _mm256_setr_epi32(0, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u, ((uint32_t)(1u)<<(8*off)) - 1u);
    //const __m256i Ashft = _mm256_setr_epi32(0, 8*off, 8*off, 8*off, 8*off, 8*off, 8*off, 8*off);
    //const __m256i Abla = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    //const __m256i A3ma = _mm256_set1_epi32(((uint32_t)(1u)<<(8*(column_A_bytes % 4))) - 1u);

    //__m256i A1 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 0u, Amask, gather_mask, 4u);
    //__m256i A11 = _mm256_sllv_epi32(A1, Ashft);
    //__m256i A12 = _mm256_srli_epi32(A1, 8*(column_A_bytes % 4));

    //__m256i A2 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 1u, Amask, gather_mask, 4u);
    //__m256i A21 = _mm256_sllv_epi32(A2, Ashft);
    //__m256i A22 = _mm256_srli_epi32(A2, 8*(column_A_bytes % 4));

    //__m256i A3 = _mm256_mask_i32gather_epi32(zero, (const int *)a32 + 2u, Amask, gather_mask, 4u);
    //__m256i A31 = _mm256_sllv_epi32(_mm256_and_si256(A3ma, A3), Ashft);
    //__m256i A32 = _mm256_permutevar8x32_epi32(_mm256_srli_epi32(A3, 8*(column_A_bytes % 4)), Abla);

    __m256i Asel, Ashup, Ashdo, Ashfl, Ash, Abla, A1, A2, A3;
    Abla = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    Asel = _mm256_setr_epi32(0, 3, 6, 9, 11, 14, 17, 20);
    Ashup = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
    Ashdo = _mm256_setr_epi32(0, 24, 16, 8, 0, 24, 16, 8);
    Ashfl = _mm256_setr_epi32(0, (1u << 8u) - 1u, (1u << 16u) - 1u, (1u << 24u) - 1u, 0, (1u << 8u) - 1u,(1u << 16u) - 1u, (1u << 24u) - 1u);
    A1 = _mm256_i32gather_epi32((const int *)a32 + 0u, Asel, 4u);
    __m256i A11 = _mm256_sllv_epi32(A1, Ashup);
    __m256i A12 = _mm256_srlv_epi32(A1, Ashdo);

    A2 = _mm256_i32gather_epi32((const int *)a32 + 1u, Asel, 4u);
    __m256i A21 = _mm256_sllv_epi32(A2, Ashup);
    __m256i A22 = _mm256_srlv_epi32(A2, Ashdo);

    A3 = _mm256_i32gather_epi32((const int *)a32 + 2u, Asel, 4u);
    __m256i A31 = _mm256_sllv_epi32(A3, Ashup);
    __m256i A32 = _mm256_permutevar8x32_epi32(A3, Abla);
    A32 = _mm256_srlv_epi32(A32, Ashdo);

    A1 = _mm256_blendv_epi8(A11, A32, Ashfl);
    A2 = _mm256_blendv_epi8(A21, A12, Ashfl);
    A3 = _mm256_blendv_epi8(A31, A22, Ashfl);

    //
    if (column_A_bytes == 10) {
        Asel = _mm256_setr_epi32(0, 2, 5, 7, 10, 12, 15, 17);
        Ash = _mm256_setr_epi32(0, 16, 0, 16, 0, 16, 0, 16);
        Ashfl = _mm256_setr_epi32(0, (-1u)<<16u, 0, (-1u)<<16u, 0, (-1u)<<16u,0, (-1u)<<16u);
        Abla = _mm256_setr_epi32(1, 1, 2, 3, 4, 5, 6, 7);

        A1 = _mm256_i32gather_epi32((const int *)a32 + 0u, Asel, 4u);
        __m256i A11 = _mm256_srlv_epi32(A1, Ash);
        //__m256i A12 = _mm256_sllv_epi32(A1, Ash);

        A2 = _mm256_i32gather_epi32((const int *)a32 + 1u, Asel, 4u);
        __m256i A21 = _mm256_srlv_epi32(A2, Ash);
        __m256i A22 = _mm256_permutevar8x32_epi32(_mm256_sllv_epi32(A2, Ash), Abla);

        A3 = _mm256_i32gather_epi32((const int *)a32 + 2u, Asel, 4u);
        __m256i A31 = _mm256_srlv_epi32(A3, Ash);
        __m256i A32 = _mm256_sllv_epi32(A3, Ash);

        A1 = _mm256_blendv_epi8(A11, A22, Ashfl);
        A2 = _mm256_blendv_epi8(A21, A32, Ashfl);
        A3 = A31;
    }

    uint32_t rdata2 = 0;

    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = *((uint32_t *) (b + (nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16v_mul_lo_u256(A1, B1);
        tmp2 = gf16v_mul_lo_u256(A2, B1);
        tmp3 = gf16v_mul_lo_u256(A3, B1);

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        uint64_t rdata = _mm256_extract_epi64(ret, 0);
        *((uint64_t *) (c + i * column_A_bytes)) = rdata;

        rdata2 = _mm256_extract_epi32(gf16_hadd_avx2_32(tmp3), 0);
        if (i < nr_cols_B - 1) {
            *((uint32_t *) (c + i * column_A_bytes + 8)) = rdata2;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes%4; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + 8 + i] = (uint8_t)(rdata2>>(i*8));
    }
}


/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2` which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
static inline
void gf16mat_prod_le4xle4_avx2_wrapper(uint8_t *__restrict c,
                                       const uint8_t *__restrict__ a,
                                       const unsigned int nr_cols_B,
                                       const unsigned int column_A_bytes,
                                       const unsigned int nr_cols_A,
                                       const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes <= 4);
    assert(nr_cols_A <= 16);
    const __m256i zero = _mm256_setzero_si256();
    __m256i Al1 = zero, Al2 = zero, tmp1, tmp2;

    const uint32_t nr_bytes_B_col = nr_cols_A >> 1;
    const uint32_t mask = nr_bytes_B_col == 4 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    const uint32_t n = ((nr_cols_A + 1) * column_A_bytes) >> 2;

    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                          3,3, 3, 3, 3, 3, 3);
    if (column_A_bytes == 4) {
        Al1 = _mm256_loadu_si256((__m256i *)a);
        if (nr_cols_A > 8) {
            Al2 = _mm256_loadu_si256((__m256i *)(a + 32));
        }
    } else if (column_A_bytes == 3) {
        const __m256i Amask = _mm256_setr_epi32(0, 1, 2, 2, 3, 4, 5, 5);

        const __m256i Ashup = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
        const __m256i Ashdo = _mm256_setr_epi32(24, 16, 8, 8, 24, 16, 8, 8);
        const __m256i Abla = _mm256_setr_epi32(0, 0, 1, 2, 0, 4, 5, 6);
        const __m256i Ashfl = _mm256_setr_epi32(0, (1 << 8) - 1, (1 << 16) - 1, (1u << 24) - 1, 0, (1 << 8) - 1,
                                                (1 << 16) - 1, (1u << 24) - 1);
        const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
                                                      3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
                                                      6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

        Al1 = _mm256_mask_i32gather_epi32(zero, (const int *)(a32 + 0u), Amask, gather_mask, 4u);
        __m256i A11 = _mm256_sllv_epi32(Al1, Ashup);
        __m256i A12 = _mm256_srlv_epi32(Al1, Ashdo);
        __m256i A13 = _mm256_permutevar8x32_epi32(A12, Abla);

        Al1 = _mm256_blendv_epi8(A11, A13, Ashfl);

        if (nr_cols_A > 8) {
            Al2 = _mm256_mask_i32gather_epi32(zero, (const int *)(a32 + 6u), Amask, gather_mask, 4u);
            __m256i A21 = _mm256_sllv_epi32(Al2, Ashup);
            __m256i A22 = _mm256_srlv_epi32(Al2, Ashdo);
            __m256i A23 = _mm256_permutevar8x32_epi32(A22, Abla);

            Al2 = _mm256_blendv_epi8(A21, A23, Ashfl);
        }
    } else if (column_A_bytes == 2) {
        const __m256i Amask = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

        const __m256i Ashdo = _mm256_set1_epi32(8);
        const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
                                                      3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
                                                      6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

        Al1 = _mm256_mask_i32gather_epi32(zero, (const int *) a32 + 0u, Amask, gather_mask, 4u);
        Al1 = _mm256_srlv_epi32(Al1, Ashdo);
    }



    uint32_t rdata = 0;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B - 1) {
            for (uint32_t j = 0; j < (nr_cols_B * nr_bytes_B_col) % 4; j++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + j] << (j * 8);
            }
        }
        b_data = *((uint32_t *)(b+(nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        if (nr_cols_A > 8) {
            b_data = *((uint32_t *)(b+(nr_bytes_B_col * i + 4)));
            uint32_t d11 = b_data & mask;
            uint32_t d12 = (b_data >> 4) & mask;

            __m256i B11 = _mm256_set1_epi32(d11);
            __m256i B12 = _mm256_set1_epi32(d12);
            B11 = _mm256_shuffle_epi8(B11, perm);
            B12 = _mm256_shuffle_epi8(B12, perm);
            const __m256i B2 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

            tmp1 = gf16v_mul_lo_u256(Al1, B1);
            tmp2 = gf16v_mul_lo_u256(Al2, B2);
            tmp1 = _mm256_xor_si256(tmp1, tmp2);
        } else {
            tmp1 = gf16v_mul_lo_u256(Al1, B1);
        }

        rdata = _mm256_extract_epi32(gf16_hadd_avx2_32(tmp1), 0);
        if (i < nr_cols_B - 1) {
            *((uint32_t *) (c + i * column_A_bytes)) = rdata;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}



/// this is an adapted version of `gf16mat_prod_16x16_avx2_wrapper_v2`
/// which allows for smaller input matrices
/// \param c out matrix
/// \param a in matrix
/// \param height_B number of columns in B
/// \param column_A_bytes number of bytes i each column of A
/// \param nr_cols_A  nr of cols in A
/// \param b in vector or length <= 8 elements
static inline
void gf16mat_prod_le8xle8_avx2_wrapper_v3(uint8_t *__restrict c,
                                          const uint8_t *__restrict__ a,
                                          const unsigned int nr_cols_B,
                                          const unsigned int column_A_bytes,
                                          const unsigned int nr_cols_A,
                                          const uint8_t *__restrict__ b) {
    const uint32_t *a32 = (const uint32_t *) a;
    assert(column_A_bytes < 8);
    assert(column_A_bytes > 4);
    assert(column_A_bytes == 5);
    assert(nr_cols_A <= 16);
    __m256i A1, A2, A3, A4, tmp1, tmp2;

    const uint32_t nr_bytes_B_col = nr_cols_A >> 1;
    const uint32_t mask =nr_bytes_B_col%4==0 ? 0x0f0f0f0f : (0x0f0f0f0f & ((1u << nr_bytes_B_col * 8) - 1ul));
    // const uint32_t n = ((nr_cols_A + 1) * column_A_bytes) >> 2;
    const __m256i perm = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);

    const __m256i Amask = _mm256_setr_epi32(0, 1, 2, 3, 5, 6, 7, 8);
    const __m256i Ashdo = _mm256_setr_epi32(0, 8, 16, 24, 0, 8, 16, 24);
    const __m256i Ashup = _mm256_setr_epi32(0, 24, 16, 8, 0, 24, 16, 8);
    const __m256i Ashfl = _mm256_setr_epi32(0, ((1<<8)-1)<<24, ((1<<16)-1)<<16, ((1<<24)-1)<<8,
                                            0, ((1<<8)-1)<<24, ((1<<16)-1)<<16, ((1<<24)-1)<<8);
    //const __m256i gather_mask = _mm256_setr_epi32(0 <= n ? -1 : 0, 1 <= n ? -1 : 0, 2 <= n ? -1 : 0,
    //                                              3 <= n ? -1 : 0, 4 <= n ? -1 : 0, 5 <= n ? -1 : 0,
    //                                              6 <= n ? -1 : 0, 7 <= n ? -1 : 0);

    A1 = _mm256_i32gather_epi32((const int *)(a32 + 0u), Amask, 4u);
    A2 = _mm256_i32gather_epi32((const int *)(a32 + 1u), Amask, 4u);
    __m256i A11 = _mm256_srlv_epi32(A1, Ashdo);
    __m256i A22 = _mm256_sllv_epi32(A2, Ashup);
    A2 = _mm256_srlv_epi32(A2, Ashdo);
    A1 = _mm256_blendv_epi8(A11, A22, Ashfl);

    if (nr_cols_A > 8) {
        A3 = _mm256_i32gather_epi32((const int *)(a32 + 10u), Amask, 4u);
        A4 = _mm256_i32gather_epi32((const int *)(a32 + 11u), Amask, 4u);
        __m256i A31 = _mm256_srlv_epi32(A3, Ashdo);
        __m256i A42 = _mm256_sllv_epi32(A4, Ashup);
        A4 = _mm256_srlv_epi32(A4, Ashdo);
        A3 = _mm256_blendv_epi8(A31, A42, Ashfl);
    }

    uint64_t rdata = 0;
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        uint32_t b_data = 0;
        if (i == nr_cols_B - 1) {
            for (uint32_t j = 0; j < (nr_cols_B * nr_bytes_B_col) % 4; j++) {
                b_data ^= b[nr_bytes_B_col * (nr_cols_B - 1) + j] << (j * 8);
            }
        }

        b_data = *((uint32_t *)(b+(nr_bytes_B_col * i)));
        uint32_t d11 = b_data & mask;
        uint32_t d12 = (b_data >> 4) & mask;

        __m256i B11 = _mm256_set1_epi32(d11);
        __m256i B12 = _mm256_set1_epi32(d12);
        B11 = _mm256_shuffle_epi8(B11, perm);
        B12 = _mm256_shuffle_epi8(B12, perm);
        const __m256i B1 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010

        tmp1 = gf16v_mul_lo_u256(A1, B1);
        tmp2 = gf16v_mul_lo_u256(A2, B1);

        if (nr_cols_A > 8) {
            b_data = *((uint32_t *)(b+(nr_bytes_B_col * i + 4)));
            d11 = b_data & mask;
            d12 = (b_data >> 4) & mask;

            B11 = _mm256_set1_epi32(d11);
            B12 = _mm256_set1_epi32(d12);
            B11 = _mm256_shuffle_epi8(B11, perm);
            B12 = _mm256_shuffle_epi8(B12, perm);
            const __m256i B2 = _mm256_blend_epi32(B11, B12, 0xaa); // 0b10101010


            tmp1 ^= gf16v_mul_lo_u256(A3, B2);
            tmp2 ^= gf16v_mul_lo_u256(A4, B2);
        }

        __m256i tempk3 = _mm256_unpacklo_epi32(tmp1, tmp2);
        __m256i tempk4 = _mm256_unpackhi_epi32(tmp1, tmp2);
        tempk3 = _mm256_xor_si256(tempk3, tempk4);
        __m256i ret = _mm256_xor_si256(tempk3, _mm256_srli_si256(tempk3, 8));
        ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 0x81)); //0b10000001
        rdata = _mm256_extract_epi64(ret, 0);
        if (i < nr_cols_B - 1) {
            *((uint64_t *) (c + i * column_A_bytes)) = rdata;
        }
    }

    // write the last element in a sane and memory safe way
    for (uint32_t i = 0; i < column_A_bytes; i++) {
        c[(nr_cols_B - 1)*column_A_bytes + i] = (uint8_t)(rdata>>(i*8));
    }
}

static inline
void gf16mat_prod_16x4xk(uint8_t *__restrict c,
                         const uint8_t *__restrict__ a,
                         const uint8_t *__restrict__ b, 
                         const uint32_t nr_cols_B) {
    // fully load A into a register 
    const __m256i A = _mm256_loadu_si256((const __m256i *)a);
    uint64_t *c64 = (uint64_t *)c;
    const uint32_t bytes_per_col_B = 2;

    // some helper variables
    const __m256i one = _mm256_set1_epi8(0x11);
    for (uint32_t i = 0; i < nr_cols_B; i++) {
        // extend a single column to an avx register
        const uint16_t t1 = *((uint16_t *)(b + i*bytes_per_col_B));
        const uint64_t t2 = _pdep_u64(t1, 0xF000F000F000F);
        const __m128i t3 = _mm_set_epi64x(0, t2);
        const __m256i t4 = _mm256_cvtepu16_epi64(t3);
        const __m256i t5 = _mm256_mul_epu32(t4, one);
        const __m256i t6 = t5^_mm256_slli_si256(t5, 4);

        const __m256i c1 = gf16v_mul_u256(t6, A);
        const uint64_t c2 = gf16_hadd_u256_64(c1);

        c64[i] = c2;
    }
}

///
/// \param b
/// \param cBb number of bytes in each column
/// \return
static inline
__m256i gf16mat_new_core_scatter_helper(const uint8_t *b,
										const uint32_t mask1,
										const uint32_t bytes) {
    const uint32_t mask = 0x0f0f0f0f & mask1;
    const __m256i perm  = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
	uint8_t arr[4] = {0};
	for (uint32_t i = 0; i<bytes; i++) {
		arr[i] = b[i];
	}

    // load 4bytes
    const uint32_t d11 = ( * (uint32_t *)arr)         & mask;
    const uint32_t d12 = ((*((uint32_t *)arr)) >> 4u) & mask;

    __m256i B11 = _mm256_set1_epi32(d11);
    __m256i B12 = _mm256_set1_epi32(d12);
    B11 = _mm256_shuffle_epi8(B11, perm);
    B12 = _mm256_shuffle_epi8(B12, perm);
    __m256i B = _mm256_blend_epi32(B11, B12, 0xaa); //0b10101010

    return B;
}

static inline
void gf16mat_new_core_scatter(uint8_t *__restrict__ c,
                              const uint8_t *__restrict__ a,
                              const unsigned int nr_cols_B,
                              const unsigned int column_A_bytes,
                              const unsigned int nr_cols_A,
                              const uint8_t *__restrict__ b) {
    __m256i Al1, Al2, Al3,
            Am1, Am2, Am3,
            Ah1, Ah2, Ah3;

    /// TODO research question:
    ///     - would it better if `cAb` >= 8 to use `_mm256_i64gather_epi32?`
    const uint32_t cBb = (nr_cols_A+1) >> 1;        // nr of bytes in each column in B
    const uint32_t cAb = column_A_bytes;            // nr of bytes in each column in A
    const uint32_t nAc = nr_cols_A;                 // nr of cols in A
    const uint32_t oA = cAb;                        // offset of each column in A
    const __m256i Amask = _mm256_setr_epi32(0, oA, 2*oA, 3*oA, 4*oA, 5*oA, 6*oA, 7*oA);

    /// load the whole matrix A into registers
                   Al1 = _mm256_i32gather_epi32((const int *) (a + 0u), Amask, 1u);
    if (cAb > 4) { Am1 = _mm256_i32gather_epi32((const int *) (a + 4u), Amask, 1u); }
    if (cAb > 8) { Ah1 = _mm256_i32gather_epi32((const int *) (a + 8u), Amask, 1u); }

    if (nAc > 8) {
                       Al2 = _mm256_i32gather_epi32((const int *)( a + 8*cAb + 0u), Amask, 1u);
        if (cAb > 4) { Am2 = _mm256_i32gather_epi32((const int *) (a + 8*cAb + 4u), Amask, 1u); }
        if (cAb > 8) { Ah2 = _mm256_i32gather_epi32((const int *) (a + 8*cAb + 8u), Amask, 1u); }
    }

    if (nAc > 16) {
                       Al3 = _mm256_i32gather_epi32((const int *) (a + 16*cAb + 0u), Amask, 1u);
        if (cAb > 4) { Am3 = _mm256_i32gather_epi32((const int *) (a + 16*cAb + 4u), Amask, 1u); }
        if (cAb > 8) { Ah3 = _mm256_i32gather_epi32((const int *) (a + 16*cAb + 8u), Amask, 1u); }
    }

    /// TODO improve mask computation
    const uint32_t mask_B_1 = cBb > 4 ? 0xffffffff : (cBb%4u == 0) ? 0xffffffff : (1u << (8*(cBb%4u))) - 1u;
    const uint32_t mask_B_2 = cBb > 8 ? 0xffffffff : (cBb%4u == 0) ? 0xffffffff : (1u << (8*(cBb%4u))) - 1u;
    const uint32_t mask_B_3 =                        (cBb%4u == 0) ? 0xffffffff : (1u << (8*(cBb%4u))) - 1u;

    const uint32_t mask1 = cAb > 4 ? 0xffffffff : (cAb%4u == 0) ? 0xffffffff : (1u << (8*(cBb%4u))) - 1u;
    const uint32_t mask2 = cAb > 4 ? 0xffffffff : (cAb%4u == 0) ? 0xffffffff : (1u << (8*(cBb%4u))) - 1u;

    __m256i acc1, acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
    for (uint32_t i = 0; i < nr_cols_B; ++i) {
        // TODO I think this can be partly replace by `_mm256_cvtepu8_epi32`
        __m256i B = gf16mat_new_core_scatter_helper(b + i*cBb, mask_B_1, MIN(cBb, 4));

        // compute the multiplication:
                       acc1 = gf16v_mul_lo_u256(Al1, B);
        if (cAb > 4) { acc2 = gf16v_mul_lo_u256(Am1, B); }
        if (cAb > 8) { acc3 = gf16v_mul_lo_u256(Ah1, B); }

        if (nAc > 8) {
            B = gf16mat_new_core_scatter_helper(b + i*cBb + 4, mask_B_2, MIN(cBb-4, 4));
                           acc1 ^= gf16v_mul_lo_u256(Al2, B);
            if (cAb > 4) { acc2 ^= gf16v_mul_lo_u256(Am2, B); }
            if (cAb > 8) { acc3 ^= gf16v_mul_lo_u256(Ah2, B); }
        }

        if (nAc > 16) {
            B = gf16mat_new_core_scatter_helper(b + i*cBb + 8, mask_B_3, MIN(cBb-8, 4));
                           acc1 ^= gf16v_mul_lo_u256(Al3, B);
            if (cAb > 4) { acc2 ^= gf16v_mul_lo_u256(Am3, B); }
            if (cAb > 8) { acc3 ^= gf16v_mul_lo_u256(Ah3, B); }
        }

        /// NOTE: do not remove this mask. For whatever reason C is not zero extending this cast.
        __uint128_t    tmp0  = _mm256_extract_epi32(gf16_hadd_avx2_32(acc1), 0) & (0xffffffff);
        if (cAb > 4) { tmp0 ^= ((__uint128_t )(_mm256_extract_epi32(gf16_hadd_avx2_32(acc2), 0) & mask1)) << 32u; }
        if (cAb > 8) { tmp0 ^= ((__uint128_t )(_mm256_extract_epi32(gf16_hadd_avx2_32(acc3), 0) & mask2)) << 64u; }

        for (uint32_t j = 0; j < cAb; j++) {
            c[i*cAb + j] ^= (uint8_t)(tmp0>>(j*8u));
        }
    }
}

// matrix1 += matrix2
static inline
void gf16_matrix_add_inplace_u256(ff_t *matrix1, 
                          const ff_t *matrix2, 
		                  const uint32_t n_rows, 
                          const uint32_t n_cols) {
    __m256i temp_sumand1, temp_sumand2, temp_result;
    uint32_t n_32, rem, n_bytes, jump = 0;

    n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    n_32 = (n_bytes >> 5);
    rem = n_bytes & 0x1f;
    
    while (n_32--) {
        temp_sumand1 = _mm256_loadu_si256((__m256i*) (matrix1 +  jump));
        temp_sumand2 = _mm256_loadu_si256((const __m256i*)(matrix2 + jump));
        temp_result = _mm256_xor_si256(temp_sumand1, temp_sumand2);
        _mm256_storeu_si256 ((__m256i*) (matrix1 + jump), temp_result);
        jump += 32;
    }

    uint32_t i = 0;

    for (; i+8 <= rem; i+=8) {
        const uint64_t temp_sumand1 = *((uint64_t *)(matrix1 + jump + i));
        const uint64_t temp_sumand2 = *((uint64_t *)(matrix2 + jump + i));
        const uint64_t temp_result = temp_sumand1 ^ temp_sumand2;
        *((uint64_t *)(matrix1 + jump + i)) = temp_result;
    }

    for (; i+4 <= rem; i+=4) {
        const uint32_t temp_sumand1 = *((uint32_t *)(matrix1 + jump + i));
        const uint32_t temp_sumand2 = *((uint32_t *)(matrix2 + jump + i));
        const uint32_t temp_result = temp_sumand1 ^ temp_sumand2;
        *((uint32_t *)(matrix1 + jump + i)) = temp_result;
    }

    for (; i < rem; ++i) {
        const uint8_t temp_sumand1 = matrix1[jump + i];
        const uint8_t temp_sumand2 = matrix2[jump + i];
        const uint8_t temp_result = temp_sumand1 ^ temp_sumand2;
        matrix1[jump + i] = temp_result;
    }
}


// matrix1 += scalar *matrix2
static inline
void gf16_matrix_add_scalar_u256(ff_t *matrix1, const ff_t scalar, const ff_t *matrix2,
    const uint32_t n_rows, const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16v_madd_avx2(matrix1, matrix2, scalar & 0xF, n_bytes);
}


// result += matrix1 * matrix2
// matrix1 of size n_rows1 * n_cols1
// matrix2 of size n_cols1 * n_cols2
// result  of size n_rows1 * n_cols2
// NOTE: only for small matrices: n_rows1 <= 24, n_cols1 <= 24
static inline
void gf16_matrix_add_mul_small_u256(ff_t *matrix1, const ff_t *matrix2, const ff_t *matrix3,
    const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2) {
    const uint32_t n_bytes_per_column1 = gf16_matrix_bytes_per_column(n_rows1);
    gf16mat_new_core_scatter(matrix1, matrix2, n_cols2, n_bytes_per_column1, 2*((n_cols1+1)>>1), matrix3);
}


// result = matrix2 * matrix3
// matrix1 of size n_rows1 * n_cols1
// matrix2 of size n_cols1 * n_cols2
// result  of size n_rows1 * n_cols2
// NOTE: only for small matrices: n_rows1 <= 24, n_cols1 <= 24
static inline
void gf16_matrix_mul_small_u256(ff_t *matrix1, const ff_t *matrix2, const ff_t *matrix3,
    const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2) {
    const uint32_t n_bytes_per_column1 = gf16_matrix_bytes_per_column(n_rows1);

	// first zero out the output matrix
	gf16_matrix_zero(matrix1, n_rows1, n_cols2);
    gf16mat_new_core_scatter(matrix1, matrix2, n_cols2, n_bytes_per_column1, 2*((n_cols1+1)>>1), matrix3);
}

/// matrix1 = matrix2 + matrix3
/// \param matrix1[out]
/// \param matrix2[in]
/// \param matrix3[in]
/// \param n_rows[in]
/// \param n_cols[in]
static inline
void gf16_matrix_add_u256(ff_t *matrix1,
                          const ff_t *matrix2,
                          const ff_t *matrix3,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16_vector_add_u256_v2(matrix1, matrix2, matrix3, 2*n_bytes);
}

/// result += matrix1 * matrix2, computed column by column:
///     result[:, j] += sum_k matrix2[k, j] * matrix1[:, k]
/// \param result[in/out]: n_rows1 x n_cols2
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf16_matrix_add_mul_u256(ff_t *result,
                              const ff_t *matrix1,
                              const ff_t *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    const uint32_t bpc1 = gf16_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const ff_t b = gf16_matrix_get(matrix2, n_cols1, k, j);
            gf16_vector_scalar_add_u256(result + j*bpc1, b, matrix1 + k*bpc1, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
/// \param result[out]: n_rows1 x n_cols2
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf16_matrix_mul_u256(ff_t *result,
                          const ff_t *matrix1,
                          const ff_t *matrix2,
                          const uint32_t n_rows1,
                          const uint32_t n_cols1,
                          const uint32_t n_cols2) {
    gf16_matrix_zero(result, n_rows1, n_cols2);
    gf16_matrix_add_mul_u256(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}

#endif // end: USE_AVX2

#ifdef USE_NEON
/// matrix1 = matrix2 + matrix3
static inline
void gf16_matrix_add_u128(ff_t *matrix1,
                          const ff_t *matrix2,
                          const ff_t *matrix3,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16_vector_add_u128_v2(matrix1, matrix2, matrix3, 2*n_bytes);
}

/// matrix1 += scalar * matrix2
static inline
void gf16_matrix_add_scalar_u128(ff_t *matrix1,
                                 const ff_t scalar,
                                 const ff_t *matrix2,
                                 const uint32_t n_rows,
                                 const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16_vector_scalar_add_u128(matrix1, scalar, matrix2, 2*n_bytes);
}

/// result += matrix1 * matrix2
static inline
void gf16_matrix_add_mul_u128(ff_t *result,
                              const ff_t *matrix1,
                              const ff_t *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    const uint32_t bpc1 = gf16_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const ff_t b = gf16_matrix_get(matrix2, n_cols1, k, j);
            gf16_vector_scalar_add_u128(result + j*bpc1, b, matrix1 + k*bpc1, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
static inline
void gf16_matrix_mul_u128(ff_t *result,
                          const ff_t *matrix1,
                          const ff_t *matrix2,
                          const uint32_t n_rows1,
                          const uint32_t n_cols1,
                          const uint32_t n_cols2) {
    gf16_matrix_zero(result, n_rows1, n_cols2);
    gf16_matrix_add_mul_u128(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif // end: USE_NEON

#ifdef USE_AVX512
/// matrix1 = matrix2 + matrix3
static inline
void gf16_matrix_add_u512(ff_t *matrix1,
                          const ff_t *matrix2,
                          const ff_t *matrix3,
                          const uint32_t n_rows,
                          const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16_vector_add_u512_v2(matrix1, matrix2, matrix3, 2*n_bytes);
}

/// matrix1 += scalar * matrix2
static inline
void gf16_matrix_add_scalar_u512(ff_t *matrix1,
                                 const ff_t scalar,
                                 const ff_t *matrix2,
                                 const uint32_t n_rows,
                                 const uint32_t n_cols) {
    const uint32_t n_bytes = gf16_matrix_bytes_size(n_rows, n_cols);
    gf16_vector_scalar_add_u512(matrix1, scalar, matrix2, 2*n_bytes);
}

/// result += matrix1 * matrix2
static inline
void gf16_matrix_add_mul_u512(ff_t *result,
                              const ff_t *matrix1,
                              const ff_t *matrix2,
                              const uint32_t n_rows1,
                              const uint32_t n_cols1,
                              const uint32_t n_cols2) {
    const uint32_t bpc1 = gf16_matrix_bytes_per_column(n_rows1);
    for (uint32_t j = 0; j < n_cols2; j++) {
        for (uint32_t k = 0; k < n_cols1; k++) {
            const ff_t b = gf16_matrix_get(matrix2, n_cols1, k, j);
            gf16_vector_scalar_add_u512(result + j*bpc1, b, matrix1 + k*bpc1, n_rows1);
        }
    }
}

/// result = matrix1 * matrix2
static inline
void gf16_matrix_mul_u512(ff_t *result,
                          const ff_t *matrix1,
                          const ff_t *matrix2,
                          const uint32_t n_rows1,
                          const uint32_t n_cols1,
                          const uint32_t n_cols2) {
    gf16_matrix_zero(result, n_rows1, n_cols2);
    gf16_matrix_add_mul_u512(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
}
#endif // end: USE_AVX512

