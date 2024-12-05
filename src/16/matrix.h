#pragma once

#include <string.h>
#include "arith.h"


#define gf16_matrix_bytes_per_column(n_rows) ((n_rows >> 1u) + (n_rows & 1u))
#define gf16_matrix_bytes_size(n_rows, n_cols) ((gf16_matrix_bytes_per_column(n_rows)) * (n_cols))


// a > b -> b - a is negative
// returns 0xFFFFFFFF if true, 0x00000000 if false
static inline uint32_t ct_is_greater_than(int a, int b) {
    int32_t diff = b - a;
    return (uint32_t) (diff >> (8*sizeof(uint32_t)-1));
}

// if a == b -> 0x00000000, else 0xFFFFFFFF
static inline uint32_t ct_compare_32(int a, int b) {
    return (uint32_t)((-(int32_t)(a ^ b)) >> (8*sizeof(uint32_t)-1));
}

// if a == b -> 0x00, else 0xFF
static inline unsigned char ct_compare_8(unsigned char a, unsigned char b) {
    return (int8_t)((-(int32_t)(a ^ b)) >> (8*sizeof(uint32_t)-1));
}


uint32_t gf16_matrix_load4(const ff_t *m,
                              const uint32_t nrows,
                              const uint32_t i,
                              const uint32_t j,
                              const uint32_t bytes) {
    uint8_t tmp[4];
    const uint32_t pos = gf16_matrix_bytes_per_column(nrows) * j + i/2;
    for (uint32_t k = 0; k < MIN(bytes, 4); k++) {
        tmp[k] = m[pos + k];
    }

    return *((uint32_t*)tmp);
}

ff_t* gf16_matrix_alloc(const uint32_t n_rows,
                        const uint32_t n_cols) {
    return (ff_t*)malloc(gf16_matrix_bytes_size(n_rows, n_cols));
}

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

///
/// \param matrix
/// \param n_rows
/// \param i
/// \param j
/// \param scalar
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
        matrix[target_byte_id] |= (scalar << 4);
    }
    else {
        matrix[target_byte_id] &= 0xf0;
        matrix[target_byte_id] |= scalar;
    }
}


// A[row1] -= a*A[row2]
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

// A[row1] -= a*A[row2]
void gf16_sub_row_transpose(uint8_t *A,
                            const uint32_t nrows,
                            const uint32_t ncols,
                            const uint32_t row1,
                            const uint32_t row2,
                            const uint8_t a) {
    const uint32_t bytes = ncols/2;
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
    for (; i < bytes; i++) {
        uint8_t d1 = gf16_matrix_get(A, ncols, i, row1);
        uint8_t d2 = gf16_matrix_get(A, ncols, i, row2);
        d1 ^= gf16_mul(a, d2);
        gf16_matrix_set(A, ncols, i, row1, d1);
    }
}

///
/// \param A
/// \param nrows
/// \param ncols
void gf16_matrix_print(uint8_t *A,
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
void gf16_matrix_print_transposed(uint8_t *A,
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

///
/// \param C
/// \param nrows
/// \param ncols
void gf16_matrix_zero(uint8_t *C,
				    const size_t nrows,
				    const size_t ncols) {
    memset((void *)C, 0, gf16_matrix_bytes_size(nrows, ncols));
}

void gf16_matrix_id(uint8_t *C,
				    const size_t nrows,
				    const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    gf16_matrix_set(C, nrows, i, j, i==j);
		}
	}
}

void gf16_matrix_rng(uint8_t *C,
				const size_t nrows,
				const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    gf16_matrix_set(C, nrows, i, j, rand()&0xF);
		}
	}
}

void gf16_matrix_rng_full_rank(uint8_t *C,
				const size_t nrows,
				const size_t ncols) {
	for (uint32_t i = 0; i < nrows; i++) {
		for (uint32_t j = 0; j < ncols; j++) {
		    gf16_matrix_set(C, nrows, i, j, rand()&0xF);
		}
	}

	for (uint32_t i = 0; i < nrows; i++) {
	    for (uint32_t j = 0; j < nrows; j++) {
	        if (i == j) { continue;}
	        uint32_t r = rand();
	        if (r&1) {
	            gf16_sub_row(C, nrows, ncols, i ,j, 1);
	        }
	    }
	}
}

/// A, B, C are in row-majow form
void matrix_mul(uint8_t *C, uint8_t *A, uint8_t *B,
				const size_t nrows_A, const size_t ncols_A, const size_t ncols_B) {
	for (uint32_t i = 0; i < nrows_A; i++) {
		for (uint32_t j = 0; j < ncols_B; j++) {
			ff_t t = 0;
			for (uint32_t k = 0; k < ncols_A; k++) {
				t = gf16_add(t, gf16_mul(A[i*ncols_A + k], B[ncols_A*k + j]));
			}

			C[i*ncols_B + j] = t;
		}
	}
}

void gf16_matrix_tranpose(uint8_t *B,
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

/// TODO: add all of these functions to `cryptanalysislib`
/// org source: https://github.com/PQCMayo/MAYO-C/blob/f2aab8ab3a0052272905bdfb57048c150938b8e5/src/mayo.c#L102
/// NOTE: A is a col-major input matrix
/// NOTE: B is a col-major output matrix
/// NOTE: stride in bytes
static void gf16_transpose_16x16(uint8_t *B,
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
static void gf16_transpose_64x64(uint8_t *B,
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
static void gf16_transpose_64x64_avx2(uint8_t *B,
                                      const uint8_t *const A,
                                      const uint32_t stride) {
    __m256i M[64];
    const __m256i mask1 = _mm256_set1_epi8  (0x0F),
                  mask2 = _mm256_set1_epi16 (0x00FF),
                  mask3 = _mm256_set1_epi32 (0x0000FFFF),
                  mask4 = _mm256_set1_epi64x(0x00000000FFFFFFFF),
                  mask5 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0),
                  mask6 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0);
    for (uint32_t i = 0; i < 64; i++) {
        M[i] = _mm256_loadu_si256((__m256i *)(A+i*32));
    }
    for (uint32_t i = 0; i < 64; i+=2) {
        __m256i t = (_mm256_srli_epi64(M[i], 4) ^ M[i+1]) & mask1;
        M[i+0] ^= _mm256_slli_epi64(t, 4);
        M[i+1] ^= t;
    }

    for (uint32_t i = 0; i < 64; i+=4) {
        __m256i t0 = (_mm256_srli_epi64(M[i+0], 8) ^ M[i+2]) & mask2;
        __m256i t1 = (_mm256_srli_epi64(M[i+1], 8) ^ M[i+3]) & mask2;
        M[i+0] ^= _mm256_slli_epi64(t0, 8);
        M[i+1] ^= _mm256_slli_epi64(t1, 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }
    for (uint32_t i = 0; i < 64; i+=8) {
        __m256i t0 = (_mm256_srli_epi64(M[i+0], 16) ^ M[i+4]) & mask3;
        __m256i t1 = (_mm256_srli_epi64(M[i+1], 16) ^ M[i+5]) & mask3;
        __m256i t2 = (_mm256_srli_epi64(M[i+2], 16) ^ M[i+6]) & mask3;
        __m256i t3 = (_mm256_srli_epi64(M[i+3], 16) ^ M[i+7]) & mask3;
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
        _mm256_storeu_si256((__m256i *)(B + i*stride), M[i]);
    }
}
#endif

/// B = A^T
void gf16_matrix_tranpose_opt(uint8_t *B,
                              const uint8_t *const A,
				              const size_t nrows,
				              const size_t ncols) {
    if (nrows >= 64 && ncols >= 64) {
        gf16_transpose_64x64_avx2(B, A, nrows/2);
    }

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 64; j < ncols; j++) {
            const ff_t a = gf16_matrix_get(A, nrows, i, j);
            gf16_matrix_set(B, ncols, j, i, a);
        }
    }
    for (size_t i = 64; i < nrows; i++) {
        for (size_t j = 0; j < 64; j++) {
            const ff_t a = gf16_matrix_get(A, nrows, i, j);
            gf16_matrix_set(B, ncols, j, i, a);
        }
    }
}

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
        _mm256_store_si256((__m256i *)(A +j*bytes + k), a);
        _mm256_store_si256((__m256i *)(A +i*bytes + k), b);
    }
#endif
    for (; k < bytes; k++) {
        const uint8_t tmp = A[i*bytes + k];
        A[i*bytes + k] = A[j*bytes + k];
        A[j*bytes + k] = tmp;
    }
}

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

// assumes that A is transpose
void gf16_swap_rows_transpose(uint8_t *A,
                               const uint32_t nrows,
                               const uint32_t ncols,
                               const uint32_t i,
                               const uint32_t j) {
    if (i == j) { return; }

    const uint32_t bytes = ncols/2;
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


/// NOTE: assumes that A is row-major
/// A[row] *= c;
void gf16_solve_row_transpose(uint8_t *A,
                              const uint32_t nrows,
                              const uint32_t ncols,
                              const uint32_t row,
                              const uint8_t c) {
    if (c == 1) { return; }
    const uint32_t bytes = ncols/2;
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
    for (; i < bytes; i++) {
        uint8_t d = gf16_matrix_get(A, nrows, row, i);
        d = gf16_mul(c, d);
        gf16_matrix_set(A, nrows, row, i, d);
    }
}


/// returns 1 on error, 0 else
/// solves Ax=b
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

/// TODO not finished, the `gf16_swap_rows is not fully correct`
/// @param B
/// @param A tmp matrix, must be size of transposed of B
/// @param b
/// @param nrows
/// @param ncols
/// @return
uint32_t gf16_solve_transpose(uint8_t *B,
                              uint8_t *A,
                              uint8_t *b,
                              const uint32_t nrows,
                              const uint32_t ncols) {
    gf16_matrix_tranpose_opt(A, B, nrows, ncols);

    uint32_t row = 0;
    for (; row < nrows; row++) {
        // find pivot
        uint32_t i = row;
        for (; i < nrows; i++) {
            if (gf16_matrix_get(A, ncols, i, row)) {
                break;
            }
        }

        if (i == nrows) { break; }

        const uint8_t c = gf16_inv(gf16_matrix_get(A, ncols, i, row));
        gf16_swap_rows_transpose(A, nrows, ncols, i, row);

        /// solve current pivot row
        // gf16_matrix_print_transposed(A, nrows, ncols);
        gf16_solve_row_transpose(A, nrows, ncols, row, c);
        gf16_vector_set(b, row, gf16_mul(gf16_vector_get(b, row), c));
        // gf16_matrix_print_transposed(A, nrows, ncols);

        // solve remaining rows below
        for (uint32_t j = row+1; j < nrows; j++) {
            const uint8_t d = gf16_matrix_get(A, ncols, row, j);
            gf16_sub_row_transpose(A, nrows, ncols, j, row, d);
            gf16_vector_set(b, j, gf16_vector_get(b, j) ^ gf16_mul(gf16_vector_get(b, row), d));
        }

        // gf16_matrix_print_transposed(A, nrows, ncols);
        // solve remaining rows above
        for (uint32_t j = 0; j < row; j++) {
            const uint8_t d = gf16_matrix_get(A, nrows, row, j);
            gf16_sub_row_transpose(A, nrows, ncols, j, row, d);
            gf16_vector_set(b, j, gf16_vector_get(b, j) ^ gf16_mul(gf16_vector_get(b, row), d));
        }
    }

    gf16_matrix_tranpose_opt(B, A, ncols, nrows);
    return row;
}

#ifdef USE_AVX2
// org code from:https://github.com/PQCMayo/MAYO-C/blob/nibbling-mayo/src/AVX2/echelon_form.h
// put matrix in row echelon form with ones on first nonzero entries in constant time
static inline void row_echelon_form_compressed(unsigned char *A, int _nrows, int _ncols) {

    (void) _nrows;
    (void) _ncols;

#define nrows 32
#define ncols 64



#define AVX_REGS_PER_ROW ((ncols + 63) / 64)
#define MAX_COLS (AVX_REGS_PER_ROW * 64)

    __m256i _pivot_row[AVX_REGS_PER_ROW];
    __m256i A_avx     [AVX_REGS_PER_ROW * nrows];

    // TODO transpose and stuff
    unsigned char* pivot_row_bytes = (unsigned char*) _pivot_row;
    unsigned char* A_bytes = (unsigned char*) A_avx;

    // load A in the tail of AVX2 registers
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < (ncols/2); j++) {
            A_bytes[i*(MAX_COLS/2) + ((MAX_COLS/2) - (ncols/2)) + j] = A[i*(ncols/2) + j ];
        }
    }
    const __m256i xf = _mm256_set1_epi8(0xf);

    // pivot row is secret, pivot col is not
    unsigned char inverse;
    int pivot_row = 0;
    int pivot_col = MAX(MAX_COLS - ncols, 0);
    for (; pivot_col < MAX_COLS-128; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-96; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-64; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS-32; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }
    for (; pivot_col < MAX_COLS; pivot_col++) {
#include "echelon_form_loop_compressed.h"
    }

    // write the matrix A back
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < (ncols/2); j++) {
            A[i * (ncols/2) + j] = A_bytes[i*AVX_REGS_PER_ROW*32 + ((MAX_COLS/2) - (ncols/2)) + j];
        }
    }
#undef ncols
#undef nrows
#undef AVX_REGS_PER_ROW
#undef MAX_COLS
}

static inline void row_echelon_form(unsigned char *A, int _nrows, int _ncols) {

    (void) _nrows;
    (void) _ncols;

#define nrows 32
#define ncols 32
#define AVX_REGS_PER_ROW ((ncols + 31) / 32)
#define MAX_COLS (AVX_REGS_PER_ROW * 32)

    __m256i _pivot_row[AVX_REGS_PER_ROW];
    __m256i A_avx     [AVX_REGS_PER_ROW * nrows];

    unsigned char* pivot_row_bytes = (unsigned char*) _pivot_row;
    unsigned char* A_bytes = (unsigned char*) A_avx;

    // load A in the tail of AVX2 registers
    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            A_bytes[i*MAX_COLS + (MAX_COLS - ncols) + j] = A[i*ncols + j ];
        }
    }
    const __m256i xf = _mm256_set1_epi8(0xf);

    // pivot row is secret, pivot col is not
    unsigned char inverse;
    int pivot_row = 0;
    int pivot_col = MAX(MAX_COLS - ncols, 0);
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
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < (ncols/2); j++) {
            A[i * ncols + j] = A_bytes[i*AVX_REGS_PER_ROW*32 + (MAX_COLS - ncols) + j];
        }
    }
#undef ncols
#undef nrows
#undef AVX_REGS_PER_ROW
#undef MAX_COLS
}
#endif