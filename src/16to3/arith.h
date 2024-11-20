#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "../helper.h"
#include "../16/arith.h"

/// Representation: a single F16to3 elements is stored in uin16_t
///  GF16to3:uint16_t: X^3 + x + 1;
typedef uint16_t gf16to3;

/// \return a+b
static inline gf16to3 gf16to3_add(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a+b for 5 elements in parallel
static inline gf16to3 gf16to3v_add(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// the same as add: char = 2 = best char
/// \return a-b = a+b
static inline gf16to3 gf16to3_sub(const gf16to3 a,
                                  const gf16to3 b) {
	return a ^ b;
}

/// \return a-b for 5 elements in parallel
static inline gf16to3 gf16to3v_sub(const uint64_t a,
                                   const uint64_t b) {
    return a ^ b;
}

/// vector multiplication
/// \return a*b
static inline gf16to3 gf16to3_mul_gf16(const gf16to3 a,
                                       const gf16 b) {
    return gf16_mul(a&0xF, b) ^
          (gf16_mul((a>>4)&0xF, b) << 4u) ^
          (gf16_mul((a>>8)&0xF, b) << 8u);
}

/// implemented using karatsuba
/// \return a*b
static inline gf16to3 gf16to3_mul(const gf16to3 a,
                                  const gf16to3 b) {
    gf16to3 r;

    const gf16 a0 = a&0xF, a1 = (a>>4)&0XF, a2 = (a>>8)&0xF;
    const gf16 b0 = b&0xF, b1 = (b>>4)&0XF, b2 = (b>>8)&0xF;

    const gf16 p0 = gf16_mul(a0, b0);
    const gf16 p1 = gf16_mul(a1, b1);
    const gf16 p2 = gf16_mul(a2, b2);

    const gf16 a01 = gf16_add(a0, a1);
    const gf16 a12 = gf16_add(a1, a2);
    const gf16 b01 = gf16_add(b0, b1);
    const gf16 b12 = gf16_add(b1, b2);
    const gf16 p01 = gf16_mul(a01, b01);
    const gf16 p12 = gf16_mul(a12, b12);

    const gf16 a012 = gf16_add(a01, a2);
    const gf16 b012 = gf16_add(b01, b2);
    const gf16 p012 = gf16_mul(a012, b012);

    // compute lowest limb
    r = gf16_add(p1, p2);
    r = gf16_add(r, p12);
    r = gf16_add(r, p0);

    r^= p2 << 4;
    r^= p01 << 4;
    r^= p0 << 4;
    r^= p1 << 4;

    r^= p012 << 8;
    r^= p01 << 8;
    r^= p12 << 8;
    return r;
}

static inline gf16to3 gf16to3v_mul(const uint64_t a,
                                   const uint64_t b) {
    const uint64_t m0 = 0x00F00F00F00F, m1 = 0x0F00F00F00F0,
                   m2 = 0xF00F00F00F00;
    uint64_t r;

    // TODO

    return r;
}


#ifdef USE_AVX2

/// vector multiplication
/// \param a
/// \param b
/// \return a*b;
static inline __m256i gf16to3v_mul_gf16_u256(const __m256i a,
                                             const uint8_t b) {
   const __m256i bb = _mm256_set1_epi8(b);
   return gf16v_mul_u256(a, bb);
}

/// NOTE there are multiple ways to implement this:
///     - assumes that each 1.5byte element is stored in 2 bytes,
///         thus there are 16 elements in a register
///     - or that that the elements are compressed together, meaning
///         we have 21 elements in a register
/// NOTE: this implementation follows the first approach
static inline __m256i gf16to3v_mul_u256(const __m256i a,
                                        const __m256i b) {
    __m256i r;
    const __m256i m0     = _mm256_set1_epi16(0x00F);
    const __m256i m1     = _mm256_set1_epi16(0x0F0);
    const __m256i m2     = _mm256_set1_epi16(0xF00);
    const __m256i m      = _mm256_set1_epi16(0xFFF);
    const __m256i pi     = gf16v_mul_full_u256(a, b);

    // bit    0     4      8     12     16
    // a01_12=[a0   , a0^a1, a1^a2,  a2]
    // b01_12=[b0   , b0^b1, b1^b2,  b2]
    const __m256i a01_12 = a ^ _mm256_slli_epi16(a, 4);
    const __m256i b01_12 = b ^ _mm256_slli_epi16(b, 4);

    // bit    0     4      8     12        16
    // a012 = [a0   , a0^a1, a1^a2, a2^a1^a0]
    // b012 = [b0   , b0^b1, b1^b2, b2^b1^b0]
    const __m256i a012 = a01_12 ^ _mm256_slli_epi16(a01_12&m1 , 8);
    const __m256i b012 = b01_12 ^ _mm256_slli_epi16(b01_12&m1 , 8);

    // bit    0    4    8    12    16
    // p012 = [p0 , p01, p12 , p012]
    // p012 = [p01, p12, p012, 0000]
    __m256i p012 = gf16v_mul_full_u256(a012, b012);
    p012 = _mm256_srli_epi16(p012, 4);

    // bit 0         4         8    16
    // r = [p0^p1^p2, p0^p1^p2, 0, 0]
    r = pi ^ (_mm256_srli_epi16(pi, 4));
    r ^= _mm256_srli_epi16(pi, 8);
    r ^= _mm256_slli_epi16(pi&m0, 4);     // ^= p0
    r &= (m0 ^ m1);

    r ^= _mm256_srli_epi16(p012, 4) & m0;   //^= p12
    r ^= _mm256_slli_epi16(p012, 4);        // ^= p01, ^= p12
    r ^= _mm256_slli_epi16(p012, 8);
    r ^= p012&m2; // ^= p012

    return r&m;
}

/// compressed representations
/// \param out
/// \param in1
/// \param d  number of elements NOT bytes
static inline void gf16to3_vector_add_u256(gf16to3 *out,
                                           const gf16to3 *in1,
                                           const uint32_t d) {
    size_t i = (size_t)((double)d * 1.5) + d&1u;

    uint8_t *o8 = (uint8_t *)out;
    uint8_t *i8 = (uint8_t *)in1;

    // avx2 code
    while (i >= 32u) {
        _mm256_storeu_si256((__m256i *)o8,
                            _mm256_loadu_si256((__m256i *)o8) ^
                            _mm256_loadu_si256((__m256i *)i8));
        i   -= 32u;
        in1 += 32u;
        out += 32u;
    }

    // sse code
    while(i >= 16u) {
        _mm_storeu_si128((__m128i *)o8,
                         _mm_loadu_si128((__m128i *)o8) ^
                         _mm_loadu_si128((__m128i *)i8));
        i   -= 16u;
        in1 += 16u;
        out += 16u;
    }

    for (; i > 0; --i) {
        *o8++ ^= *i8++;
    }
}
#endif

#ifdef USE_NEON

// TODO neon
uint16x8_t gf16to3v_mul_u128(uint16x8_t a, uint16x8_t b) {
    uint16x8_t m0 = vdupq_n_u8(0x00f);
    uint16x8_t m1 = vdupq_n_u8(0x0f0);
    uint16x8_t m2 = vdupq_n_u8(0xf00);
    uint8x16_t tab_reduce = vld1q_u8(__gf16_reduce);
    uint8x16_t bp = vdupq_n_u8(b);

    uint8x16_t a0 = a&m0;
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
    uint8x16_t a1 = vshrq_n_u8( a , 4 );
	// mul
    poly8x16_t abl = vmulq_p8( al0 , bp );
    poly8x16_t abh = vmulq_p8( ah0 , bp );

    poly8x16_t rl = abl ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abl,4) );
    poly8x16_t rh = abh ^ vqtbl1q_u8( tab_reduce , vshrq_n_u8(abh,4) );

    return vsliq_n_u8( rl , rh , 4 );
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// NOTE all matrices are column major
gf16to3* gf16to3_matrix_alloc(const uint32_t nrows,
                              const uint32_t ncols) {
    return (gf16to3 *)malloc(sizeof(gf16to3) * nrows * ncols);
}

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


void gf16to3_matrix_rand(gf16to3 *m,
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
    for (int i = 0; i < nrows; ++i) {
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
            entry_i_j = gf16to3_matrix_get(result, n_rows1, i, j);

            for (uint32_t k = 0; k < n_cols1; k++) {
                entry_i_k = gf16to3_matrix_get(matrix1, n_rows1, i, k);
                entry_k_j = gf16to3_matrix_get(matrix2, n_cols1, k, j);
                entry_i_j ^= gf16to3_mul(entry_i_k, entry_k_j);
            }

            gf16to3_matrix_set(result, n_rows1, i, j, entry_i_j);
        }
    }
}


#ifdef USE_AVX2
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

// TODO not finished
// much slower
// static inline void gf16to3_matrix_mul_16x16_v2(gf16to3 *result,
//                                                const gf16to3 *A,
//                                                const gf16to3 *B) {
//     gf16to3_matrix_zero(result, 16, 16);
//     const uint32_t nrows_B = 16;
//     const uint32_t ncols_B = 16;
//
//     uint32_t *r32 = (uint32_t *)result;
//     const __m256i Am = _mm256_setr_epi32(0,8,16,24,32,40,48,56);
//
//     for (uint32_t j = 0; j < 8; ++j) {
//         const __m256i a1 = _mm256_i32gather_epi32((int *const)(A + 4*j +   0), Am, 4);
//         const __m256i a2 = _mm256_i32gather_epi32((int *const)(A + 4*j + 256), Am, 4);
//
//         for (uint32_t i = 0; i < ncols_B; ++i) {
//             const __m128i b128_1 = _mm_loadu_si128((const __m128i *)(B + i*ncols_B +  0));
//             const __m128i b128_2 = _mm_loadu_si128((const __m128i *)(B + i*ncols_B + 16));
//             const __m256i b256_t1= _mm256_cvtepu16_epi32(b128_1);
//             const __m256i b256_t2= _mm256_cvtepu16_epi32(b128_2);
//             const __m256i b1     = b256_t1 ^_mm256_slli_epi32(b256_t1, 16);
//             const __m256i b2     = b256_t2 ^_mm256_slli_epi32(b256_t2, 16);
//
//             __m256i c = gf16to3v_mul_u256(a1, b1);
//             c ^= gf16to3v_mul_u256(a2, b2);
//
//             uint32_t c32 = _mm256_extract_epi32(gf16_hadd_avx2_32(c), 0);
//             *(r32 + j + i*8) = c32;
//         }
//     }
// }

/// Assumes 16 rows in A, the rest doesnt matter
/// col major
static inline void gf16to3_matrix_mul_16xX(gf16to3 *result,
                                           const gf16to3 *A,
                                           const gf16to3 *B,
                                           const uint32_t nrows_B,
                                           const uint32_t ncols_B) {
    gf16to3_matrix_zero(result, 16, nrows_B);

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
#endif
#undef MODULUS
