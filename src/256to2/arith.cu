#pragma once

/// CUDA implementation of GF(256^2) = GF(256)[X]/(X^2 + X + 0x20),
/// identical to `arith.h`.
///
/// Representation (identical to `arith.h`):
///  - an element a = a_1*X + a_0 is a `uint16_t`, a_0 in the low byte,
///    a_1 in the high byte (a_i in the AES field GF(256), see ../256/arith.cu)
///  - matrices are column major: entry (i, j) is at index j*nrows + i, no padding.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../256/arith.cu"

namespace gf256to2_cuda {

using gf256to2 = uint16_t;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf256to2_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows * (uint32_t)sizeof(gf256to2);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf256to2_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return (size_t)nrows * ncols * sizeof(gf256to2);
}

/// Addition in GF(256^2)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
gf256to2 gf256to2_add(const gf256to2 a,
                      const gf256to2 b) {
    return a ^ b;
}

/// Subtraction in GF(256^2), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
gf256to2 gf256to2_sub(const gf256to2 a,
                      const gf256to2 b) {
    return a ^ b;
}

/// Lane-wise multiplication of 2 GF(256^2) elements packed into 32 bits
/// (one element per 16-bit lane). Constant time, no lookup tables.
/// Per lane (Karatsuba, X^2 = X + 0x20):
///     low  = a_0*b_0 + 0x20*a_1*b_1
///     high = (a_0 + a_1)*(b_0 + b_1) + a_0*b_0
/// \param a[in]: 2 packed elements
/// \param b[in]: 2 packed elements
/// \return [a_0*b_0, a_1*b_1] packed
__host__ __device__ __forceinline__
uint32_t gf256to2v_mul_u32(const uint32_t a,
                           const uint32_t b) {
    const uint32_t mask_low = 0x00FF00FFu;
    // bytes: [a_0*b_0, a_1*b_1] per lane
    const uint32_t ab = gf256_cuda::gf256v_mul_u32(a, b);
    // low bytes: (a_0 + a_1), (b_0 + b_1) per lane
    const uint32_t sa = (a ^ (a >> 8u)) & mask_low;
    const uint32_t sb = (b ^ (b >> 8u)) & mask_low;
    const uint32_t m = gf256_cuda::gf256v_mul_u32(sa, sb);

    // 0x20 * a_1*b_1 = x^5 * a_1*b_1
    uint32_t h = (ab >> 8u) & mask_low;
#pragma unroll
    for (uint32_t i = 0; i < 5; i++) {
        h = gf256_cuda::gf256v_mulx_u32(h);
    }

    const uint32_t low  = (ab ^ h) & mask_low;
    const uint32_t high = (m ^ ab) & mask_low;
    return low | (high << 8u);
}

/// Multiplication in GF(256^2)
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
gf256to2 gf256to2_mul(const gf256to2 a,
                      const gf256to2 b) {
    return (gf256to2)gf256to2v_mul_u32(a, b);
}

/// element-wise GF(256^2) multiplication
struct gf256to2_mul_op {
    __device__ __forceinline__ uint32_t word(const uint32_t a, const uint32_t b) const { return gf256to2v_mul_u32(a, b); }
    __device__ __forceinline__ gf256to2 operator()(const gf256to2 a, const gf256to2 b) const { return gf256to2_mul(a, b); }
};

/// GF(256^2) lane description for `gf256_cuda::matrix_mul_words`: 2 elements per word
struct gf256to2_field {
    static __device__ __forceinline__ uint32_t broadcast(const gf256to2 s) { return (uint32_t)s * 0x00010001u; }
    static __device__ __forceinline__ uint32_t mul_word(const uint32_t a, const uint32_t b) { return gf256to2v_mul_u32(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to2_vector_add(gf256to2 *c,
                         const gf256to2 *a,
                         const gf256to2 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256_cuda::xor_op<gf256to2>{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to2_vector_sub(gf256to2 *c,
                         const gf256to2 *a,
                         const gf256to2 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256_cuda::xor_op<gf256to2>{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to2_vector_mul(gf256to2 *c,
                         const gf256to2 *a,
                         const gf256to2 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256to2_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256to2_matrix_add(gf256to2 *matrix1,
                         const gf256to2 *matrix2,
                         const gf256to2 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols,
                                   gf256_cuda::xor_op<gf256to2>{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256to2_matrix_sub(gf256to2 *matrix1,
                         const gf256to2 *matrix2,
                         const gf256to2 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols,
                                   gf256_cuda::xor_op<gf256to2>{}, stream);
}

/// result = matrix1 * matrix2, column major
/// \param result[out]: device pointer, n_rows1 x n_cols2. Must not alias the inputs.
/// \param matrix1[in]: device pointer, n_rows1 x n_cols1
/// \param matrix2[in]: device pointer, n_cols1 x n_cols2
/// \param n_rows1[in]: number of rows of matrix1
/// \param n_cols1[in]: number of columns of matrix1 = number of rows of matrix2
/// \param n_cols2[in]: number of columns of matrix2
/// \param stream[in]: CUDA stream
static inline
void gf256to2_matrix_mul(gf256to2 *result,
                         const gf256to2 *matrix1,
                         const gf256to2 *matrix2,
                         const uint32_t n_rows1,
                         const uint32_t n_cols1,
                         const uint32_t n_cols2,
                         cudaStream_t stream = 0) {
    gf256_cuda::matrix_mul_words<gf256to2, gf256to2_field>(result, matrix1, matrix2,
                                                           n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256to2_cuda
