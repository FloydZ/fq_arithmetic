#pragma once

/// CUDA implementation of GF(256^4) = GF(256^2)[Y]/(Y^2 + Y + 0x2000),
/// identical to `arith.h`.
///
/// Representation (identical to `arith.h`):
///  - an element a = a_1*Y + a_0 is a `uint32_t`, a_0 in the low 16 bits,
///    a_1 in the high 16 bits (a_i in GF(256^2), see ../256to2/arith.cu)
///  - matrices are column major: entry (i, j) is at index j*nrows + i, no padding.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../256to2/arith.cu"

namespace gf256to4_cuda {

using gf256to4 = uint32_t;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf256to4_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows * (uint32_t)sizeof(gf256to4);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf256to4_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return (size_t)nrows * ncols * sizeof(gf256to4);
}

/// Addition in GF(256^4)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
gf256to4 gf256to4_add(const gf256to4 a,
                      const gf256to4 b) {
    return a ^ b;
}

/// Subtraction in GF(256^4), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
gf256to4 gf256to4_sub(const gf256to4 a,
                      const gf256to4 b) {
    return a ^ b;
}

/// Multiplication in GF(256^4). Constant time, no lookup tables.
/// Karatsuba with Y^2 = Y + 0x2000:
///     low  = a_0*b_0 + 0x2000*a_1*b_1
///     high = (a_0 + a_1)*(b_0 + b_1) + a_0*b_0
/// The 4 GF(256^2) products are computed with 2 calls to the 2-lane
/// `gf256to2v_mul_u32`.
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
gf256to4 gf256to4_mul(const gf256to4 a,
                      const gf256to4 b) {
    // lanes: [a_0*b_0, a_1*b_1]
    const uint32_t p = gf256to2_cuda::gf256to2v_mul_u32(a, b);
    const uint32_t a0b0 = p & 0xFFFFu;
    const uint32_t a1b1 = p >> 16u;

    // lanes: [(a_0 + a_1)*(b_0 + b_1), 0x2000*a_1*b_1]
    const uint32_t x = ((a ^ (a >> 16u)) & 0xFFFFu) | (a1b1 << 16u);
    const uint32_t y = ((b ^ (b >> 16u)) & 0xFFFFu) | (0x2000u << 16u);
    const uint32_t q = gf256to2_cuda::gf256to2v_mul_u32(x, y);

    const uint32_t low  = a0b0 ^ (q >> 16u);
    const uint32_t high = (q & 0xFFFFu) ^ a0b0;
    return low | (high << 16u);
}

/// element-wise GF(256^4) multiplication
struct gf256to4_mul_op {
    __device__ __forceinline__ uint32_t word(const uint32_t a, const uint32_t b) const { return gf256to4_mul(a, b); }
    __device__ __forceinline__ gf256to4 operator()(const gf256to4 a, const gf256to4 b) const { return gf256to4_mul(a, b); }
};

/// GF(256^4) lane description for `gf256_cuda::matrix_mul_words`: 1 element per word
struct gf256to4_field {
    static __device__ __forceinline__ uint32_t broadcast(const gf256to4 s) { return s; }
    static __device__ __forceinline__ uint32_t mul_word(const uint32_t a, const uint32_t b) { return gf256to4_mul(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to4_vector_add(gf256to4 *c,
                         const gf256to4 *a,
                         const gf256to4 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256_cuda::xor_op<gf256to4>{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to4_vector_sub(gf256to4 *c,
                         const gf256to4 *a,
                         const gf256to4 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256_cuda::xor_op<gf256to4>{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256to4_vector_mul(gf256to4 *c,
                         const gf256to4 *a,
                         const gf256to4 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(c, a, b, n, gf256to4_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256to4_matrix_add(gf256to4 *matrix1,
                         const gf256to4 *matrix2,
                         const gf256to4 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols,
                                   gf256_cuda::xor_op<gf256to4>{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256to4_matrix_sub(gf256to4 *matrix1,
                         const gf256to4 *matrix2,
                         const gf256to4 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    gf256_cuda::packed_elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols,
                                   gf256_cuda::xor_op<gf256to4>{}, stream);
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
void gf256to4_matrix_mul(gf256to4 *result,
                         const gf256to4 *matrix1,
                         const gf256to4 *matrix2,
                         const uint32_t n_rows1,
                         const uint32_t n_cols1,
                         const uint32_t n_cols2,
                         cudaStream_t stream = 0) {
    gf256_cuda::matrix_mul_words<gf256to4, gf256to4_field>(result, matrix1, matrix2,
                                                           n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256to4_cuda
