#pragma once

/// CUDA implementation of GF(256^16) = GF(256^8)[X]/(X^2 + X + 0x2000000000000000),
/// the same tower as `arith.h` (see ../256to8/arith.cu for the lower levels).
///
/// An element is stored exactly like the C type `uint8_t[16]`: two little
/// endian GF(256^8) limbs, v[0] = a_0 (bytes 0..7), v[1] = a_1 (bytes 8..15),
/// a = a_1*X + a_0. `gf256to16` has the same size, but an alignment of 8, so
/// device pointers to vectors/matrices must be 8-byte aligned (cudaMalloc is).
///
/// Vectors/matrices: plain arrays of elements, matrices column-major
/// (entry (i, j) at j*nrows + i). All host wrappers take DEVICE pointers,
/// enqueue on `stream` and do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"
#include "../256to8/arith.cu"

namespace gf256to16_cuda {

struct gf256to16 {
    uint64_t v[2];
};
static_assert(sizeof(gf256to16) == 16, "must match the C type uint8_t[16]");

/// \return a+b
__host__ __device__ __forceinline__
gf256to16 gf256to16_add(const gf256to16 &a,
                        const gf256to16 &b) {
    gf256to16 r;
    r.v[0] = a.v[0] ^ b.v[0];
    r.v[1] = a.v[1] ^ b.v[1];
    return r;
}

/// \return a-b
__host__ __device__ __forceinline__
gf256to16 gf256to16_sub(const gf256to16 &a,
                        const gf256to16 &b) {
    return gf256to16_add(a, b);
}

/// Karatsuba over GF(256^8), X^2 = X + gamma_3:
///     r_0 = a_0*b_0 + gamma_3*a_1*b_1,  r_1 = (a_0+a_1)*(b_0+b_1) + a_0*b_0
/// \return a*b
__host__ __device__ __forceinline__
gf256to16 gf256to16_mul(const gf256to16 &a,
                        const gf256to16 &b) {
    using namespace gf256to8_cuda;
    const gf256to8 l = gf256to8_mul(a.v[0], b.v[0]);
    const gf256to8 h = gf256to8_mul(a.v[1], b.v[1]);
    const gf256to8 s = gf256to8_mul(a.v[0] ^ a.v[1], b.v[0] ^ b.v[1]);
    gf256to16 r;
    r.v[0] = l ^ gf256to8_mul_gamma(h);
    r.v[1] = s ^ l;
    return r;
}

/// field policy for the generic kernels
struct gf256to16_field {
    typedef gf256to16 T;
    __host__ __device__ static T zero() { T r; r.v[0] = 0; r.v[1] = 0; return r; }
    __host__ __device__ static T add(const T &a, const T &b) { return gf256to16_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256to16_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256to16_mul(a, b); }
};

struct gf256to16_add_op {
    __device__ gf256to16 operator()(const gf256to16 &a, const gf256to16 &b) const { return gf256to16_add(a, b); }
};
struct gf256to16_sub_op {
    __device__ gf256to16 operator()(const gf256to16 &a, const gf256to16 &b) const { return gf256to16_sub(a, b); }
};
struct gf256to16_mul_op {
    __device__ gf256to16 operator()(const gf256to16 &a, const gf256to16 &b) const { return gf256to16_mul(a, b); }
};

/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf256to16_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf256to16);
}

/// \return number of bytes of a nrows x ncols matrix
__host__ __device__ __forceinline__
size_t gf256to16_matrix_bytes_size(const uint32_t nrows,
                                   const uint32_t ncols) {
    return gf256to16_matrix_bytes_per_column(nrows) * ncols;
}

/// c = a + b element-wise, n = number of elements, device pointers
static inline
void gf256to16_vector_add(gf256to16 *c, const gf256to16 *a, const gf256to16 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to16_add_op{}, stream);
}

/// c = a - b element-wise, n = number of elements, device pointers
static inline
void gf256to16_vector_sub(gf256to16 *c, const gf256to16 *a, const gf256to16 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to16_sub_op{}, stream);
}

/// c = a * b element-wise, n = number of elements, device pointers
static inline
void gf256to16_vector_mul(gf256to16 *c, const gf256to16 *a, const gf256to16 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to16_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, nrows x ncols, device pointers
static inline
void gf256to16_matrix_add(gf256to16 *matrix1, const gf256to16 *matrix2, const gf256to16 *matrix3,
                          const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to16_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, nrows x ncols, device pointers
static inline
void gf256to16_matrix_sub(gf256to16 *matrix1, const gf256to16 *matrix2, const gf256to16 *matrix3,
                          const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to16_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, device pointers
/// \param result[out]: n_rows1 x n_cols2, must not alias the inputs
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf256to16_matrix_mul(gf256to16 *result, const gf256to16 *matrix1, const gf256to16 *matrix2,
                          const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2,
                          cudaStream_t stream = 0) {
    gf256to8_cuda::matrix_mul_tiled<gf256to16_field>(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256to16_cuda
