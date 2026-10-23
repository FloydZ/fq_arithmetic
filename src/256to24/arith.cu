#pragma once

/// CUDA implementation of GF(256^24) = GF(256^8)[X]/(X^3 + X + 1), the same
/// representation as `arith.h` (see ../256to8/arith.cu for GF(256^8)).
///
/// An element is stored exactly like the C type `uint8_t[24]`: three little
/// endian GF(256^8) limbs v[0], v[1], v[2], a = a_2*X^2 + a_1*X + a_0.
/// `gf256to24` has the same size, but an alignment of 8, so device pointers to
/// vectors/matrices must be 8-byte aligned (cudaMalloc is).
///
/// Vectors/matrices: plain arrays of elements, matrices column-major
/// (entry (i, j) at j*nrows + i). All host wrappers take DEVICE pointers,
/// enqueue on `stream` and do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"
#include "../256to8/arith.cu"

namespace gf256to24_cuda {

struct gf256to24 {
    uint64_t v[3];
};
static_assert(sizeof(gf256to24) == 24, "must match the C type uint8_t[24]");

/// \return a+b
__host__ __device__ __forceinline__
gf256to24 gf256to24_add(const gf256to24 &a,
                        const gf256to24 &b) {
    gf256to24 r;
    r.v[0] = a.v[0] ^ b.v[0];
    r.v[1] = a.v[1] ^ b.v[1];
    r.v[2] = a.v[2] ^ b.v[2];
    return r;
}

/// \return a-b
__host__ __device__ __forceinline__
gf256to24 gf256to24_sub(const gf256to24 &a,
                        const gf256to24 &b) {
    return gf256to24_add(a, b);
}

/// Karatsuba with 6 GF(256^8) products, reduced by X^3 + X + 1:
///     r_0 = p0 + p1 + p2 + p12
///     r_1 = p0 + p01 + p12
///     r_2 = p012 + p01 + p12 + p2
/// \return a*b
__host__ __device__ __forceinline__
gf256to24 gf256to24_mul(const gf256to24 &a,
                        const gf256to24 &b) {
    using namespace gf256to8_cuda;
    const gf256to8 a01 = a.v[0] ^ a.v[1], a12 = a.v[1] ^ a.v[2], a012 = a01 ^ a.v[2];
    const gf256to8 b01 = b.v[0] ^ b.v[1], b12 = b.v[1] ^ b.v[2], b012 = b01 ^ b.v[2];

    const gf256to8 p0   = gf256to8_mul(a.v[0], b.v[0]);
    const gf256to8 p1   = gf256to8_mul(a.v[1], b.v[1]);
    const gf256to8 p2   = gf256to8_mul(a.v[2], b.v[2]);
    const gf256to8 p01  = gf256to8_mul(a01, b01);
    const gf256to8 p12  = gf256to8_mul(a12, b12);
    const gf256to8 p012 = gf256to8_mul(a012, b012);

    gf256to24 r;
    r.v[0] = p0 ^ p1 ^ p2 ^ p12;
    r.v[1] = p0 ^ p01 ^ p12;
    r.v[2] = p012 ^ p01 ^ p12 ^ p2;
    return r;
}

/// field policy for the generic kernels
struct gf256to24_field {
    typedef gf256to24 T;
    __host__ __device__ static T zero() { T r; r.v[0] = 0; r.v[1] = 0; r.v[2] = 0; return r; }
    __host__ __device__ static T add(const T &a, const T &b) { return gf256to24_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256to24_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256to24_mul(a, b); }
};

struct gf256to24_add_op {
    __device__ gf256to24 operator()(const gf256to24 &a, const gf256to24 &b) const { return gf256to24_add(a, b); }
};
struct gf256to24_sub_op {
    __device__ gf256to24 operator()(const gf256to24 &a, const gf256to24 &b) const { return gf256to24_sub(a, b); }
};
struct gf256to24_mul_op {
    __device__ gf256to24 operator()(const gf256to24 &a, const gf256to24 &b) const { return gf256to24_mul(a, b); }
};

/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf256to24_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf256to24);
}

/// \return number of bytes of a nrows x ncols matrix
__host__ __device__ __forceinline__
size_t gf256to24_matrix_bytes_size(const uint32_t nrows,
                                   const uint32_t ncols) {
    return gf256to24_matrix_bytes_per_column(nrows) * ncols;
}

/// c = a + b element-wise, n = number of elements, device pointers
static inline
void gf256to24_vector_add(gf256to24 *c, const gf256to24 *a, const gf256to24 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to24_add_op{}, stream);
}

/// c = a - b element-wise, n = number of elements, device pointers
static inline
void gf256to24_vector_sub(gf256to24 *c, const gf256to24 *a, const gf256to24 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to24_sub_op{}, stream);
}

/// c = a * b element-wise, n = number of elements, device pointers
static inline
void gf256to24_vector_mul(gf256to24 *c, const gf256to24 *a, const gf256to24 *b,
                          const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to24_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, nrows x ncols, device pointers
static inline
void gf256to24_matrix_add(gf256to24 *matrix1, const gf256to24 *matrix2, const gf256to24 *matrix3,
                          const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to24_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, nrows x ncols, device pointers
static inline
void gf256to24_matrix_sub(gf256to24 *matrix1, const gf256to24 *matrix2, const gf256to24 *matrix3,
                          const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to24_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, device pointers
/// \param result[out]: n_rows1 x n_cols2, must not alias the inputs
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf256to24_matrix_mul(gf256to24 *result, const gf256to24 *matrix1, const gf256to24 *matrix2,
                          const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2,
                          cudaStream_t stream = 0) {
    gf256to8_cuda::matrix_mul_tiled<gf256to24_field>(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256to24_cuda
