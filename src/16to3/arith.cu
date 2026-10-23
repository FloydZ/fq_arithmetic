#pragma once

/// CUDA implementation of GF(16^3) = GF(16)[X]/(X^3 + X + 1), with
/// GF(16) = GF(2)[x]/(x^4 + x + 1).
///
/// Representation (identical to `arith.h`/`matrix.h`):
///  - a single element is stored in an `uint16_t`, the coefficient of X^i
///    (i = 0, 1, 2) in nibble i. The upper nibble must be zero.
///  - vectors are arrays of `gf16to3`
///  - matrices are column major: entry (i, j) at index j*nrows + i.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"
#include "../16/arith.cu"

namespace gf16to3_cuda {

typedef uint16_t gf16to3;

/// number of elements of a column of B cached in shared memory per tile in
/// `gf16to3_matrix_mul`
constexpr uint32_t GF16TO3_CUDA_MUL_TILE_K = 1024;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf16to3_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf16to3);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf16to3_matrix_bytes_size(const uint32_t nrows,
                                 const uint32_t ncols) {
    return gf16to3_matrix_bytes_per_column(nrows) * ncols;
}

/// \return a+b
__host__ __device__ __forceinline__
gf16to3 gf16to3_add(const gf16to3 a,
                    const gf16to3 b) {
    return (gf16to3)(a ^ b);
}

/// \return a-b = a+b
__host__ __device__ __forceinline__
gf16to3 gf16to3_sub(const gf16to3 a,
                    const gf16to3 b) {
    return (gf16to3)(a ^ b);
}

/// Spreads the operand of the Karatsuba multiplication of `arith.h` into the
/// 6 nibbles [x0, x1, x2, x0+x1, x1+x2, x0+x2], so that a single packed
/// GF(16) multiplication computes all 6 partial products.
/// \param x[in]: element of GF(16^3)
/// \return packed nibbles
__host__ __device__ __forceinline__
uint32_t gf16to3_pack(const uint32_t x) {
    const uint32_t x01 = (x ^ (x >> 4u)) & 0xFu;
    const uint32_t x12 = ((x >> 4u) ^ (x >> 8u)) & 0xFu;
    const uint32_t x02 = (x ^ (x >> 8u)) & 0xFu;
    return (x & 0xFFFu) | (x01 << 12u) | (x12 << 16u) | (x02 << 20u);
}

/// Combines the 6 partial products [p0, p1, p2, p01, p12, p02] into the
/// reduced product. This map is linear, so packed products may be XOR
/// accumulated first and combined once.
/// \param p[in]: packed partial products
/// \return element of GF(16^3)
__host__ __device__ __forceinline__
gf16to3 gf16to3_combine(const uint32_t p) {
    const uint32_t p0  =  p         & 0xFu;
    const uint32_t p1  = (p >>  4u) & 0xFu;
    const uint32_t p2  = (p >>  8u) & 0xFu;
    const uint32_t p01 = (p >> 12u) & 0xFu;
    const uint32_t p12 = (p >> 16u) & 0xFu;
    const uint32_t p02 = (p >> 20u) & 0xFu;
    // X^3 = X + 1, X^4 = X^2 + X
    const uint32_t r0 = p0 ^ p1 ^ p2 ^ p12;
    const uint32_t r1 = p0 ^ p01 ^ p12;
    const uint32_t r2 = p0 ^ p1 ^ p02;
    return (gf16to3)(r0 | (r1 << 4u) | (r2 << 8u));
}

/// Karatsuba multiplication (same formula as `gf16to3_mul` in `arith.h`),
/// constant time and table free.
/// \return a*b
__host__ __device__ __forceinline__
gf16to3 gf16to3_mul(const gf16to3 a,
                    const gf16to3 b) {
    return gf16to3_combine(gf16_cuda::gf16v_mul_u32(gf16to3_pack(a), gf16to3_pack(b)));
}

/// element-wise operators for fq_cuda::elementwise
struct gf16to3_add_op {
    __device__ __forceinline__ gf16to3 operator()(const gf16to3 &a, const gf16to3 &b) const { return gf16to3_add(a, b); }
};
struct gf16to3_sub_op {
    __device__ __forceinline__ gf16to3 operator()(const gf16to3 &a, const gf16to3 &b) const { return gf16to3_sub(a, b); }
};
struct gf16to3_mul_op {
    __device__ __forceinline__ gf16to3 operator()(const gf16to3 &a, const gf16to3 &b) const { return gf16to3_mul(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to3_vector_add(gf16to3 *c,
                        const gf16to3 *a,
                        const gf16to3 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to3_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to3_vector_sub(gf16to3 *c,
                        const gf16to3 *a,
                        const gf16to3 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to3_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to3_vector_mul(gf16to3 *c,
                        const gf16to3 *a,
                        const gf16to3 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to3_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16to3_matrix_add(gf16to3 *matrix1,
                        const gf16to3 *matrix2,
                        const gf16to3 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf16to3_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16to3_matrix_sub(gf16to3 *matrix1,
                        const gf16to3 *matrix2,
                        const gf16to3 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf16to3_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of it (consecutive threads -> consecutive rows,
/// so all accesses to matrix1/result are coalesced). The column j of matrix2
/// is staged through shared memory in tiles of GF16TO3_CUDA_MUL_TILE_K
/// elements, already spread by `gf16to3_pack`. The packed partial products
/// are XOR accumulated and combined (= reduced) once per output entry.
static __global__
void gf16to3_matrix_mul_kernel(gf16to3 *__restrict__ result,
                               const gf16to3 *__restrict__ matrix1,
                               const gf16to3 *__restrict__ matrix2,
                               const uint32_t n_rows1,
                               const uint32_t n_cols1,
                               const uint32_t n_cols2) {
    __shared__ uint32_t sB[GF16TO3_CUDA_MUL_TILE_K];

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const gf16to3 *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;

            uint32_t acc = 0;
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF16TO3_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF16TO3_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = gf16to3_pack(Bj[k0 + t]);
                }
                __syncthreads();

                if (i < n_rows1) {
                    const gf16to3 *Aik = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Aik += n_rows1) {
                        acc ^= gf16_cuda::gf16v_mul_u32(gf16to3_pack(*Aik), sB[k]);
                    }
                }
            }

            if (i < n_rows1) {
                result[(size_t)j * n_rows1 + i] = gf16to3_combine(acc);
            }
        }
    }
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
void gf16to3_matrix_mul(gf16to3 *result,
                        const gf16to3 *matrix1,
                        const gf16to3 *matrix2,
                        const uint32_t n_rows1,
                        const uint32_t n_cols1,
                        const uint32_t n_cols2,
                        cudaStream_t stream = 0) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t threads = 128;
    uint32_t gx = (n_rows1 + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    gf16to3_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf16to3_cuda
