#pragma once

/// CUDA implementation of GF(2^12) = GF(2)[x]/(x^12 + x^3 + 1).
///
/// Representation (identical to `arith.h`/`vector.h`/`matrix.h`):
///  - one element per `uint16_t`, bit i is the coefficient of x^i, bits 12..15
///    are zero
///  - vectors: arrays of elements
///  - matrices: column major, entry (i,j) at index j*nrows + i
///
/// Multiplication: GPUs have no carry-less multiply instruction, so the
/// product is computed constant time by shift-and-add (12 masked XORs of the
/// shifted operand into a 32-bit accumulator), followed by a two-step fold
/// reduction using x^12 = x^3 + 1. Since the reduction is linear,
/// `gf2to12_matrix_mul` XORs the unreduced 23-bit products and reduces only
/// once per output entry.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Everything lives in `namespace gf2to12_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf2to12_cuda {

typedef uint16_t gf2to12;

/// x^12 + x^3 + 1 (= `MODULUS` in arith.h)
constexpr uint32_t GF2TO12_CUDA_MODULUS = 0x1009u;

/// mask of the valid bits of an element
constexpr uint32_t GF2TO12_CUDA_MASK = 0xFFFu;

/// number of elements of a column of B cached in shared memory per tile in
/// `gf2to12_matrix_mul`
constexpr uint32_t GF2TO12_CUDA_MUL_TILE_K = 2048;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf2to12_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows * (uint32_t)sizeof(gf2to12);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf2to12_matrix_bytes_size(const uint32_t nrows,
                                 const uint32_t ncols) {
    return (size_t)nrows * ncols * sizeof(gf2to12);
}

/// Addition in GF(2^12)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
gf2to12 gf2to12_add(const gf2to12 a,
                    const gf2to12 b) {
    return (gf2to12)(a ^ b);
}

/// Subtraction in GF(2^12), identical to the addition (characteristic 2)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
gf2to12 gf2to12_sub(const gf2to12 a,
                    const gf2to12 b) {
    return (gf2to12)(a ^ b);
}

/// Carry-less product of two polynomials of degree < 12, constant time.
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2)[x], degree <= 22
__host__ __device__ __forceinline__
uint32_t gf2to12_clmul(const gf2to12 a,
                       const gf2to12 b) {
    uint32_t r = 0;
#pragma unroll
    for (uint32_t i = 0; i < 12; i++) {
        r ^= (0u - (((uint32_t)b >> i) & 1u)) & ((uint32_t)a << i);
    }
    return r;
}

/// Reduction modulo x^12 + x^3 + 1
/// \param r[in]: polynomial of degree <= 22
/// \return r mod the field polynomial
__host__ __device__ __forceinline__
gf2to12 gf2to12_reduce(uint32_t r) {
    // first fold: h has degree <= 10, h*(x^3 + 1) degree <= 13
    uint32_t h = r >> 12u;
    r = (r & GF2TO12_CUDA_MASK) ^ h ^ (h << 3u);
    // second fold: h has degree <= 1, the result degree <= 11
    h = r >> 12u;
    r = (r & GF2TO12_CUDA_MASK) ^ h ^ (h << 3u);
    return (gf2to12)r;
}

/// Multiplication in GF(2^12), constant time
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
gf2to12 gf2to12_mul(const gf2to12 a,
                    const gf2to12 b) {
    return gf2to12_reduce(gf2to12_clmul(a, b));
}

/// element-wise operators for `fq_cuda::elementwise`
struct gf2to12_add_op {
    __device__ __forceinline__ gf2to12 operator()(const gf2to12 a, const gf2to12 b) const { return gf2to12_add(a, b); }
};
struct gf2to12_sub_op {
    __device__ __forceinline__ gf2to12 operator()(const gf2to12 a, const gf2to12 b) const { return gf2to12_sub(a, b); }
};
struct gf2to12_mul_op {
    __device__ __forceinline__ gf2to12 operator()(const gf2to12 a, const gf2to12 b) const { return gf2to12_mul(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to12_vector_add(gf2to12 *c,
                        const gf2to12 *a,
                        const gf2to12 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to12_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to12_vector_sub(gf2to12 *c,
                        const gf2to12 *a,
                        const gf2to12 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to12_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to12_vector_mul(gf2to12 *c,
                        const gf2to12 *a,
                        const gf2to12 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to12_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to12_matrix_add(gf2to12 *matrix1,
                        const gf2to12 *matrix2,
                        const gf2to12 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to12_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to12_matrix_sub(gf2to12 *matrix1,
                        const gf2to12 *matrix2,
                        const gf2to12 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to12_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of that column (consecutive threads read
/// consecutive rows of matrix1, i.e. coalesced). Column j of matrix2 is staged
/// through shared memory in tiles of GF2TO12_CUDA_MUL_TILE_K elements.
static __global__
void gf2to12_matrix_mul_kernel(gf2to12 *__restrict__ result,
                               const gf2to12 *__restrict__ matrix1,
                               const gf2to12 *__restrict__ matrix2,
                               const uint32_t n_rows1,
                               const uint32_t n_cols1,
                               const uint32_t n_cols2) {
    __shared__ gf2to12 sB[GF2TO12_CUDA_MUL_TILE_K];

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const gf2to12 *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;

            uint32_t acc = 0; // unreduced sum of products
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF2TO12_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF2TO12_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (i < n_rows1) {
                    const gf2to12 *Aik = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Aik += n_rows1) {
                        acc ^= gf2to12_clmul(*Aik, sB[k]);
                    }
                }
            }

            if (i < n_rows1) {
                result[(size_t)j * n_rows1 + i] = gf2to12_reduce(acc);
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
void gf2to12_matrix_mul(gf2to12 *result,
                        const gf2to12 *matrix1,
                        const gf2to12 *matrix2,
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

    gf2to12_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf2to12_cuda
