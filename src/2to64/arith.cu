#pragma once

/// CUDA implementation of GF(2^64) = GF(2)[x]/(x^64 + x^4 + x^3 + x + 1).
///
/// Representation (identical to `arith.h`/`vector.h`/`matrix.h`):
///  - one element per `uint64_t`, bit i is the coefficient of x^i
///  - vectors: arrays of elements
///  - matrices: column major, entry (i,j) at index j*nrows + i
///
/// Multiplication: GPUs have no carry-less multiply instruction, so the
/// product is computed constant time by shift-and-add (64 masked XORs of the
/// shifted operand into a 128-bit accumulator held as two 64-bit limbs, no
/// __int128 on the device), followed by a two-step fold reduction using
/// x^64 = x^4 + x^3 + x + 1. Since the reduction is linear,
/// `gf2to64_matrix_mul` XORs the unreduced 127-bit products and reduces only
/// once per output entry.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Everything lives in `namespace gf2to64_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf2to64_cuda {

typedef uint64_t gf2to64;

/// x^64 + x^4 + x^3 + x + 1 without the leading term (= `MODULUS` in arith.h)
constexpr uint64_t GF2TO64_CUDA_MODULUS = 0x1Bu;

/// number of elements of a column of B cached in shared memory per tile in
/// `gf2to64_matrix_mul`
constexpr uint32_t GF2TO64_CUDA_MUL_TILE_K = 1024;

/// unreduced product of two elements: hi * x^64 + lo, degree <= 126
struct gf2to64_wide {
    uint64_t lo, hi;
};

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf2to64_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows * (uint32_t)sizeof(gf2to64);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf2to64_matrix_bytes_size(const uint32_t nrows,
                                 const uint32_t ncols) {
    return (size_t)nrows * ncols * sizeof(gf2to64);
}

/// Addition in GF(2^64)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
gf2to64 gf2to64_add(const gf2to64 a,
                    const gf2to64 b) {
    return a ^ b;
}

/// Subtraction in GF(2^64), identical to the addition (characteristic 2)
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
gf2to64 gf2to64_sub(const gf2to64 a,
                    const gf2to64 b) {
    return a ^ b;
}

/// Carry-less product of two polynomials of degree < 64, constant time.
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b in GF(2)[x], degree <= 126
__host__ __device__ __forceinline__
gf2to64_wide gf2to64_clmul(const gf2to64 a,
                           const gf2to64 b) {
    gf2to64_wide r = {0, 0};
    const uint64_t a1 = a >> 1u;
#pragma unroll 8
    for (uint32_t i = 0; i < 64; i++) {
        const uint64_t m = 0ull - ((b >> i) & 1ull);
        r.lo ^= m & (a << i);
        // the bits of a shifted out of the low limb: a >> (64 - i), written
        // as (a >> 1) >> (63 - i) to avoid the undefined shift by 64 for i = 0
        r.hi ^= m & (a1 >> (63u - i));
    }
    return r;
}

/// Reduction modulo x^64 + x^4 + x^3 + x + 1
/// \param r[in]: polynomial of degree <= 126
/// \return r mod the field polynomial
__host__ __device__ __forceinline__
gf2to64 gf2to64_reduce(const gf2to64_wide r) {
    const uint64_t h = r.hi;
    // bits of h*(x^4 + x^3 + x) of degree >= 64; degree <= 3
    const uint64_t c = (h >> 60u) ^ (h >> 61u) ^ (h >> 63u);
    uint64_t lo = r.lo ^ h ^ (h << 1u) ^ (h << 3u) ^ (h << 4u);
    lo ^= c ^ (c << 1u) ^ (c << 3u) ^ (c << 4u);
    return lo;
}

/// Multiplication in GF(2^64), constant time
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
gf2to64 gf2to64_mul(const gf2to64 a,
                    const gf2to64 b) {
    return gf2to64_reduce(gf2to64_clmul(a, b));
}

/// element-wise operators for `fq_cuda::elementwise`
struct gf2to64_add_op {
    __device__ __forceinline__ gf2to64 operator()(const gf2to64 a, const gf2to64 b) const { return gf2to64_add(a, b); }
};
struct gf2to64_sub_op {
    __device__ __forceinline__ gf2to64 operator()(const gf2to64 a, const gf2to64 b) const { return gf2to64_sub(a, b); }
};
struct gf2to64_mul_op {
    __device__ __forceinline__ gf2to64 operator()(const gf2to64 a, const gf2to64 b) const { return gf2to64_mul(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to64_vector_add(gf2to64 *c,
                        const gf2to64 *a,
                        const gf2to64 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to64_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to64_vector_sub(gf2to64 *c,
                        const gf2to64 *a,
                        const gf2to64 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to64_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to64_vector_mul(gf2to64 *c,
                        const gf2to64 *a,
                        const gf2to64 *b,
                        const uint32_t n,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to64_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to64_matrix_add(gf2to64 *matrix1,
                        const gf2to64 *matrix2,
                        const gf2to64 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to64_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to64_matrix_sub(gf2to64 *matrix1,
                        const gf2to64 *matrix2,
                        const gf2to64 *matrix3,
                        const uint32_t nrows,
                        const uint32_t ncols,
                        cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to64_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of that column (consecutive threads read
/// consecutive rows of matrix1, i.e. coalesced). Column j of matrix2 is staged
/// through shared memory in tiles of GF2TO64_CUDA_MUL_TILE_K elements.
static __global__
void gf2to64_matrix_mul_kernel(gf2to64 *__restrict__ result,
                               const gf2to64 *__restrict__ matrix1,
                               const gf2to64 *__restrict__ matrix2,
                               const uint32_t n_rows1,
                               const uint32_t n_cols1,
                               const uint32_t n_cols2) {
    __shared__ gf2to64 sB[GF2TO64_CUDA_MUL_TILE_K];

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const gf2to64 *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;

            gf2to64_wide acc = {0, 0}; // unreduced sum of products
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF2TO64_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF2TO64_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (i < n_rows1) {
                    const gf2to64 *Aik = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Aik += n_rows1) {
                        const gf2to64_wide p = gf2to64_clmul(*Aik, sB[k]);
                        acc.lo ^= p.lo;
                        acc.hi ^= p.hi;
                    }
                }
            }

            if (i < n_rows1) {
                result[(size_t)j * n_rows1 + i] = gf2to64_reduce(acc);
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
void gf2to64_matrix_mul(gf2to64 *result,
                        const gf2to64 *matrix1,
                        const gf2to64 *matrix2,
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

    gf2to64_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf2to64_cuda
