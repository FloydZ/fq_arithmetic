#pragma once

/// CUDA implementation of GF(127) arithmetic.
///
/// Representation (identical to `arith.h`/`matrix.h`):
///  - one element per `uint8_t`, canonical value in [0, 127)
///  - matrices are column major: entry (i, j) is stored at `j * nrows + i`
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Everything lives in `namespace gf127_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).
/// NOTE: `arith.h` defines the macros `PRIME` and `PRIME_LOG_CEIL`, so this
/// file never uses those names.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf127_cuda {

using gf127 = uint8_t;

/// the characteristic of the field
constexpr uint32_t GF127_P = 127;

/// number of elements of a column of B cached in shared memory per tile in
/// `gf127_matrix_mul`. The accumulators are reduced once per tile.
constexpr uint32_t GF127_CUDA_MUL_TILE_K = 2048;

/// Number of products of two bytes (each <= 255, so this even covers
/// non-canonical inputs) that a uint32_t accumulator, which holds a reduced
/// value < GF127_P, can absorb without overflowing.
constexpr uint32_t GF127_CUDA_MAX_LAZY_TERMS = (0xFFFFFFFFu - (GF127_P - 1u)) / (255u * 255u);
static_assert(GF127_CUDA_MUL_TILE_K <= GF127_CUDA_MAX_LAZY_TERMS,
              "the lazy reduction in gf127_matrix_mul would overflow");

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf127_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows;
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf127_matrix_bytes_size(const uint32_t nrows,
                               const uint32_t ncols) {
    return (size_t)nrows * ncols;
}

/// \param x[in]: any 32-bit value
/// \return x mod 127. The modulus is a compile time constant, so the compiler
///         emits a multiply-high + shift sequence instead of a division.
__host__ __device__ __forceinline__
uint32_t gf127_reduce(const uint32_t x) {
    return x % GF127_P;
}

/// Addition in GF(127), constant time
/// \param a[in]: first addend in [0, 127)
/// \param b[in]: second addend in [0, 127)
/// \return a+b mod 127
__host__ __device__ __forceinline__
gf127 gf127_add(const gf127 a,
                const gf127 b) {
    // a + b - 127 is negative iff a + b < 127: add 127 back in that case
    uint32_t s = (uint32_t)a + b - GF127_P;
    s += GF127_P & (0u - (s >> 31u));
    return (gf127)s;
}

/// Subtraction in GF(127), constant time
/// \param a[in]: minuend in [0, 127)
/// \param b[in]: subtrahend in [0, 127)
/// \return a-b mod 127
__host__ __device__ __forceinline__
gf127 gf127_sub(const gf127 a,
                const gf127 b) {
    uint32_t s = (uint32_t)a - b;
    s += GF127_P & (0u - (s >> 31u));
    return (gf127)s;
}

/// Multiplication in GF(127)
/// \param a[in]: first factor in [0, 127)
/// \param b[in]: second factor in [0, 127)
/// \return a*b mod 127
__host__ __device__ __forceinline__
gf127 gf127_mul(const gf127 a,
                const gf127 b) {
    return (gf127)gf127_reduce((uint32_t)a * b);
}

/// element-wise operators used by the kernels below
struct gf127_add_op {
    __device__ __forceinline__ uint8_t operator()(const uint8_t a, const uint8_t b) const { return gf127_add(a, b); }
};
struct gf127_sub_op {
    __device__ __forceinline__ uint8_t operator()(const uint8_t a, const uint8_t b) const { return gf127_sub(a, b); }
};
struct gf127_mul_op {
    __device__ __forceinline__ uint8_t operator()(const uint8_t a, const uint8_t b) const { return gf127_mul(a, b); }
};

/// applies `op` to each of the 4 bytes (= elements) of a and b
template<typename Op>
__device__ __forceinline__
uint32_t gf127_bytewise(const uint32_t a,
                        const uint32_t b,
                        const Op op) {
    uint32_t r = 0;
#pragma unroll
    for (uint32_t k = 0; k < 32; k += 8) {
        r |= (uint32_t)op((uint8_t)(a >> k), (uint8_t)(b >> k)) << k;
    }
    return r;
}

/// c[i] = op(a[i], b[i]) on 16-byte chunks (16 elements), grid-stride loop
template<typename Op>
__global__ void gf127_elementwise_u128_kernel(uint4 *c,
                                              const uint4 *a,
                                              const uint4 *b,
                                              const size_t n,
                                              const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 x = a[i], y = b[i];
        uint4 z;
        z.x = gf127_bytewise(x.x, y.x, op);
        z.y = gf127_bytewise(x.y, y.y, op);
        z.z = gf127_bytewise(x.z, y.z, op);
        z.w = gf127_bytewise(x.w, y.w, op);
        c[i] = z;
    }
}

/// c = op(a, b) on `n` elements. Uses 16-byte accesses if all pointers are
/// 16-byte aligned, single bytes for the rest. c may alias a and/or b.
template<typename Op>
static inline
void gf127_elementwise(uint8_t *c,
                       const uint8_t *a,
                       const uint8_t *b,
                       const size_t n,
                       const Op op,
                       cudaStream_t stream) {
    if (n == 0) {
        return;
    }

    size_t done = 0;
    const uintptr_t align = (uintptr_t)a | (uintptr_t)b | (uintptr_t)c;
    if ((align & 15u) == 0) {
        const size_t n16 = n / 16;
        if (n16) {
            gf127_elementwise_u128_kernel<Op><<<fq_cuda::grid_size(n16), fq_cuda::BLOCK_SIZE, 0, stream>>>
                ((uint4 *)c, (const uint4 *)a, (const uint4 *)b, n16, op);
            FQ_CUDA_CHECK(cudaGetLastError());
        }
        done = n16 * 16;
    }

    if (done < n) {
        fq_cuda::elementwise<uint8_t, Op>(c + done, a + done, b + done, n - done, op, stream);
    }
}

/// c = a + b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf127_vector_add(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    gf127_elementwise(c, a, b, n, gf127_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf127_vector_sub(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    gf127_elementwise(c, a, b, n, gf127_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf127_vector_mul(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    gf127_elementwise(c, a, b, n, gf127_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf127_matrix_add(uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint8_t *matrix3,
                      const uint32_t nrows,
                      const uint32_t ncols,
                      cudaStream_t stream = 0) {
    gf127_elementwise(matrix1, matrix2, matrix3, gf127_matrix_bytes_size(nrows, ncols), gf127_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf127_matrix_sub(uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint8_t *matrix3,
                      const uint32_t nrows,
                      const uint32_t ncols,
                      cudaStream_t stream = 0) {
    gf127_elementwise(matrix1, matrix2, matrix3, gf127_matrix_bytes_size(nrows, ncols), gf127_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes 4 consecutive rows of that column. The column j of
/// matrix2 is staged through shared memory in tiles of GF127_CUDA_MUL_TILE_K
/// elements. Products are accumulated lazily in uint32_t and reduced once
/// per tile (see GF127_CUDA_MAX_LAZY_TERMS).
/// \param aligned4[in]: matrix1 and result are 4-byte aligned and n_rows1 is
///                      a multiple of 4, so 32-bit loads/stores can be used
static __global__
void gf127_matrix_mul_kernel(uint8_t *__restrict__ result,
                             const uint8_t *__restrict__ matrix1,
                             const uint8_t *__restrict__ matrix2,
                             const uint32_t n_rows1,
                             const uint32_t n_cols1,
                             const uint32_t n_cols2,
                             const bool aligned4) {
    __shared__ uint8_t sB[GF127_CUDA_MUL_TILE_K];

    const uint32_t nwords = (n_rows1 + 3u) / 4u;

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const uint8_t *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t w0 = blockIdx.x * blockDim.x; w0 < nwords; w0 += gridDim.x * blockDim.x) {
            const uint32_t w = w0 + threadIdx.x;
            const uint32_t row0 = 4u * w;
            const uint32_t nb = (w < nwords) ? min(4u, n_rows1 - row0) : 0u;

            uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF127_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF127_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (nb) {
                    const uint8_t *Ak = matrix1 + (size_t)k0 * n_rows1 + row0;
                    for (uint32_t k = 0; k < kn; k++, Ak += n_rows1) {
                        const uint32_t s = sB[k];

                        uint32_t a;
                        if (aligned4) {
                            a = *(const uint32_t *)Ak;
                        } else {
                            a = Ak[0];
                            if (nb > 1) { a |= (uint32_t)Ak[1] << 8u; }
                            if (nb > 2) { a |= (uint32_t)Ak[2] << 16u; }
                            if (nb > 3) { a |= (uint32_t)Ak[3] << 24u; }
                        }

                        acc0 += ( a         & 0xFFu) * s;
                        acc1 += ((a >>  8u) & 0xFFu) * s;
                        acc2 += ((a >> 16u) & 0xFFu) * s;
                        acc3 += ( a >> 24u         ) * s;
                    }

                    // at most GF127_CUDA_MUL_TILE_K <= GF127_CUDA_MAX_LAZY_TERMS
                    // products were added to values < GF127_P: no overflow
                    acc0 = gf127_reduce(acc0);
                    acc1 = gf127_reduce(acc1);
                    acc2 = gf127_reduce(acc2);
                    acc3 = gf127_reduce(acc3);
                }
            }

            if (nb) {
                const uint32_t r = acc0 | (acc1 << 8u) | (acc2 << 16u) | (acc3 << 24u);
                uint8_t *Cj = result + (size_t)j * n_rows1 + row0;
                if (aligned4) {
                    *(uint32_t *)Cj = r;
                } else {
                    for (uint32_t t = 0; t < nb; t++) {
                        Cj[t] = (uint8_t)(r >> (8u * t));
                    }
                }
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
void gf127_matrix_mul(uint8_t *result,
                      const uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint32_t n_rows1,
                      const uint32_t n_cols1,
                      const uint32_t n_cols2,
                      cudaStream_t stream = 0) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t nwords = (n_rows1 + 3u) / 4u;
    const bool aligned4 = ((n_rows1 & 3u) == 0) &&
                          ((((uintptr_t)matrix1) | ((uintptr_t)result)) & 3u) == 0;

    const uint32_t threads = 128;
    uint32_t gx = (nwords + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    gf127_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, aligned4);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf127_cuda
