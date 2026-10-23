#pragma once

/// CUDA implementation of GF(16) = GF(2)[x]/(x^4 + x + 1) arithmetic.
///
/// Representation (identical to `arith.h`/`matrix.h`):
///  - a single element is stored in the low nibble of an `uint8_t`
///  - vectors/matrices are nibble packed: 2 elements per byte, the element with
///    the even index lives in the low nibble.
///  - matrices are column major: column j starts at byte
///    `gf16_matrix_bytes_per_column(nrows) * j`, with
///    `gf16_matrix_bytes_per_column(nrows) = ceil(nrows / 2)`. For an odd
///    number of rows the high nibble of the last byte of each column is padding
///    and is expected to be (and kept) zero.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Everything lives in `namespace gf16_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf16_cuda {

/// number of elements of a column of B cached in shared memory per tile in
/// `gf16_matrix_mul` (must be even)
constexpr uint32_t GF16_CUDA_MUL_TILE_K = 2048;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single nibble packed column
__host__ __device__ __forceinline__
uint32_t gf16_matrix_bytes_per_column(const uint32_t nrows) {
    return (nrows >> 1u) + (nrows & 1u);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a nibble packed column major matrix
__host__ __device__ __forceinline__
size_t gf16_matrix_bytes_size(const uint32_t nrows,
                              const uint32_t ncols) {
    return (size_t)gf16_matrix_bytes_per_column(nrows) * ncols;
}

/// Addition in GF(16). Also works on packed bytes/words.
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
uint8_t gf16_add(const uint8_t a,
                 const uint8_t b) {
    return a ^ b;
}

/// Subtraction in GF(16), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
uint8_t gf16_sub(const uint8_t a,
                 const uint8_t b) {
    return a ^ b;
}

/// Lane-wise multiplication of 8 GF(16) elements packed into 32 bits
/// (one element per nibble). Constant time, no lookup tables.
/// \param a[in]: 8 packed elements
/// \param b[in]: 8 packed elements
/// \return [a_0*b_0, ..., a_7*b_7] packed
__host__ __device__ __forceinline__
uint32_t gf16v_mul_u32(uint32_t a,
                       const uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        // 0xF in every nibble whose i-th bit of b is set
        const uint32_t m = ((b >> i) & 0x11111111u) * 0xFu;
        r ^= a & m;
        // a = a*x mod x^4 + x + 1, in every nibble
        a = ((a & 0x77777777u) << 1u) ^ (((a >> 3u) & 0x11111111u) * 3u);
    }
    return r;
}

/// Multiplication in GF(16).
/// \param a[in]: first factor (low nibble)
/// \param b[in]: second factor (low nibble)
/// \return a*b
__host__ __device__ __forceinline__
uint8_t gf16_mul(const uint8_t a,
                 const uint8_t b) {
    return (uint8_t)gf16v_mul_u32(a & 0xFu, b & 0xFu);
}

/// Lane-wise multiplication of two bytes, each holding 2 packed elements.
/// \param a[in]: 2 packed elements
/// \param b[in]: 2 packed elements
/// \return [a_0*b_0, a_1*b_1] packed
__host__ __device__ __forceinline__
uint8_t gf16v_mul_u8(const uint8_t a,
                     const uint8_t b) {
    return (uint8_t)gf16v_mul_u32(a, b);
}

/// element-wise operators used by the generic kernels below. They operate on
/// 32-bit words of packed elements; for single bytes the upper 24 bits are 0.
struct gf16_add_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a ^ b; }
};
struct gf16_sub_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a ^ b; }
};
struct gf16_mul_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return gf16v_mul_u32(a, b); }
};

/// c[i] = op(a[i], b[i]) on 16-byte chunks, grid-stride loop
template<typename Op>
__global__ void gf16_elementwise_u128_kernel(uint4 *c,
                                             const uint4 *a,
                                             const uint4 *b,
                                             const size_t n,
                                             const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 x = a[i], y = b[i];
        uint4 z;
        z.x = op(x.x, y.x);
        z.y = op(x.y, y.y);
        z.z = op(x.z, y.z);
        z.w = op(x.w, y.w);
        c[i] = z;
    }
}

/// c[i] = op(a[i], b[i]) on single bytes, grid-stride loop
template<typename Op>
__global__ void gf16_elementwise_u8_kernel(uint8_t *c,
                                           const uint8_t *a,
                                           const uint8_t *b,
                                           const size_t n,
                                           const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = (uint8_t)op(a[i], b[i]);
    }
}

/// c = op(a, b) on `nbytes` bytes of packed elements. Uses 16-byte accesses
/// if all pointers are 16-byte aligned, single bytes for the rest.
/// c may alias a and/or b.
template<typename Op>
static inline
void gf16_elementwise(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const size_t nbytes,
                      const Op op,
                      cudaStream_t stream) {
    if (nbytes == 0) {
        return;
    }

    size_t done = 0;
    const uintptr_t align = (uintptr_t)a | (uintptr_t)b | (uintptr_t)c;
    if ((align & 15u) == 0) {
        const size_t n16 = nbytes / 16;
        if (n16) {
            gf16_elementwise_u128_kernel<Op><<<fq_cuda::grid_size(n16), fq_cuda::BLOCK_SIZE, 0, stream>>>
                ((uint4 *)c, (const uint4 *)a, (const uint4 *)b, n16, op);
            FQ_CUDA_CHECK(cudaGetLastError());
        }
        done = n16 * 16;
    }

    if (done < nbytes) {
        gf16_elementwise_u8_kernel<Op><<<fq_cuda::grid_size(nbytes - done), fq_cuda::BLOCK_SIZE, 0, stream>>>
            (c + done, a + done, b + done, nbytes - done, op);
        FQ_CUDA_CHECK(cudaGetLastError());
    }
}

/// c = a + b, element-wise on nibble packed vectors
/// \param c[out]: device pointer, ceil(n/2) bytes
/// \param a[in]: device pointer, ceil(n/2) bytes
/// \param b[in]: device pointer, ceil(n/2) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16_vector_add(uint8_t *c,
                     const uint8_t *a,
                     const uint8_t *b,
                     const uint32_t n,
                     cudaStream_t stream = 0) {
    gf16_elementwise(c, a, b, gf16_matrix_bytes_per_column(n), gf16_add_op{}, stream);
}

/// c = a - b, element-wise on nibble packed vectors
/// \param c[out]: device pointer, ceil(n/2) bytes
/// \param a[in]: device pointer, ceil(n/2) bytes
/// \param b[in]: device pointer, ceil(n/2) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16_vector_sub(uint8_t *c,
                     const uint8_t *a,
                     const uint8_t *b,
                     const uint32_t n,
                     cudaStream_t stream = 0) {
    gf16_elementwise(c, a, b, gf16_matrix_bytes_per_column(n), gf16_sub_op{}, stream);
}

/// c = a * b, element-wise on nibble packed vectors
/// \param c[out]: device pointer, ceil(n/2) bytes
/// \param a[in]: device pointer, ceil(n/2) bytes
/// \param b[in]: device pointer, ceil(n/2) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16_vector_mul(uint8_t *c,
                     const uint8_t *a,
                     const uint8_t *b,
                     const uint32_t n,
                     cudaStream_t stream = 0) {
    gf16_elementwise(c, a, b, gf16_matrix_bytes_per_column(n), gf16_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major, nibble packed
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16_matrix_add(uint8_t *matrix1,
                     const uint8_t *matrix2,
                     const uint8_t *matrix3,
                     const uint32_t nrows,
                     const uint32_t ncols,
                     cudaStream_t stream = 0) {
    gf16_elementwise(matrix1, matrix2, matrix3, gf16_matrix_bytes_size(nrows, ncols), gf16_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major, nibble packed
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16_matrix_sub(uint8_t *matrix1,
                     const uint8_t *matrix2,
                     const uint8_t *matrix3,
                     const uint32_t nrows,
                     const uint32_t ncols,
                     cudaStream_t stream = 0) {
    gf16_elementwise(matrix1, matrix2, matrix3, gf16_matrix_bytes_size(nrows, ncols), gf16_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, nibble packed.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes 4 consecutive bytes (8 rows) of that column. The column
/// j of matrix2 is staged through shared memory in tiles of
/// GF16_CUDA_MUL_TILE_K elements.
/// \param aligned4[in]: matrix1 is 4-byte aligned and its column size is a
///                      multiple of 4 bytes, so 32-bit loads can be used
static __global__
void gf16_matrix_mul_kernel(uint8_t *__restrict__ result,
                            const uint8_t *__restrict__ matrix1,
                            const uint8_t *__restrict__ matrix2,
                            const uint32_t n_rows1,
                            const uint32_t n_cols1,
                            const uint32_t n_cols2,
                            const bool aligned4) {
    __shared__ uint8_t sB[GF16_CUDA_MUL_TILE_K / 2];

    const uint32_t bpc1 = gf16_matrix_bytes_per_column(n_rows1); // also of result
    const uint32_t bpc2 = gf16_matrix_bytes_per_column(n_cols1);
    const uint32_t nwords = (bpc1 + 3u) / 4u;

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const uint8_t *Bj = matrix2 + (size_t)j * bpc2;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t w0 = blockIdx.x * blockDim.x; w0 < nwords; w0 += gridDim.x * blockDim.x) {
            const uint32_t w = w0 + threadIdx.x;
            const uint32_t byte0 = 4u * w;
            const uint32_t nb = (w < nwords) ? min(4u, bpc1 - byte0) : 0u;

            uint32_t acc = 0;
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF16_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF16_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < (kn + 1u) / 2u; t += blockDim.x) {
                    sB[t] = Bj[k0 / 2u + t];
                }
                __syncthreads();

                if (nb) {
                    const uint8_t *Ak = matrix1 + (size_t)k0 * bpc1 + byte0;
                    for (uint32_t k = 0; k < kn; k++, Ak += bpc1) {
                        const uint8_t bb = sB[k >> 1u];
                        const uint32_t s = ((k & 1u) ? (bb >> 4u) : (bb & 0xFu)) * 0x11111111u;

                        uint32_t a;
                        if (aligned4) {
                            a = *(const uint32_t *)Ak;
                        } else {
                            a = Ak[0];
                            if (nb > 1) { a |= (uint32_t)Ak[1] << 8u; }
                            if (nb > 2) { a |= (uint32_t)Ak[2] << 16u; }
                            if (nb > 3) { a |= (uint32_t)Ak[3] << 24u; }
                        }

                        acc ^= gf16v_mul_u32(a, s);
                    }
                }
            }

            if (nb) {
                // keep the padding nibble of the last byte of a column zero
                if ((n_rows1 & 1u) && (byte0 + nb == bpc1)) {
                    acc &= ~(0xF0u << (8u * (nb - 1u)));
                }

                uint8_t *Cj = result + (size_t)j * bpc1 + byte0;
                if (aligned4) {
                    *(uint32_t *)Cj = acc;
                } else {
                    for (uint32_t t = 0; t < nb; t++) {
                        Cj[t] = (uint8_t)(acc >> (8u * t));
                    }
                }
            }
        }
    }
}

/// result = matrix1 * matrix2, column major, nibble packed
/// \param result[out]: device pointer, n_rows1 x n_cols2. Must not alias the inputs.
/// \param matrix1[in]: device pointer, n_rows1 x n_cols1
/// \param matrix2[in]: device pointer, n_cols1 x n_cols2
/// \param n_rows1[in]: number of rows of matrix1
/// \param n_cols1[in]: number of columns of matrix1 = number of rows of matrix2
/// \param n_cols2[in]: number of columns of matrix2
/// \param stream[in]: CUDA stream
static inline
void gf16_matrix_mul(uint8_t *result,
                     const uint8_t *matrix1,
                     const uint8_t *matrix2,
                     const uint32_t n_rows1,
                     const uint32_t n_cols1,
                     const uint32_t n_cols2,
                     cudaStream_t stream = 0) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t bpc1 = gf16_matrix_bytes_per_column(n_rows1);
    const uint32_t nwords = (bpc1 + 3u) / 4u;
    const bool aligned4 = ((bpc1 & 3u) == 0) &&
                          ((((uintptr_t)matrix1) | ((uintptr_t)result)) & 3u) == 0;

    const uint32_t threads = 128;
    uint32_t gx = (nwords + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    gf16_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, aligned4);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf16_cuda
