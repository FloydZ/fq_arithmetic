#pragma once

/// CUDA implementation of GF(4) = GF(2)[X]/(X^2 + X + 1) arithmetic.
///
/// Representation (identical to `arith.h`/`vector.h`/`matrix.h`):
///  - an element a0 + a1*X is encoded as the 2-bit value a0 | (a1 << 1),
///    stored in the low two bits of an `uint8_t`
///  - vectors/matrices are packed: element i is stored in the bits
///    [2*(i%4), 2*(i%4)+1] of byte i/4.
///  - matrices are column major: column j starts at byte
///    `gf4_matrix_bytes_per_column(nrows) * j`, with
///    `gf4_matrix_bytes_per_column(nrows) = ceil(nrows / 4)`. The unused bits
///    of the last byte of each column are padding and are expected to be (and
///    kept) zero.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Everything lives in `namespace gf4_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).
/// NOTE: do not include the C `matrix.h` together with this file: it defines
/// `gf4_matrix_bytes_per_column` as a function-like macro.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf4_cuda {

/// number of elements of a column of B cached in shared memory per tile in
/// `gf4_matrix_mul` (must be a multiple of 4)
constexpr uint32_t GF4_CUDA_MUL_TILE_K = 4096;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single packed column
__host__ __device__ __forceinline__
uint32_t gf4_matrix_bytes_per_column(const uint32_t nrows) {
    return (nrows + 3u) >> 2u;
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a packed column major matrix
__host__ __device__ __forceinline__
size_t gf4_matrix_bytes_size(const uint32_t nrows,
                             const uint32_t ncols) {
    return (size_t)gf4_matrix_bytes_per_column(nrows) * ncols;
}

/// \param n[in]: number of elements
/// \return mask of the valid bits of the last byte of a packed vector of n elements
__host__ __device__ __forceinline__
uint8_t gf4_vector_tail_mask(const uint32_t n) {
    return (n & 3u) ? (uint8_t)((1u << (2u * (n & 3u))) - 1u) : (uint8_t)0xFFu;
}

/// Addition in GF(4). Also works on packed bytes/words.
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
uint8_t gf4_add(const uint8_t a,
                const uint8_t b) {
    return a ^ b;
}

/// Subtraction in GF(4), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
uint8_t gf4_sub(const uint8_t a,
                const uint8_t b) {
    return a ^ b;
}

/// Lane-wise multiplication of 16 GF(4) elements packed into 32 bits
/// (2 bits per element). Constant time, no lookup tables.
/// (a1 X + a0)(b1 X + b0) = (a1b0 + a0b1 + a1b1) X + (a0b0 + a1b1)
/// \param a[in]: 16 packed elements
/// \param b[in]: 16 packed elements
/// \return [a_0*b_0, ..., a_15*b_15] packed
__host__ __device__ __forceinline__
uint32_t gf4v_mul_u32(const uint32_t a,
                      const uint32_t b) {
    const uint32_t and_ = a & b;                        // bit0: a0b0, bit1: a1b1
    const uint32_t l = ((a << 1u) & b) ^ (a & (b << 1u)); // bit1: a0b1 + a1b0
    return ((and_ ^ (and_ >> 1u)) & 0x55555555u) ^ ((l ^ and_) & 0xAAAAAAAAu);
}

/// Multiplication in GF(4).
/// \param a[in]: first factor (low two bits)
/// \param b[in]: second factor (low two bits)
/// \return a*b
__host__ __device__ __forceinline__
uint8_t gf4_mul(const uint8_t a,
                const uint8_t b) {
    return (uint8_t)(gf4v_mul_u32(a & 3u, b & 3u) & 3u);
}

/// Lane-wise multiplication of two bytes, each holding 4 packed elements.
/// \param a[in]: 4 packed elements
/// \param b[in]: 4 packed elements
/// \return [a_0*b_0, ..., a_3*b_3] packed
__host__ __device__ __forceinline__
uint8_t gf4v_mul_u8(const uint8_t a,
                    const uint8_t b) {
    return (uint8_t)gf4v_mul_u32(a, b);
}

/// element-wise operators used by the generic kernels below. They operate on
/// 32-bit words of packed elements; for single bytes the upper 24 bits are 0.
struct gf4_add_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a ^ b; }
};
struct gf4_sub_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return a ^ b; }
};
struct gf4_mul_op {
    __device__ __forceinline__ uint32_t operator()(const uint32_t a, const uint32_t b) const { return gf4v_mul_u32(a, b); }
};

/// c[i] = op(a[i], b[i]) on 16-byte chunks, grid-stride loop
template<typename Op>
__global__ void gf4_elementwise_u128_kernel(uint4 *c,
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

/// c[i] = op(a[i], b[i]) on single bytes, grid-stride loop. The last byte
/// (i == n-1) is masked with `last_mask`.
template<typename Op>
__global__ void gf4_elementwise_u8_kernel(uint8_t *c,
                                          const uint8_t *a,
                                          const uint8_t *b,
                                          const size_t n,
                                          const uint8_t last_mask,
                                          const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint8_t r = (uint8_t)op(a[i], b[i]);
        if (i == n - 1) {
            r &= last_mask;
        }
        c[i] = r;
    }
}

/// c = op(a, b) on `nbytes` bytes of packed elements; the last byte is masked
/// with `last_mask`. Uses 16-byte accesses if all pointers are 16-byte
/// aligned, single bytes for the rest (incl. a partially valid last byte).
/// c may alias a and/or b.
template<typename Op>
static inline
void gf4_elementwise(uint8_t *c,
                     const uint8_t *a,
                     const uint8_t *b,
                     const size_t nbytes,
                     const uint8_t last_mask,
                     const Op op,
                     cudaStream_t stream) {
    if (nbytes == 0) {
        return;
    }

    size_t done = 0;
    const uintptr_t align = (uintptr_t)a | (uintptr_t)b | (uintptr_t)c;
    if ((align & 15u) == 0) {
        // a masked last byte must be handled by the byte kernel
        const size_t full = (last_mask == 0xFFu) ? nbytes : nbytes - 1;
        const size_t n16 = full / 16;
        if (n16) {
            gf4_elementwise_u128_kernel<Op><<<fq_cuda::grid_size(n16), fq_cuda::BLOCK_SIZE, 0, stream>>>
                ((uint4 *)c, (const uint4 *)a, (const uint4 *)b, n16, op);
            FQ_CUDA_CHECK(cudaGetLastError());
        }
        done = n16 * 16;
    }

    if (done < nbytes) {
        gf4_elementwise_u8_kernel<Op><<<fq_cuda::grid_size(nbytes - done), fq_cuda::BLOCK_SIZE, 0, stream>>>
            (c + done, a + done, b + done, nbytes - done, last_mask, op);
        FQ_CUDA_CHECK(cudaGetLastError());
    }
}

/// c = a + b, element-wise on packed vectors. The unused bits of the last
/// byte of c are set to zero.
/// \param c[out]: device pointer, ceil(n/4) bytes
/// \param a[in]: device pointer, ceil(n/4) bytes
/// \param b[in]: device pointer, ceil(n/4) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf4_vector_add(uint8_t *c,
                    const uint8_t *a,
                    const uint8_t *b,
                    const uint32_t n,
                    cudaStream_t stream = 0) {
    gf4_elementwise(c, a, b, gf4_matrix_bytes_per_column(n), gf4_vector_tail_mask(n), gf4_add_op{}, stream);
}

/// c = a - b, element-wise on packed vectors. The unused bits of the last
/// byte of c are set to zero.
/// \param c[out]: device pointer, ceil(n/4) bytes
/// \param a[in]: device pointer, ceil(n/4) bytes
/// \param b[in]: device pointer, ceil(n/4) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf4_vector_sub(uint8_t *c,
                    const uint8_t *a,
                    const uint8_t *b,
                    const uint32_t n,
                    cudaStream_t stream = 0) {
    gf4_elementwise(c, a, b, gf4_matrix_bytes_per_column(n), gf4_vector_tail_mask(n), gf4_sub_op{}, stream);
}

/// c = a * b, element-wise on packed vectors. The unused bits of the last
/// byte of c are set to zero.
/// \param c[out]: device pointer, ceil(n/4) bytes
/// \param a[in]: device pointer, ceil(n/4) bytes
/// \param b[in]: device pointer, ceil(n/4) bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf4_vector_mul(uint8_t *c,
                    const uint8_t *a,
                    const uint8_t *b,
                    const uint32_t n,
                    cudaStream_t stream = 0) {
    gf4_elementwise(c, a, b, gf4_matrix_bytes_per_column(n), gf4_vector_tail_mask(n), gf4_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major, packed. The padding bits of the
/// inputs are expected to be zero (then they stay zero in matrix1).
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf4_matrix_add(uint8_t *matrix1,
                    const uint8_t *matrix2,
                    const uint8_t *matrix3,
                    const uint32_t nrows,
                    const uint32_t ncols,
                    cudaStream_t stream = 0) {
    gf4_elementwise(matrix1, matrix2, matrix3, gf4_matrix_bytes_size(nrows, ncols), 0xFFu, gf4_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major, packed. The padding bits of the
/// inputs are expected to be zero (then they stay zero in matrix1).
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf4_matrix_sub(uint8_t *matrix1,
                    const uint8_t *matrix2,
                    const uint8_t *matrix3,
                    const uint32_t nrows,
                    const uint32_t ncols,
                    cudaStream_t stream = 0) {
    gf4_elementwise(matrix1, matrix2, matrix3, gf4_matrix_bytes_size(nrows, ncols), 0xFFu, gf4_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, packed.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes 4 consecutive bytes (16 rows) of that column. The column
/// j of matrix2 is staged through shared memory in tiles of
/// GF4_CUDA_MUL_TILE_K elements.
/// \param aligned4[in]: matrix1 and result are 4-byte aligned and the
///                      column size is a multiple of 4 bytes, so 32-bit
///                      loads/stores can be used
static __global__
void gf4_matrix_mul_kernel(uint8_t *__restrict__ result,
                           const uint8_t *__restrict__ matrix1,
                           const uint8_t *__restrict__ matrix2,
                           const uint32_t n_rows1,
                           const uint32_t n_cols1,
                           const uint32_t n_cols2,
                           const bool aligned4) {
    __shared__ uint8_t sB[GF4_CUDA_MUL_TILE_K / 4];

    const uint32_t bpc1 = gf4_matrix_bytes_per_column(n_rows1); // also of result
    const uint32_t bpc2 = gf4_matrix_bytes_per_column(n_cols1);
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
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF4_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF4_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < (kn + 3u) / 4u; t += blockDim.x) {
                    sB[t] = Bj[k0 / 4u + t];
                }
                __syncthreads();

                if (nb) {
                    const uint8_t *Ak = matrix1 + (size_t)k0 * bpc1 + byte0;
                    for (uint32_t k = 0; k < kn; k++, Ak += bpc1) {
                        const uint32_t e = (sB[k >> 2u] >> (2u * (k & 3u))) & 3u;
                        const uint32_t s = e * 0x55555555u;

                        uint32_t a;
                        if (aligned4) {
                            a = *(const uint32_t *)Ak;
                        } else {
                            a = Ak[0];
                            if (nb > 1) { a |= (uint32_t)Ak[1] << 8u; }
                            if (nb > 2) { a |= (uint32_t)Ak[2] << 16u; }
                            if (nb > 3) { a |= (uint32_t)Ak[3] << 24u; }
                        }

                        acc ^= gf4v_mul_u32(a, s);
                    }
                }
            }

            if (nb) {
                // keep the padding bits of the last byte of a column zero
                const uint32_t rem = n_rows1 & 3u;
                if (rem && (byte0 + nb == bpc1)) {
                    acc &= ~(((0xFFu << (2u * rem)) & 0xFFu) << (8u * (nb - 1u)));
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

/// result = matrix1 * matrix2, column major, packed
/// \param result[out]: device pointer, n_rows1 x n_cols2. Must not alias the inputs.
/// \param matrix1[in]: device pointer, n_rows1 x n_cols1
/// \param matrix2[in]: device pointer, n_cols1 x n_cols2
/// \param n_rows1[in]: number of rows of matrix1
/// \param n_cols1[in]: number of columns of matrix1 = number of rows of matrix2
/// \param n_cols2[in]: number of columns of matrix2
/// \param stream[in]: CUDA stream
static inline
void gf4_matrix_mul(uint8_t *result,
                    const uint8_t *matrix1,
                    const uint8_t *matrix2,
                    const uint32_t n_rows1,
                    const uint32_t n_cols1,
                    const uint32_t n_cols2,
                    cudaStream_t stream = 0) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t bpc1 = gf4_matrix_bytes_per_column(n_rows1);
    const uint32_t nwords = (bpc1 + 3u) / 4u;
    const bool aligned4 = ((bpc1 & 3u) == 0) &&
                          ((((uintptr_t)matrix1) | ((uintptr_t)result)) & 3u) == 0;

    const uint32_t threads = 128;
    uint32_t gx = (nwords + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    gf4_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, aligned4);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf4_cuda
