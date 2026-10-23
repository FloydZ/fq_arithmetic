#pragma once

/// CUDA implementation of GF(2^256) = GF(2)[x]/(x^256 + x^10 + x^5 + x^2 + 1).
///
/// Representation (identical to `arith.h`, where `gf2to256` = `uint64_t[4]`):
///  - bit j of limb v[i] is the coefficient of x^(64*i + j)
///  - `gf2to256_cuda::gf2to256` is a struct of four 64-bit limbs with exactly
///    the memory layout of the C type, so data can be copied between the C and
///    the CUDA implementation unchanged.
///  - vectors are arrays of elements; matrices are column major, entry (i, j)
///    at index j*nrows + i (same as `matrix.h`).
///
/// GPUs have no carry-less multiplication instruction. `gf2to256_clmul64` is
/// the constant-time "integer multiplication with holes" method (BearSSL,
/// ghash_ctmul64), the 256x256 product is a two level Karatsuba (9 carry-less
/// 64x64 multiplications) and the reduction folds the upper half twice.
///
/// All host wrappers take DEVICE pointers (16-byte aligned) and enqueue the
/// work on `stream`; they do not synchronize.
///
/// Everything lives in `namespace gf2to256_cuda`, so this file can be included
/// together with the C implementation in `arith.h` (e.g. as a test reference).

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf2to256_cuda {

/// GF(2^256) element: v[i] = coefficients of x^(64i)..x^(64i+63)
struct alignas(16) gf2to256 {
    uint64_t v[4];
};
static_assert(sizeof(gf2to256) == 32, "gf2to256 must have the layout of uint64_t[4]");

/// number of elements of a column of B cached in shared memory per tile in
/// `gf2to256_matrix_mul`
constexpr uint32_t GF2TO256_CUDA_MUL_TILE_K = 256;

/// threads per block in `gf2to256_matrix_mul`
constexpr uint32_t GF2TO256_CUDA_MUL_THREADS = 128;

/// \param x[in]: 64-bit word
/// \return x with the order of its bits reversed
__host__ __device__ __forceinline__
uint64_t gf2to256_rev64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __brevll(x);
#else
    x = ((x & 0x5555555555555555ull) << 1u)  | ((x >> 1u)  & 0x5555555555555555ull);
    x = ((x & 0x3333333333333333ull) << 2u)  | ((x >> 2u)  & 0x3333333333333333ull);
    x = ((x & 0x0F0F0F0F0F0F0F0Full) << 4u)  | ((x >> 4u)  & 0x0F0F0F0F0F0F0F0Full);
    x = ((x & 0x00FF00FF00FF00FFull) << 8u)  | ((x >> 8u)  & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16u) | ((x >> 16u) & 0x0000FFFF0000FFFFull);
    return (x << 32u) | (x >> 32u);
#endif
}

/// Lower 64 bits of the carry-less product x*y, constant time.
/// Each operand is split into 4 interleaved bit classes ("holes" of 3 bits),
/// so the integer products only accumulate <= 15 terms per 4-bit digit within
/// the low 64 bits and their parity is the carry-less result.
/// \param x[in]: first factor
/// \param y[in]: second factor
/// \return (x*y mod x^64) over GF(2)[x]
__host__ __device__ __forceinline__
uint64_t gf2to256_bmul64(const uint64_t x,
                         const uint64_t y) {
    const uint64_t m0 = 0x1111111111111111ull, m1 = 0x2222222222222222ull;
    const uint64_t m2 = 0x4444444444444444ull, m3 = 0x8888888888888888ull;
    const uint64_t x0 = x & m0, x1 = x & m1, x2 = x & m2, x3 = x & m3;
    const uint64_t y0 = y & m0, y1 = y & m1, y2 = y & m2, y3 = y & m3;
    const uint64_t z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
    const uint64_t z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
    const uint64_t z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
    const uint64_t z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
    return (z0 & m0) | (z1 & m1) | (z2 & m2) | (z3 & m3);
}

/// Full carry-less 64x64 -> 128 bit multiplication, constant time.
/// The upper half is computed from the bit-reversed operands.
/// \param lo[out]: coefficients 0..63 of a*b
/// \param hi[out]: coefficients 64..127 of a*b
/// \param a[in]: first factor
/// \param b[in]: second factor
__host__ __device__ __forceinline__
void gf2to256_clmul64(uint64_t &lo,
                      uint64_t &hi,
                      const uint64_t a,
                      const uint64_t b) {
    lo = gf2to256_bmul64(a, b);
    hi = gf2to256_rev64(gf2to256_bmul64(gf2to256_rev64(a), gf2to256_rev64(b))) >> 1u;
}

/// carry-less 128x128 -> 256 bit multiplication (Karatsuba, 3 carry-less
/// 64x64 multiplications)
/// \param p[out]: 4 limbs, p[0] = coefficients 0..63
/// \param a0, a1[in]: first factor, low and high limb
/// \param b0, b1[in]: second factor, low and high limb
__host__ __device__ __forceinline__
void gf2to256_clmul128(uint64_t p[4],
                       const uint64_t a0,
                       const uint64_t a1,
                       const uint64_t b0,
                       const uint64_t b1) {
    uint64_t l0, l1, h0, h1, m0, m1;
    gf2to256_clmul64(l0, l1, a0, b0);
    gf2to256_clmul64(h0, h1, a1, b1);
    gf2to256_clmul64(m0, m1, a0 ^ a1, b0 ^ b1);
    m0 ^= l0 ^ h0;
    m1 ^= l1 ^ h1;
    p[0] = l0;
    p[1] = l1 ^ m0;
    p[2] = h0 ^ m1;
    p[3] = h1;
}

/// unreduced 512-bit product of two elements (Karatsuba on 128-bit halves)
/// \param p[out]: 8 limbs, p[0] = coefficients 0..63
/// \param a[in]: first factor
/// \param b[in]: second factor
__host__ __device__ __forceinline__
void gf2to256_mul_wide(uint64_t p[8],
                       const gf2to256 &a,
                       const gf2to256 &b) {
    uint64_t l[4], h[4], m[4];
    gf2to256_clmul128(l, a.v[0], a.v[1], b.v[0], b.v[1]);
    gf2to256_clmul128(h, a.v[2], a.v[3], b.v[2], b.v[3]);
    gf2to256_clmul128(m, a.v[0] ^ a.v[2], a.v[1] ^ a.v[3], b.v[0] ^ b.v[2], b.v[1] ^ b.v[3]);
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        m[i] ^= l[i] ^ h[i];
        p[i] = l[i];
        p[i + 4] = h[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        p[i + 2] ^= m[i];
    }
}

/// reduction modulo x^256 + x^10 + x^5 + x^2 + 1, i.e. x^256 = x^10 + x^5 + x^2 + 1
/// \param p[in]: unreduced 512-bit polynomial
/// \return p mod (x^256 + x^10 + x^5 + x^2 + 1)
__host__ __device__ __forceinline__
gf2to256 gf2to256_reduce(const uint64_t p[8]) {
    const uint64_t *h = p + 4;
    gf2to256 r;
    // p + h*(x^10 + x^5 + x^2 + 1), truncated to 256 bits
    r.v[0] = p[0] ^ h[0] ^ (h[0] << 2u) ^ (h[0] << 5u) ^ (h[0] << 10u);
#pragma unroll
    for (uint32_t i = 1; i < 4; i++) {
        r.v[i] = p[i] ^ h[i] ^ ((h[i] << 2u)  | (h[i - 1] >> 62u))
                             ^ ((h[i] << 5u)  | (h[i - 1] >> 59u))
                             ^ ((h[i] << 10u) | (h[i - 1] >> 54u));
    }
    // the bits shifted beyond x^255 (degree <= 9) are folded once more
    const uint64_t o = (h[3] >> 62u) ^ (h[3] >> 59u) ^ (h[3] >> 54u);
    r.v[0] ^= o ^ (o << 2u) ^ (o << 5u) ^ (o << 10u);
    return r;
}

/// Addition in GF(2^256)
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
gf2to256 gf2to256_add(const gf2to256 &a,
                      const gf2to256 &b) {
    gf2to256 r;
    r.v[0] = a.v[0] ^ b.v[0];
    r.v[1] = a.v[1] ^ b.v[1];
    r.v[2] = a.v[2] ^ b.v[2];
    r.v[3] = a.v[3] ^ b.v[3];
    return r;
}

/// Subtraction in GF(2^256), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
gf2to256 gf2to256_sub(const gf2to256 &a,
                      const gf2to256 &b) {
    return gf2to256_add(a, b);
}

/// Multiplication in GF(2^256), constant time
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
gf2to256 gf2to256_mul(const gf2to256 &a,
                      const gf2to256 &b) {
    uint64_t p[8];
    gf2to256_mul_wide(p, a, b);
    return gf2to256_reduce(p);
}

/// element-wise operators for fq_cuda::elementwise
struct gf2to256_add_op {
    __device__ __forceinline__ gf2to256 operator()(const gf2to256 &a, const gf2to256 &b) const { return gf2to256_add(a, b); }
};
struct gf2to256_sub_op {
    __device__ __forceinline__ gf2to256 operator()(const gf2to256 &a, const gf2to256 &b) const { return gf2to256_sub(a, b); }
};
struct gf2to256_mul_op {
    __device__ __forceinline__ gf2to256 operator()(const gf2to256 &a, const gf2to256 &b) const { return gf2to256_mul(a, b); }
};

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf2to256_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf2to256);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf2to256_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return gf2to256_matrix_bytes_per_column(nrows) * ncols;
}

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to256_vector_add(gf2to256 *c,
                         const gf2to256 *a,
                         const gf2to256 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to256_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to256_vector_sub(gf2to256 *c,
                         const gf2to256 *a,
                         const gf2to256 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to256_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf2to256_vector_mul(gf2to256 *c,
                         const gf2to256 *a,
                         const gf2to256 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf2to256_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to256_matrix_add(gf2to256 *matrix1,
                         const gf2to256 *matrix2,
                         const gf2to256 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to256_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf2to256_matrix_sub(gf2to256 *matrix1,
                         const gf2to256 *matrix2,
                         const gf2to256 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf2to256_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of that column (consecutive threads read
/// consecutive rows of matrix1). The column j of matrix2 is staged through
/// shared memory in tiles of GF2TO256_CUDA_MUL_TILE_K elements. The products
/// are accumulated unreduced (the reduction is GF(2)-linear) and reduced once.
static __global__
void gf2to256_matrix_mul_kernel(gf2to256 *__restrict__ result,
                                const gf2to256 *__restrict__ matrix1,
                                const gf2to256 *__restrict__ matrix2,
                                const uint32_t n_rows1,
                                const uint32_t n_cols1,
                                const uint32_t n_cols2) {
    __shared__ gf2to256 sB[GF2TO256_CUDA_MUL_TILE_K];

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const gf2to256 *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;

            uint64_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF2TO256_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF2TO256_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (i < n_rows1) {
                    const gf2to256 *Ak = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Ak += n_rows1) {
                        uint64_t p[8];
                        gf2to256_mul_wide(p, *Ak, sB[k]);
#pragma unroll
                        for (uint32_t t = 0; t < 8; t++) {
                            acc[t] ^= p[t];
                        }
                    }
                }
            }

            if (i < n_rows1) {
                result[(size_t)j * n_rows1 + i] = gf2to256_reduce(acc);
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
void gf2to256_matrix_mul(gf2to256 *result,
                         const gf2to256 *matrix1,
                         const gf2to256 *matrix2,
                         const uint32_t n_rows1,
                         const uint32_t n_cols1,
                         const uint32_t n_cols2,
                         cudaStream_t stream = 0) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t threads = GF2TO256_CUDA_MUL_THREADS;
    uint32_t gx = (n_rows1 + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    gf2to256_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf2to256_cuda
