#pragma once

/// CUDA implementation of GF(16^32) = GF(16)[X]/(X^32 + B*X^28 + X^21 + F*X^2 + 9),
/// with GF(16) = GF(2)[x]/(x^4 + x + 1).
///
/// Representation (identical to `arith.h`/`matrix.h`):
///  - a single element is 16 bytes, the coefficient of X^i is stored in byte
///    i/2, low nibble if i is even, high nibble otherwise. Here it is a struct
///    of four little endian 32-bit words, so word w holds the coefficients
///    8w..8w+7 as nibbles (coefficient i at bits 4*(i%8) of word i/8). The
///    memory layout is the same as the C `uint8_t[16]`.
///  - vectors are arrays of `gf16to32`, the base pointer must be 16-byte
///    aligned (as returned by cudaMalloc)
///  - matrices are column major: entry (i, j) at index j*nrows + i.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf16to32_cuda {

/// element of GF(16^32), see above for the layout
struct __align__(16) gf16to32 {
    uint32_t v[4];
};
static_assert(sizeof(gf16to32) == 16, "gf16to32 must have the same size as the C type");

/// number of elements of a column of B cached in shared memory per tile in
/// `gf16to32_matrix_mul`
constexpr uint32_t GF16TO32_CUDA_MUL_TILE_K = 256;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf16to32_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf16to32);
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf16to32_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return gf16to32_matrix_bytes_per_column(nrows) * ncols;
}

/// \return a+b
__host__ __device__ __forceinline__
gf16to32 gf16to32_add(const gf16to32 &a,
                      const gf16to32 &b) {
    gf16to32 r;
#pragma unroll
    for (uint32_t w = 0; w < 4; w++) {
        r.v[w] = a.v[w] ^ b.v[w];
    }
    return r;
}

/// \return a-b = a+b
__host__ __device__ __forceinline__
gf16to32 gf16to32_sub(const gf16to32 &a,
                      const gf16to32 &b) {
    return gf16to32_add(a, b);
}

/// multiplies the 8 GF(16) elements packed into the nibbles of a by x
/// \return [a_0*x, ..., a_7*x] packed
__host__ __device__ __forceinline__
uint32_t gf16to32_xtime_u32(const uint32_t a) {
    // x^4 = x + 1
    return ((a & 0x77777777u) << 1u) ^ (((a >> 3u) & 0x11111111u) * 3u);
}

/// Multiplication, constant time and table free. Horner scheme over the
/// coefficients of b (as the SIMD code in `arith.h`):
///     r = 0; for j = 31..0: r = r*X mod f; r += b_j * a
/// r*X is a 4-bit shift, the dropped coefficient c of X^32 is reduced by
/// adding c * (B*X^28 + X^21 + F*X^2 + 9). The products of a GF(16) scalar
/// with all 32 coefficients are computed bitsliced from the precomputed
/// a*x^k (resp. f*x^k), k = 0..3.
/// \return a*b
__host__ __device__ __forceinline__
gf16to32 gf16to32_mul(const gf16to32 &a,
                      const gf16to32 &b) {
    // A[k][w] = word w of a * x^k (coefficient-wise)
    uint32_t A[4][4];
#pragma unroll
    for (uint32_t w = 0; w < 4; w++) {
        A[0][w] = a.v[w];
        A[1][w] = gf16to32_xtime_u32(A[0][w]);
        A[2][w] = gf16to32_xtime_u32(A[1][w]);
        A[3][w] = gf16to32_xtime_u32(A[2][w]);
    }

    // M[k][.] = (X^32 mod f) * x^k, only the words 0, 2, 3 are non-zero:
    //  word 0: 9 + F*X^2, word 2: X^21 (nibble 5), word 3: B*X^28 (nibble 4)
    uint32_t M[4][3];
    M[0][0] = 0x00000F09u;
    M[0][1] = 0x00100000u;
    M[0][2] = 0x000B0000u;
#pragma unroll
    for (uint32_t k = 1; k < 4; k++) {
#pragma unroll
        for (uint32_t w = 0; w < 3; w++) {
            M[k][w] = gf16to32_xtime_u32(M[k - 1][w]);
        }
    }

    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#pragma unroll
    for (int32_t w = 3; w >= 0; w--) {
        const uint32_t bw = b.v[w];
#pragma unroll
        for (int32_t t = 7; t >= 0; t--) {
            // r = r * X mod f
            const uint32_t c = r3 >> 28u;
            r3 = (r3 << 4u) | (r2 >> 28u);
            r2 = (r2 << 4u) | (r1 >> 28u);
            r1 = (r1 << 4u) | (r0 >> 28u);
            r0 =  r0 << 4u;

            // r += c * (X^32 mod f) + b_j * a
            const uint32_t bj = (bw >> (4u * t)) & 0xFu;
#pragma unroll
            for (uint32_t k = 0; k < 4; k++) {
                const uint32_t mc = 0u - ((c  >> k) & 1u);
                const uint32_t mb = 0u - ((bj >> k) & 1u);
                r0 ^= (M[k][0] & mc) ^ (A[k][0] & mb);
                r1 ^=                  (A[k][1] & mb);
                r2 ^= (M[k][1] & mc) ^ (A[k][2] & mb);
                r3 ^= (M[k][2] & mc) ^ (A[k][3] & mb);
            }
        }
    }

    gf16to32 r = {{r0, r1, r2, r3}};
    return r;
}

/// element-wise operators for fq_cuda::elementwise
struct gf16to32_add_op {
    __device__ __forceinline__ gf16to32 operator()(const gf16to32 &a, const gf16to32 &b) const { return gf16to32_add(a, b); }
};
struct gf16to32_sub_op {
    __device__ __forceinline__ gf16to32 operator()(const gf16to32 &a, const gf16to32 &b) const { return gf16to32_sub(a, b); }
};
struct gf16to32_mul_op {
    __device__ __forceinline__ gf16to32 operator()(const gf16to32 &a, const gf16to32 &b) const { return gf16to32_mul(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to32_vector_add(gf16to32 *c,
                         const gf16to32 *a,
                         const gf16to32 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to32_add_op{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to32_vector_sub(gf16to32 *c,
                         const gf16to32 *a,
                         const gf16to32 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to32_sub_op{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n elements
/// \param a[in]: device pointer, n elements
/// \param b[in]: device pointer, n elements
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf16to32_vector_mul(gf16to32 *c,
                         const gf16to32 *a,
                         const gf16to32 *b,
                         const uint32_t n,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf16to32_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16to32_matrix_add(gf16to32 *matrix1,
                         const gf16to32 *matrix2,
                         const gf16to32 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf16to32_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf16to32_matrix_sub(gf16to32 *matrix1,
                         const gf16to32 *matrix2,
                         const gf16to32 *matrix3,
                         const uint32_t nrows,
                         const uint32_t ncols,
                         cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf16to32_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major.
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of it (consecutive threads -> consecutive rows,
/// so all accesses to matrix1/result are coalesced). The column j of matrix2
/// is staged through shared memory in tiles of GF16TO32_CUDA_MUL_TILE_K
/// elements; all threads of a block multiply with the same B[k, j], so the
/// Horner loop over its coefficients does not diverge.
static __global__
void gf16to32_matrix_mul_kernel(gf16to32 *__restrict__ result,
                                const gf16to32 *__restrict__ matrix1,
                                const gf16to32 *__restrict__ matrix2,
                                const uint32_t n_rows1,
                                const uint32_t n_cols1,
                                const uint32_t n_cols2) {
    __shared__ gf16to32 sB[GF16TO32_CUDA_MUL_TILE_K];

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const gf16to32 *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;

            gf16to32 acc = {{0, 0, 0, 0}};
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF16TO32_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF16TO32_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (i < n_rows1) {
                    const gf16to32 *Aik = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Aik += n_rows1) {
                        acc = gf16to32_add(acc, gf16to32_mul(*Aik, sB[k]));
                    }
                }
            }

            if (i < n_rows1) {
                result[(size_t)j * n_rows1 + i] = acc;
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
void gf16to32_matrix_mul(gf16to32 *result,
                         const gf16to32 *matrix1,
                         const gf16to32 *matrix2,
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

    gf16to32_matrix_mul_kernel<<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace gf16to32_cuda
