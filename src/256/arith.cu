#pragma once

/// CUDA implementation of GF(256) = GF(2)[x]/(x^8 + x^4 + x^3 + x + 1)
/// (AES field, identical to `arith.h`).
///
/// Representation (identical to `arith.h`/`matrix.h`):
///  - one element per byte (`uint8_t`)
///  - matrices are column major: entry (i, j) is at index j*nrows + i, no padding.
///
/// All host wrappers take DEVICE pointers and enqueue the work on `stream`;
/// they do not synchronize.
///
/// Besides the GF(256) API, this file provides the generic word-packed
/// building blocks (`packed_elementwise`, `matrix_mul_words`) that are reused
/// by the tower fields GF(256^2) (src/256to2/arith.cu) and GF(256^4)
/// (src/256to4/arith.cu): they work on any element type T with
/// sizeof(T) <= 4 whose multiplication can be done lane-wise on 32-bit words.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf256_cuda {

/// number of elements of a column of B cached in shared memory per tile in
/// `matrix_mul_words`
constexpr uint32_t GF256_CUDA_MUL_TILE_K = 1024;

/// \param nrows[in]: number of rows
/// \return number of bytes of a single column
__host__ __device__ __forceinline__
uint32_t gf256_matrix_bytes_per_column(const uint32_t nrows) {
    return nrows;
}

/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \return number of bytes of a column major matrix
__host__ __device__ __forceinline__
size_t gf256_matrix_bytes_size(const uint32_t nrows,
                               const uint32_t ncols) {
    return (size_t)nrows * ncols;
}

/// Addition in GF(256). Also works on packed bytes/words.
/// \param a[in]: first addend
/// \param b[in]: second addend
/// \return a+b
__host__ __device__ __forceinline__
uint8_t gf256_add(const uint8_t a,
                  const uint8_t b) {
    return a ^ b;
}

/// Subtraction in GF(256), identical to the addition (characteristic 2).
/// \param a[in]: minuend
/// \param b[in]: subtrahend
/// \return a-b
__host__ __device__ __forceinline__
uint8_t gf256_sub(const uint8_t a,
                  const uint8_t b) {
    return a ^ b;
}

/// Lane-wise multiplication by x of 4 GF(256) elements packed into 32 bits.
/// \param a[in]: 4 packed elements
/// \return [a_0*x, ..., a_3*x] packed
__host__ __device__ __forceinline__
uint32_t gf256v_mulx_u32(const uint32_t a) {
    const uint32_t msb = a & 0x80808080u;
    return ((a ^ msb) << 1u) ^ ((msb >> 7u) * 0x1Bu);
}

/// Lane-wise multiplication of 4 GF(256) elements packed into 32 bits
/// (one element per byte). Constant time, no lookup tables.
/// \param a[in]: 4 packed elements
/// \param b[in]: 4 packed elements
/// \return [a_0*b_0, ..., a_3*b_3] packed
__host__ __device__ __forceinline__
uint32_t gf256v_mul_u32(uint32_t a,
                        const uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        // 0xFF in every byte whose i-th bit of b is set
        const uint32_t m = ((b >> i) & 0x01010101u) * 0xFFu;
        r ^= a & m;
        a = gf256v_mulx_u32(a);
    }
    return r;
}

/// Multiplication in GF(256).
/// \param a[in]: first factor
/// \param b[in]: second factor
/// \return a*b
__host__ __device__ __forceinline__
uint8_t gf256_mul(const uint8_t a,
                  const uint8_t b) {
    return (uint8_t)gf256v_mul_u32(a, b);
}

/// element-wise XOR (= add = sub in characteristic 2) on elements of type T,
/// and on 32-bit words of packed elements
template<typename T>
struct xor_op {
    __device__ __forceinline__ uint32_t word(const uint32_t a, const uint32_t b) const { return a ^ b; }
    __device__ __forceinline__ T operator()(const T a, const T b) const { return (T)(a ^ b); }
};

/// element-wise GF(256) multiplication
struct gf256_mul_op {
    __device__ __forceinline__ uint32_t word(const uint32_t a, const uint32_t b) const { return gf256v_mul_u32(a, b); }
    __device__ __forceinline__ uint8_t operator()(const uint8_t a, const uint8_t b) const { return gf256_mul(a, b); }
};

/// c[i] = op.word(a[i], b[i]) on 16-byte chunks (4 words each), grid-stride loop
template<typename Op>
__global__ void packed_u128_kernel(uint4 *c,
                                   const uint4 *a,
                                   const uint4 *b,
                                   const size_t n,
                                   const Op op) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 x = a[i], y = b[i];
        uint4 z;
        z.x = op.word(x.x, y.x);
        z.y = op.word(x.y, y.y);
        z.z = op.word(x.z, y.z);
        z.w = op.word(x.w, y.w);
        c[i] = z;
    }
}

/// c = op(a, b) on n elements of type T (sizeof(T) in {1, 2, 4}).
/// Uses 16-byte accesses with `op.word` if all pointers are 16-byte aligned,
/// and `op(T, T)` for the remaining elements. c may alias a and/or b.
/// `Op` must provide `word(uint32_t, uint32_t)` operating lane-wise on
/// 4/sizeof(T) packed elements, and `T operator()(T, T)`.
template<typename T, typename Op>
static inline
void packed_elementwise(T *c,
                        const T *a,
                        const T *b,
                        const size_t n,
                        const Op op,
                        cudaStream_t stream) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "unsupported element size");
    if (n == 0) {
        return;
    }

    size_t done = 0;
    const uintptr_t align = (uintptr_t)a | (uintptr_t)b | (uintptr_t)c;
    if ((align & 15u) == 0) {
        const size_t n16 = (n * sizeof(T)) / 16;
        if (n16) {
            packed_u128_kernel<Op><<<fq_cuda::grid_size(n16), fq_cuda::BLOCK_SIZE, 0, stream>>>
                ((uint4 *)c, (const uint4 *)a, (const uint4 *)b, n16, op);
            FQ_CUDA_CHECK(cudaGetLastError());
        }
        done = n16 * (16 / sizeof(T));
    }

    if (done < n) {
        fq_cuda::elementwise<T, Op>(c + done, a + done, b + done, n - done, op, stream);
    }
}

/// result = matrix1 * matrix2, column major, element type T (sizeof(T) <= 4).
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes 4 consecutive bytes (4/sizeof(T) rows) of that column as
/// one 32-bit word. Column j of matrix2 is staged through shared memory in
/// tiles of GF256_CUDA_MUL_TILE_K elements.
/// `Field` provides `static uint32_t broadcast(T s)` (s in every lane of a
/// word) and `static uint32_t mul_word(uint32_t a, uint32_t b)` (lane-wise
/// product; zero lanes must stay zero).
/// \param aligned4[in]: matrix1/result are 4-byte aligned and the column size
///                      is a multiple of 4 bytes, so 32-bit accesses can be used
template<typename T, typename Field>
__global__ void matrix_mul_words_kernel(T *__restrict__ result,
                                        const T *__restrict__ matrix1,
                                        const T *__restrict__ matrix2,
                                        const uint32_t n_rows1,
                                        const uint32_t n_cols1,
                                        const uint32_t n_cols2,
                                        const bool aligned4) {
    __shared__ T sB[GF256_CUDA_MUL_TILE_K];

    const uint8_t *A8 = (const uint8_t *)matrix1;
    uint8_t *R8 = (uint8_t *)result;
    const uint32_t bpc = n_rows1 * (uint32_t)sizeof(T); // bytes per column of matrix1 and result
    const uint32_t nwords = (bpc + 3u) / 4u;

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const T *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t w0 = blockIdx.x * blockDim.x; w0 < nwords; w0 += gridDim.x * blockDim.x) {
            const uint32_t w = w0 + threadIdx.x;
            const uint32_t byte0 = 4u * w;
            const uint32_t nb = (w < nwords) ? min(4u, bpc - byte0) : 0u;

            uint32_t acc = 0;
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += GF256_CUDA_MUL_TILE_K) {
                const uint32_t kn = min(GF256_CUDA_MUL_TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (nb) {
                    const uint8_t *Ak = A8 + (size_t)k0 * bpc + byte0;
                    for (uint32_t k = 0; k < kn; k++, Ak += bpc) {
                        const uint32_t s = Field::broadcast(sB[k]);

                        uint32_t a;
                        if (aligned4) {
                            a = *(const uint32_t *)Ak;
                        } else {
                            a = Ak[0];
                            if (nb > 1) { a |= (uint32_t)Ak[1] << 8u; }
                            if (nb > 2) { a |= (uint32_t)Ak[2] << 16u; }
                            if (nb > 3) { a |= (uint32_t)Ak[3] << 24u; }
                        }

                        acc ^= Field::mul_word(a, s);
                    }
                }
            }

            if (nb) {
                uint8_t *Cj = R8 + (size_t)j * bpc + byte0;
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

/// result = matrix1 * matrix2, column major, element type T (sizeof(T) <= 4)
/// \param result[out]: device pointer, n_rows1 x n_cols2. Must not alias the inputs.
/// \param matrix1[in]: device pointer, n_rows1 x n_cols1
/// \param matrix2[in]: device pointer, n_cols1 x n_cols2
/// \param n_rows1[in]: number of rows of matrix1 (n_rows1*sizeof(T) < 2^32)
/// \param n_cols1[in]: number of columns of matrix1 = number of rows of matrix2
/// \param n_cols2[in]: number of columns of matrix2
/// \param stream[in]: CUDA stream
template<typename T, typename Field>
static inline
void matrix_mul_words(T *result,
                      const T *matrix1,
                      const T *matrix2,
                      const uint32_t n_rows1,
                      const uint32_t n_cols1,
                      const uint32_t n_cols2,
                      cudaStream_t stream) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "unsupported element size");
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    const uint32_t bpc = n_rows1 * (uint32_t)sizeof(T);
    const uint32_t nwords = (bpc + 3u) / 4u;
    const bool aligned4 = ((bpc & 3u) == 0) &&
                          ((((uintptr_t)matrix1) | ((uintptr_t)result)) & 3u) == 0;

    const uint32_t threads = 128;
    uint32_t gx = (nwords + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    matrix_mul_words_kernel<T, Field><<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, aligned4);
    FQ_CUDA_CHECK(cudaGetLastError());
}

/// GF(256) lane description for `matrix_mul_words`: 4 elements per word
struct gf256_field {
    static __device__ __forceinline__ uint32_t broadcast(const uint8_t s) { return s * 0x01010101u; }
    static __device__ __forceinline__ uint32_t mul_word(const uint32_t a, const uint32_t b) { return gf256v_mul_u32(a, b); }
};

/// c = a + b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256_vector_add(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    packed_elementwise(c, a, b, n, xor_op<uint8_t>{}, stream);
}

/// c = a - b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256_vector_sub(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    packed_elementwise(c, a, b, n, xor_op<uint8_t>{}, stream);
}

/// c = a * b, element-wise
/// \param c[out]: device pointer, n bytes
/// \param a[in]: device pointer, n bytes
/// \param b[in]: device pointer, n bytes
/// \param n[in]: number of elements
/// \param stream[in]: CUDA stream
static inline
void gf256_vector_mul(uint8_t *c,
                      const uint8_t *a,
                      const uint8_t *b,
                      const uint32_t n,
                      cudaStream_t stream = 0) {
    packed_elementwise(c, a, b, n, gf256_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256_matrix_add(uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint8_t *matrix3,
                      const uint32_t nrows,
                      const uint32_t ncols,
                      cudaStream_t stream = 0) {
    packed_elementwise(matrix1, matrix2, matrix3, gf256_matrix_bytes_size(nrows, ncols), xor_op<uint8_t>{}, stream);
}

/// matrix1 = matrix2 - matrix3, column major
/// \param matrix1[out]: device pointer, nrows x ncols
/// \param matrix2[in]: device pointer, nrows x ncols
/// \param matrix3[in]: device pointer, nrows x ncols
/// \param nrows[in]: number of rows
/// \param ncols[in]: number of columns
/// \param stream[in]: CUDA stream
static inline
void gf256_matrix_sub(uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint8_t *matrix3,
                      const uint32_t nrows,
                      const uint32_t ncols,
                      cudaStream_t stream = 0) {
    packed_elementwise(matrix1, matrix2, matrix3, gf256_matrix_bytes_size(nrows, ncols), xor_op<uint8_t>{}, stream);
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
void gf256_matrix_mul(uint8_t *result,
                      const uint8_t *matrix1,
                      const uint8_t *matrix2,
                      const uint32_t n_rows1,
                      const uint32_t n_cols1,
                      const uint32_t n_cols2,
                      cudaStream_t stream = 0) {
    matrix_mul_words<uint8_t, gf256_field>(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256_cuda
