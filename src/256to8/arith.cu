#pragma once

/// CUDA implementation of GF(256^8), using exactly the tower of `arith.h`:
///     GF(256)   = GF(2)[x]/(x^8 + x^4 + x^3 + x + 1)   uint8_t  (AES field)
///     GF(256^2) = GF(256)  [Y]/(Y^2 + Y + 0x20)        uint16_t
///     GF(256^4) = GF(256^2)[Z]/(Z^2 + Z + 0x2000)      uint32_t
///     GF(256^8) = GF(256^4)[W]/(W^2 + W + 0x20000000)  uint64_t
/// Every level is built the same way: an element is v = v_1*X + v_0, the upper
/// half of the bits holds v_1, and X^2 = X + gamma with
///     gamma_0 = 0x20, gamma_L = gamma_{L-1} * X_L (upper half gamma_{L-1}, lower half 0).
/// So multiplying by gamma_L only needs multiplications by gamma_{L-1}:
///     gamma_L * v = gamma_{L-1}*(v_0 + v_1) * X + gamma_{L-1}^2 * v_1
/// and a product is computed with Karatsuba:
///     (a*b)_0 = a_0*b_0 + gamma*a_1*b_1,  (a*b)_1 = (a_0+a_1)*(b_0+b_1) + a_0*b_0
/// All functions are constant time and table free.
///
/// Elements are stored as in the C code (uint64_t, little endian limbs), so
/// data can be copied between the C and the CUDA implementation unchanged.
/// Vectors/matrices: plain arrays of elements, matrices column-major
/// (entry (i, j) at j*nrows + i).
///
/// This file also provides the generic tiled matrix multiplication
/// (`matrix_mul_tiled`) used by the 256to16, 256to24 and 256toX ports.
///
/// All host wrappers take DEVICE pointers, enqueue on `stream` and do not
/// synchronize.

#include <cstdint>
#include <cuda_runtime.h>

#include "../helper.cuh"

namespace gf256to8_cuda {

typedef uint8_t  gf256;
typedef uint16_t gf256to2;
typedef uint32_t gf256to4;
typedef uint64_t gf256to8;

// ---------------------------------------------------------------- GF(256)

/// lane-wise multiplication by x of 4 GF(256) elements packed into 32 bits
__host__ __device__ __forceinline__
uint32_t gf256v_xtime_u32(const uint32_t a) {
    return ((a & 0x7F7F7F7Fu) << 1u) ^ (((a >> 7u) & 0x01010101u) * 0x1Bu);
}

/// lane-wise multiplication by x of 8 GF(256) elements packed into 64 bits
__host__ __device__ __forceinline__
uint64_t gf256v_xtime_u64(const uint64_t a) {
    return ((a & 0x7F7F7F7F7F7F7F7Full) << 1u) ^ (((a >> 7u) & 0x0101010101010101ull) * 0x1Bull);
}

/// lane-wise multiplication of 4 GF(256) elements packed into 32 bits
/// \return [a_0*b_0, ..., a_3*b_3]
__host__ __device__ __forceinline__
uint32_t gf256v_mul_u32(uint32_t a,
                        const uint32_t b) {
    uint32_t r = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        // 0xFF in every byte whose i-th bit of b is set
        r ^= a & (((b >> i) & 0x01010101u) * 0xFFu);
        a = gf256v_xtime_u32(a);
    }
    return r;
}

/// lane-wise multiplication of 8 GF(256) elements packed into 64 bits
/// \return [a_0*b_0, ..., a_7*b_7]
__host__ __device__ __forceinline__
uint64_t gf256v_mul_u64(uint64_t a,
                        const uint64_t b) {
    uint64_t r = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        r ^= a & (((b >> i) & 0x0101010101010101ull) * 0xFFull);
        a = gf256v_xtime_u64(a);
    }
    return r;
}

/// lane-wise multiplication by gamma_0 = 0x20 = x^5
__host__ __device__ __forceinline__
uint32_t gf256v_mul_gamma_u32(uint32_t a) {
#pragma unroll
    for (uint32_t i = 0; i < 5; i++) {
        a = gf256v_xtime_u32(a);
    }
    return a;
}

__host__ __device__ __forceinline__
gf256 gf256_add(const gf256 a, const gf256 b) { return a ^ b; }

__host__ __device__ __forceinline__
gf256 gf256_sub(const gf256 a, const gf256 b) { return a ^ b; }

/// \return a*b in GF(256)
__host__ __device__ __forceinline__
gf256 gf256_mul(const gf256 a,
                const gf256 b) {
    return (gf256)gf256v_mul_u32(a, b);
}

// ---------------------------------------------------------------- GF(256^2)

__host__ __device__ __forceinline__
gf256to2 gf256to2_add(const gf256to2 a, const gf256to2 b) { return a ^ b; }

__host__ __device__ __forceinline__
gf256to2 gf256to2_sub(const gf256to2 a, const gf256to2 b) { return a ^ b; }

/// \return a*b in GF(256^2). The three Karatsuba products are computed with a
///         single lane-wise multiplication.
__host__ __device__ __forceinline__
gf256to2 gf256to2_mul(const gf256to2 a,
                      const gf256to2 b) {
    const uint32_t a0 = a & 0xFFu, a1 = a >> 8u;
    const uint32_t b0 = b & 0xFFu, b1 = b >> 8u;
    // [a0*b0, a1*b1, (a0+a1)*(b0+b1), 0]
    const uint32_t p = gf256v_mul_u32(a0 | (a1 << 8u) | ((a0 ^ a1) << 16u),
                                      b0 | (b1 << 8u) | ((b0 ^ b1) << 16u));
    const uint32_t l = p & 0xFFu, h = (p >> 8u) & 0xFFu, s = (p >> 16u) & 0xFFu;
    const uint32_t lo = l ^ gf256v_mul_gamma_u32(h);
    const uint32_t hi = s ^ l;
    return (gf256to2)(lo | (hi << 8u));
}

/// \return gamma_1 * v, gamma_1 = 0x2000
__host__ __device__ __forceinline__
gf256to2 gf256to2_mul_gamma(const gf256to2 v) {
    const uint32_t v0 = v & 0xFFu, v1 = v >> 8u;
    // [gamma_0*v1, gamma_0*(v0+v1), 0, 0]
    const uint32_t y = gf256v_mul_gamma_u32(v1 | ((v0 ^ v1) << 8u));
    const uint32_t lo = gf256v_mul_gamma_u32(y & 0xFFu);
    return (gf256to2)(lo | (y & 0xFF00u));
}

// ---------------------------------------------------------------- GF(256^4)

__host__ __device__ __forceinline__
gf256to4 gf256to4_add(const gf256to4 a, const gf256to4 b) { return a ^ b; }

__host__ __device__ __forceinline__
gf256to4 gf256to4_sub(const gf256to4 a, const gf256to4 b) { return a ^ b; }

/// \return a*b in GF(256^4)
__host__ __device__ __forceinline__
gf256to4 gf256to4_mul(const gf256to4 a,
                      const gf256to4 b) {
    const gf256to2 a0 = (gf256to2)a, a1 = (gf256to2)(a >> 16u);
    const gf256to2 b0 = (gf256to2)b, b1 = (gf256to2)(b >> 16u);
    const gf256to2 l = gf256to2_mul(a0, b0);
    const gf256to2 h = gf256to2_mul(a1, b1);
    const gf256to2 s = gf256to2_mul(a0 ^ a1, b0 ^ b1);
    const gf256to4 lo = (gf256to2)(l ^ gf256to2_mul_gamma(h));
    const gf256to4 hi = (gf256to2)(s ^ l);
    return lo | (hi << 16u);
}

/// \return gamma_2 * v, gamma_2 = 0x20000000
__host__ __device__ __forceinline__
gf256to4 gf256to4_mul_gamma(const gf256to4 v) {
    const gf256to2 v0 = (gf256to2)v, v1 = (gf256to2)(v >> 16u);
    const gf256to4 hi = gf256to2_mul_gamma(v0 ^ v1);
    const gf256to4 lo = gf256to2_mul_gamma(gf256to2_mul_gamma(v1));
    return lo | (hi << 16u);
}

// ---------------------------------------------------------------- GF(256^8)

/// \return a+b
__host__ __device__ __forceinline__
gf256to8 gf256to8_add(const gf256to8 a, const gf256to8 b) { return a ^ b; }

/// \return a-b
__host__ __device__ __forceinline__
gf256to8 gf256to8_sub(const gf256to8 a, const gf256to8 b) { return a ^ b; }

/// \return a*b in GF(256^8)
__host__ __device__ __forceinline__
gf256to8 gf256to8_mul(const gf256to8 a,
                      const gf256to8 b) {
    const gf256to4 a0 = (gf256to4)a, a1 = (gf256to4)(a >> 32u);
    const gf256to4 b0 = (gf256to4)b, b1 = (gf256to4)(b >> 32u);
    const gf256to4 l = gf256to4_mul(a0, b0);
    const gf256to4 h = gf256to4_mul(a1, b1);
    const gf256to4 s = gf256to4_mul(a0 ^ a1, b0 ^ b1);
    const gf256to8 lo = l ^ gf256to4_mul_gamma(h);
    const gf256to8 hi = s ^ l;
    return lo | (hi << 32u);
}

/// \return gamma_3 * v, gamma_3 = 0x2000000000000000 (defines GF(256^16))
__host__ __device__ __forceinline__
gf256to8 gf256to8_mul_gamma(const gf256to8 v) {
    const gf256to4 v0 = (gf256to4)v, v1 = (gf256to4)(v >> 32u);
    const gf256to8 hi = gf256to4_mul_gamma(v0 ^ v1);
    const gf256to8 lo = gf256to4_mul_gamma(gf256to4_mul_gamma(v1));
    return lo | (hi << 32u);
}

// ---------------------------------------------------------------- generic matrix multiplication

/// result = A*B, column major. `F` is a field policy with
///     typename F::T                                   (trivially copyable)
///     __host__ __device__ static T F::zero()
///     __host__ __device__ static T F::add(const T&, const T&)
///     __host__ __device__ static T F::mul(const T&, const T&)
/// Thread layout: blockIdx.y (grid-stride) selects the output column j, each
/// thread computes one row i of it, so consecutive threads read consecutive
/// rows of A. Column j of B is staged through shared memory in tiles of
/// TILE_K elements.
template<typename F, uint32_t TILE_K>
__global__ void matrix_mul_tiled_kernel(typename F::T *__restrict__ result,
                                        const typename F::T *__restrict__ matrix1,
                                        const typename F::T *__restrict__ matrix2,
                                        const uint32_t n_rows1,
                                        const uint32_t n_cols1,
                                        const uint32_t n_cols2) {
    typedef typename F::T T;
    __shared__ __align__(16) unsigned char smem[TILE_K * sizeof(T)];
    T *sB = reinterpret_cast<T *>(smem);

    for (uint32_t j = blockIdx.y; j < n_cols2; j += gridDim.y) {
        const T *Bj = matrix2 + (size_t)j * n_cols1;

        // the bounds of both loops only depend on the block, so every thread
        // of a block reaches the __syncthreads() below
        for (uint32_t i0 = blockIdx.x * blockDim.x; i0 < n_rows1; i0 += gridDim.x * blockDim.x) {
            const uint32_t i = i0 + threadIdx.x;
            const bool valid = i < n_rows1;

            T acc = F::zero();
            for (uint32_t k0 = 0; k0 < n_cols1; k0 += TILE_K) {
                const uint32_t kn = min(TILE_K, n_cols1 - k0);

                __syncthreads(); // previous tile is consumed
                for (uint32_t t = threadIdx.x; t < kn; t += blockDim.x) {
                    sB[t] = Bj[k0 + t];
                }
                __syncthreads();

                if (valid) {
                    const T *Ak = matrix1 + (size_t)k0 * n_rows1 + i;
                    for (uint32_t k = 0; k < kn; k++, Ak += n_rows1) {
                        acc = F::add(acc, F::mul(*Ak, sB[k]));
                    }
                }
            }

            if (valid) {
                result[(size_t)j * n_rows1 + i] = acc;
            }
        }
    }
}

/// enqueues result = matrix1 * matrix2 (column major) using
/// `matrix_mul_tiled_kernel`. result must not alias the inputs.
template<typename F>
static inline
void matrix_mul_tiled(typename F::T *result,
                      const typename F::T *matrix1,
                      const typename F::T *matrix2,
                      const uint32_t n_rows1,
                      const uint32_t n_cols1,
                      const uint32_t n_cols2,
                      cudaStream_t stream) {
    if ((n_rows1 == 0) || (n_cols2 == 0)) {
        return;
    }

    // 4 KiB of shared memory per block
    constexpr uint32_t tile_k = (uint32_t)(4096u / sizeof(typename F::T));
    static_assert(tile_k > 0, "element type too large");

    const uint32_t threads = 128;
    uint32_t gx = (n_rows1 + threads - 1) / threads;
    gx = gx < 65535u ? gx : 65535u;
    const uint32_t gy = n_cols2 < 65535u ? n_cols2 : 65535u;

    matrix_mul_tiled_kernel<F, tile_k><<<dim3(gx, gy), threads, 0, stream>>>
        (result, matrix1, matrix2, n_rows1, n_cols1, n_cols2);
    FQ_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------- vectors/matrices over GF(256^8)

/// field policy for the generic kernels
struct gf256to8_field {
    typedef gf256to8 T;
    __host__ __device__ static T zero() { return 0; }
    __host__ __device__ static T add(const T &a, const T &b) { return gf256to8_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256to8_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256to8_mul(a, b); }
};

struct gf256to8_add_op {
    __device__ gf256to8 operator()(const gf256to8 &a, const gf256to8 &b) const { return gf256to8_add(a, b); }
};
struct gf256to8_sub_op {
    __device__ gf256to8 operator()(const gf256to8 &a, const gf256to8 &b) const { return gf256to8_sub(a, b); }
};
struct gf256to8_mul_op {
    __device__ gf256to8 operator()(const gf256to8 &a, const gf256to8 &b) const { return gf256to8_mul(a, b); }
};

/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf256to8_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf256to8);
}

/// \return number of bytes of a nrows x ncols matrix
__host__ __device__ __forceinline__
size_t gf256to8_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return gf256to8_matrix_bytes_per_column(nrows) * ncols;
}

/// c = a + b element-wise, n = number of elements, device pointers
static inline
void gf256to8_vector_add(gf256to8 *c, const gf256to8 *a, const gf256to8 *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to8_add_op{}, stream);
}

/// c = a - b element-wise, n = number of elements, device pointers
static inline
void gf256to8_vector_sub(gf256to8 *c, const gf256to8 *a, const gf256to8 *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to8_sub_op{}, stream);
}

/// c = a * b element-wise, n = number of elements, device pointers
static inline
void gf256to8_vector_mul(gf256to8 *c, const gf256to8 *a, const gf256to8 *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256to8_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, nrows x ncols, device pointers
static inline
void gf256to8_matrix_add(gf256to8 *matrix1, const gf256to8 *matrix2, const gf256to8 *matrix3,
                         const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to8_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, nrows x ncols, device pointers
static inline
void gf256to8_matrix_sub(gf256to8 *matrix1, const gf256to8 *matrix2, const gf256to8 *matrix3,
                         const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256to8_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, device pointers
/// \param result[out]: n_rows1 x n_cols2, must not alias the inputs
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf256to8_matrix_mul(gf256to8 *result, const gf256to8 *matrix1, const gf256to8 *matrix2,
                         const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2,
                         cudaStream_t stream = 0) {
    matrix_mul_tiled<gf256to8_field>(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256to8_cuda
