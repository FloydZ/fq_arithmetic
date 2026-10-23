#pragma once

/// CUDA implementation of GF(256^X) = GF(256)[X]/(X^D + X^5 + X^2 + X + 9),
/// D = EXTENSION_DEGREE, the same representation as `arith.h`.
///
/// Like the C code, the extension degree is the macro `EXTENSION_DEGREE`
/// (default 48, the value of `arith.h`; including `arith.h` first or passing
/// -DEXTENSION_DEGREE=... selects another one). The reduction polynomial is the
/// one of the C code, which is only known to be irreducible for D = 48.
///
/// An element is stored exactly like the C type `uint8_t[EXTENSION_DEGREE]`:
/// coefficient i (of X^i) in byte i, each coefficient in the AES field
/// GF(256) = GF(2)[x]/(x^8 + x^4 + x^3 + x + 1). If D is a multiple of 8,
/// `gf256toX` has an alignment of 8, so device pointers to vectors/matrices
/// must be 8-byte aligned (cudaMalloc is).
///
/// Vectors/matrices: plain arrays of elements, matrices column-major
/// (entry (i, j) at j*nrows + i). All host wrappers take DEVICE pointers,
/// enqueue on `stream` and do not synchronize.

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "../helper.cuh"
#include "../256to8/arith.cu"

#ifndef EXTENSION_DEGREE
#define EXTENSION_DEGREE 48
#endif

namespace gf256toX_cuda {

/// extension degree D
constexpr uint32_t GF256TOX_DEGREE = EXTENSION_DEGREE;
static_assert(GF256TOX_DEGREE > 5, "the reduction needs D > deg(X^5 + X^2 + X + 9)");

/// number of 64-bit words holding the D coefficients of an element
constexpr uint32_t GF256TOX_WORDS = (GF256TOX_DEGREE + 7u) / 8u;

/// number of terms of the reduction polynomial below X^D
constexpr uint32_t GF256TOX_RED_TERMS = 4;

/// X^D = sum_j coef(j) * X^pol(j): X^5 + X^2 + X + 9, as in `arith.h`
__host__ __device__ constexpr uint32_t gf256toX_red_pol(const uint32_t j) {
    return j == 0 ? 0u : (j == 1 ? 1u : (j == 2 ? 2u : 5u));
}
__host__ __device__ constexpr uint32_t gf256toX_red_coef(const uint32_t j) {
    return j == 0 ? 9u : 1u;
}

struct alignas(GF256TOX_DEGREE % 8u == 0 ? 8 : 1) gf256toX {
    uint8_t v[GF256TOX_DEGREE];
};
static_assert(sizeof(gf256toX) == GF256TOX_DEGREE, "must match the C type uint8_t[EXTENSION_DEGREE]");

/// \return a+b
__host__ __device__ __forceinline__
gf256toX gf256toX_add(const gf256toX &a,
                      const gf256toX &b) {
    gf256toX r;
#pragma unroll
    for (uint32_t i = 0; i < GF256TOX_DEGREE; i++) {
        r.v[i] = a.v[i] ^ b.v[i];
    }
    return r;
}

/// \return a-b
__host__ __device__ __forceinline__
gf256toX gf256toX_sub(const gf256toX &a,
                      const gf256toX &b) {
    return gf256toX_add(a, b);
}

/// Schoolbook multiplication followed by the reduction of `arith.h`.
/// The coefficients are processed 8 at a time in 64-bit words: with
/// xa[t] = a*x^t (t = 0..7, lane-wise in GF(256)), a*b_j is the XOR of the
/// xa[t] for which bit t of b_j is set, and is added to the product shifted
/// by j bytes. Constant time and table free.
/// \return a*b
__host__ __device__ __forceinline__
gf256toX gf256toX_mul(const gf256toX &a,
                      const gf256toX &b) {
    using namespace gf256to8_cuda;
    constexpr uint32_t D = GF256TOX_DEGREE;
    constexpr uint32_t W = GF256TOX_WORDS;
    constexpr uint32_t PW = 2u * W; // > W + (D-1)/8, so `p[w + q + 1]` is in range

    // xa[t][w] = a*x^t
    uint64_t xa[8][W];
    {
        uint64_t aw[W];
#pragma unroll
        for (uint32_t w = 0; w < W; w++) { aw[w] = 0; }
        memcpy(aw, a.v, D);

#pragma unroll
        for (uint32_t t = 0; t < 8; t++) {
#pragma unroll
            for (uint32_t w = 0; w < W; w++) {
                xa[t][w] = aw[w];
                aw[w] = gf256v_xtime_u64(aw[w]);
            }
        }
    }

    // p = a*b, unreduced, 2D-1 coefficients
    uint64_t p[PW];
#pragma unroll
    for (uint32_t w = 0; w < PW; w++) { p[w] = 0; }

#pragma unroll
    for (uint32_t j = 0; j < D; j++) {
        const uint32_t bj = b.v[j];
        const uint32_t q = j >> 3u, sh = 8u * (j & 7u);
#pragma unroll
        for (uint32_t w = 0; w < W; w++) {
            uint64_t m = 0;
#pragma unroll
            for (uint32_t t = 0; t < 8; t++) {
                m ^= xa[t][w] & (0ull - (uint64_t)((bj >> t) & 1u));
            }
            p[w + q] ^= m << sh;
            if (sh) {
                p[w + q + 1] ^= m >> (64u - sh);
            }
        }
    }

    // reduction, coefficient by coefficient from the top, exactly as in `arith.h`
#pragma unroll
    for (uint32_t k = 2u * D - 2u; k >= D; k--) {
        const uint32_t c = (uint32_t)(p[k >> 3u] >> (8u * (k & 7u))) & 0xFFu;
#pragma unroll
        for (uint32_t j = 0; j < GF256TOX_RED_TERMS; j++) {
            const uint32_t idx = k - (D - gf256toX_red_pol(j));
            p[idx >> 3u] ^= (uint64_t)gf256_mul((gf256)c, (gf256)gf256toX_red_coef(j)) << (8u * (idx & 7u));
        }
    }

    gf256toX r;
    memcpy(r.v, p, D);
    return r;
}

/// field policy for the generic kernels
struct gf256toX_field {
    typedef gf256toX T;
    __host__ __device__ static T zero() {
        T r;
#pragma unroll
        for (uint32_t i = 0; i < GF256TOX_DEGREE; i++) { r.v[i] = 0; }
        return r;
    }
    __host__ __device__ static T add(const T &a, const T &b) { return gf256toX_add(a, b); }
    __host__ __device__ static T sub(const T &a, const T &b) { return gf256toX_sub(a, b); }
    __host__ __device__ static T mul(const T &a, const T &b) { return gf256toX_mul(a, b); }
};

struct gf256toX_add_op {
    __device__ gf256toX operator()(const gf256toX &a, const gf256toX &b) const { return gf256toX_add(a, b); }
};
struct gf256toX_sub_op {
    __device__ gf256toX operator()(const gf256toX &a, const gf256toX &b) const { return gf256toX_sub(a, b); }
};
struct gf256toX_mul_op {
    __device__ gf256toX operator()(const gf256toX &a, const gf256toX &b) const { return gf256toX_mul(a, b); }
};

/// \return number of bytes of a single column
__host__ __device__ __forceinline__
size_t gf256toX_matrix_bytes_per_column(const uint32_t nrows) {
    return (size_t)nrows * sizeof(gf256toX);
}

/// \return number of bytes of a nrows x ncols matrix
__host__ __device__ __forceinline__
size_t gf256toX_matrix_bytes_size(const uint32_t nrows,
                                  const uint32_t ncols) {
    return gf256toX_matrix_bytes_per_column(nrows) * ncols;
}

/// c = a + b element-wise, n = number of elements, device pointers
static inline
void gf256toX_vector_add(gf256toX *c, const gf256toX *a, const gf256toX *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256toX_add_op{}, stream);
}

/// c = a - b element-wise, n = number of elements, device pointers
static inline
void gf256toX_vector_sub(gf256toX *c, const gf256toX *a, const gf256toX *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256toX_sub_op{}, stream);
}

/// c = a * b element-wise, n = number of elements, device pointers
static inline
void gf256toX_vector_mul(gf256toX *c, const gf256toX *a, const gf256toX *b,
                         const uint32_t n, cudaStream_t stream = 0) {
    fq_cuda::elementwise(c, a, b, n, gf256toX_mul_op{}, stream);
}

/// matrix1 = matrix2 + matrix3, nrows x ncols, device pointers
static inline
void gf256toX_matrix_add(gf256toX *matrix1, const gf256toX *matrix2, const gf256toX *matrix3,
                         const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256toX_add_op{}, stream);
}

/// matrix1 = matrix2 - matrix3, nrows x ncols, device pointers
static inline
void gf256toX_matrix_sub(gf256toX *matrix1, const gf256toX *matrix2, const gf256toX *matrix3,
                         const uint32_t nrows, const uint32_t ncols, cudaStream_t stream = 0) {
    fq_cuda::elementwise(matrix1, matrix2, matrix3, (size_t)nrows * ncols, gf256toX_sub_op{}, stream);
}

/// result = matrix1 * matrix2, column major, device pointers
/// \param result[out]: n_rows1 x n_cols2, must not alias the inputs
/// \param matrix1[in]: n_rows1 x n_cols1
/// \param matrix2[in]: n_cols1 x n_cols2
static inline
void gf256toX_matrix_mul(gf256toX *result, const gf256toX *matrix1, const gf256toX *matrix2,
                         const uint32_t n_rows1, const uint32_t n_cols1, const uint32_t n_cols2,
                         cudaStream_t stream = 0) {
    gf256to8_cuda::matrix_mul_tiled<gf256toX_field>(result, matrix1, matrix2, n_rows1, n_cols1, n_cols2, stream);
}

} // namespace gf256toX_cuda
